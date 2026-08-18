//! Shared model loading, prompt rendering, sampling, and token generation.

use std::{
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Instant,
};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc as async_mpsc;

use crate::{
    config::Paths,
    gemma4::{
        ChatMessage, Gemma4Tokenizer,
        cpu::{
            CancellationFlag, Gemma4CpuModel, Gemma4CpuSession, GenerationProfile,
            GenerationResult, RuntimeError,
        },
        render_chat,
    },
    glm52::{
        cpu::{Glm52CpuModel, Glm52CpuSession, Glm52ForwardProfile},
        tokenizer::render_chat as render_glm52_chat,
    },
    model::{ModelCatalog, ModelError},
};

#[cfg(target_os = "macos")]
use crate::gemma4::metal::{ExpertResidency, Gemma4MetalModel, Gemma4MetalSession};

pub const DEFAULT_CONTEXT: usize = 4_096;
pub const DEFAULT_MAXIMUM_NEW_TOKENS: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error(transparent)]
    Model(#[from] ModelError),

    #[error("model {name:?} has no local tensor path")]
    MissingModelPath { name: String },

    #[error("model {name:?} uses unsupported architecture {architecture:?}")]
    UnsupportedArchitecture { name: String, architecture: String },

    #[error("loaded-model limit of {limit} has been reached")]
    ModelLimit { limit: usize },

    #[error("request queue for model {model:?} is full")]
    QueueFull { model: String },

    #[error("generation worker for model {model:?} stopped unexpectedly")]
    WorkerStopped { model: String },

    #[error("could not start generation worker for model {model:?}: {source}")]
    SpawnWorker {
        model: String,
        source: std::io::Error,
    },

    #[error("invalid runtime limits: {0}")]
    InvalidLimits(String),

    #[error("invalid generation request: {0}")]
    InvalidRequest(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeLimits {
    pub context: usize,
    pub maximum_loaded_models: usize,
    pub request_queue: usize,
    pub token_buffer: usize,
    pub maximum_sessions_per_model: usize,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            context: DEFAULT_CONTEXT,
            maximum_loaded_models: 1,
            request_queue: 8,
            token_buffer: 32,
            maximum_sessions_per_model: 2,
        }
    }
}

impl RuntimeLimits {
    pub fn from_environment() -> Result<Self, ServiceError> {
        Self {
            context: bounded_environment("DISKMULE_CONTEXT", DEFAULT_CONTEXT, 1, 262_144)?,
            maximum_loaded_models: bounded_environment("DISKMULE_MAX_MODELS", 1, 1, 16)?,
            request_queue: bounded_environment("DISKMULE_REQUEST_QUEUE", 8, 1, 1_024)?,
            token_buffer: bounded_environment("DISKMULE_TOKEN_BUFFER", 32, 1, 4_096)?,
            maximum_sessions_per_model: bounded_environment("DISKMULE_MAX_SESSIONS", 2, 1, 16)?,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, ServiceError> {
        if !(1..=262_144).contains(&self.context) {
            return Err(ServiceError::InvalidLimits(
                "context must be in 1..=262144".to_owned(),
            ));
        }
        if !(1..=16).contains(&self.maximum_loaded_models) {
            return Err(ServiceError::InvalidLimits(
                "maximum_loaded_models must be in 1..=16".to_owned(),
            ));
        }
        if !(1..=1_024).contains(&self.request_queue) {
            return Err(ServiceError::InvalidLimits(
                "request_queue must be in 1..=1024".to_owned(),
            ));
        }
        if !(1..=4_096).contains(&self.token_buffer) {
            return Err(ServiceError::InvalidLimits(
                "token_buffer must be in 1..=4096".to_owned(),
            ));
        }
        if !(1..=16).contains(&self.maximum_sessions_per_model) {
            return Err(ServiceError::InvalidLimits(
                "maximum_sessions_per_model must be in 1..=16".to_owned(),
            ));
        }
        Ok(self)
    }
}

fn bounded_environment(
    name: &'static str,
    default: usize,
    minimum: usize,
    maximum: usize,
) -> Result<usize, ServiceError> {
    let Some(raw) = env::var_os(name) else {
        return Ok(default);
    };
    raw.to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            ServiceError::InvalidLimits(format!(
                "{name} must be an integer in {minimum}..={maximum}"
            ))
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal {
        residency: ExpertResidency,
    },
}

impl BackendSelection {
    pub fn from_environment() -> Result<Self, RuntimeError> {
        let requested = env::var("DISKMULE_BACKEND").unwrap_or_else(|_| "auto".to_owned());
        match requested.as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto" => {
                #[cfg(target_os = "macos")]
                {
                    Ok(Self::Metal {
                        residency: expert_residency_from_environment()?,
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Ok(Self::Cpu)
                }
            }
            "metal" => {
                #[cfg(target_os = "macos")]
                {
                    Ok(Self::Metal {
                        residency: expert_residency_from_environment()?,
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(RuntimeError::InvalidConfiguration(
                        "DISKMULE_BACKEND=metal requires macOS".to_owned(),
                    ))
                }
            }
            _ => Err(RuntimeError::InvalidConfiguration(
                "DISKMULE_BACKEND must be auto, metal, or cpu".to_owned(),
            )),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Cpu => "CPU".to_owned(),
            #[cfg(target_os = "macos")]
            Self::Metal {
                residency: ExpertResidency::Mapped,
            } => "Metal".to_owned(),
            #[cfg(target_os = "macos")]
            Self::Metal {
                residency: ExpertResidency::Streamed { slots_per_layer },
            } => format!("Metal streamed ({slots_per_layer} expert slots/layer)"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingConfig {
    /// Zero selects deterministic greedy decoding.
    pub temperature: f32,
    /// Zero disables top-k filtering.
    pub top_k: usize,
    /// Probability mass retained after top-k filtering.
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.9,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    pub fn validate(self) -> Result<Self, RuntimeError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 || self.temperature > 10.0 {
            return Err(RuntimeError::InvalidConfiguration(
                "temperature must be finite and in 0..=10".to_owned(),
            ));
        }
        if !(self.top_p.is_finite() && 0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(RuntimeError::InvalidConfiguration(
                "top_p must be finite and in (0, 1]".to_owned(),
            ));
        }
        if self.top_k > 1_000_000 {
            return Err(RuntimeError::InvalidConfiguration(
                "top_k must be at most 1000000".to_owned(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct GenerationOptions {
    pub maximum_new_tokens: usize,
    #[serde(flatten)]
    pub sampling: SamplingConfig,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            maximum_new_tokens: DEFAULT_MAXIMUM_NEW_TOKENS,
            sampling: SamplingConfig::default(),
        }
    }
}

impl GenerationOptions {
    pub fn from_environment() -> Result<Self, RuntimeError> {
        Self {
            maximum_new_tokens: runtime_usize_environment(
                "DISKMULE_MAX_TOKENS",
                DEFAULT_MAXIMUM_NEW_TOKENS,
                1,
                4_096,
            )?,
            sampling: SamplingConfig {
                temperature: runtime_f32_environment(
                    "DISKMULE_TEMPERATURE",
                    SamplingConfig::default().temperature,
                )?,
                top_k: runtime_usize_environment(
                    "DISKMULE_TOP_K",
                    SamplingConfig::default().top_k,
                    0,
                    1_000_000,
                )?,
                top_p: runtime_f32_environment("DISKMULE_TOP_P", SamplingConfig::default().top_p)?,
                seed: runtime_u64_environment("DISKMULE_SEED", SamplingConfig::default().seed)?,
            },
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, RuntimeError> {
        if !(1..=4_096).contains(&self.maximum_new_tokens) {
            return Err(RuntimeError::InvalidConfiguration(
                "maximum_new_tokens must be in 1..=4096".to_owned(),
            ));
        }
        self.sampling.validate()?;
        Ok(self)
    }
}

pub struct GenerationEngine {
    name: String,
    path: PathBuf,
    backend: BackendSelection,
    model: ArchitectureModel,
}

enum ArchitectureModel {
    Gemma4Cpu(Box<Gemma4CpuModel>),
    Glm52Cpu(Box<Glm52CpuModel>),
    #[cfg(target_os = "macos")]
    Gemma4Metal(Box<Gemma4MetalModel>),
}

enum ArchitectureSession {
    Gemma4Cpu(Gemma4CpuSession),
    Glm52Cpu(Glm52CpuSession),
    #[cfg(target_os = "macos")]
    Gemma4Metal(Gemma4MetalSession),
}

struct ResidentSession {
    state: ArchitectureSession,
    last_used: u64,
}

impl GenerationEngine {
    pub fn open(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        architecture: &str,
        context: usize,
        backend: BackendSelection,
    ) -> Result<Self, RuntimeError> {
        match architecture {
            "gemma4" => Self::open_gemma4(name, path, context, backend),
            "glm_moe_dsa" => Self::open_glm52(name, path, context, backend),
            other => Err(RuntimeError::InvalidConfiguration(format!(
                "unsupported architecture {other:?}"
            ))),
        }
    }

    pub fn open_gemma4(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        context: usize,
        backend: BackendSelection,
    ) -> Result<Self, RuntimeError> {
        let path = path.as_ref().to_path_buf();
        let model = match backend {
            BackendSelection::Cpu => {
                ArchitectureModel::Gemma4Cpu(Box::new(Gemma4CpuModel::open(&path, context)?))
            }
            #[cfg(target_os = "macos")]
            BackendSelection::Metal { residency } => ArchitectureModel::Gemma4Metal(Box::new(
                Gemma4MetalModel::open_with_expert_residency(&path, context, residency)?,
            )),
        };
        Ok(Self {
            name: name.into(),
            path,
            backend,
            model,
        })
    }

    pub fn open_glm52(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        context: usize,
        backend: BackendSelection,
    ) -> Result<Self, RuntimeError> {
        let path = path.as_ref().to_path_buf();
        let model = match backend {
            BackendSelection::Cpu => Glm52CpuModel::from_directory_with_context(&path, context)?,
            #[cfg(target_os = "macos")]
            BackendSelection::Metal { .. } => {
                Glm52CpuModel::from_directory_with_context_and_metal(&path, context)?
            }
        };
        tracing::info!(
            model = %path.display(),
            backend = %backend.label(),
            read_cache = model.read_cache_policy().label(),
            "loaded GLM architecture"
        );
        Ok(Self {
            name: name.into(),
            path,
            backend,
            model: ArchitectureModel::Glm52Cpu(Box::new(model)),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn backend(&self) -> BackendSelection {
        self.backend
    }

    fn gemma4_tokenizer(&self) -> Option<&Gemma4Tokenizer> {
        match &self.model {
            ArchitectureModel::Gemma4Cpu(model) => Some(&model.tokenizer),
            ArchitectureModel::Glm52Cpu(_) => None,
            #[cfg(target_os = "macos")]
            ArchitectureModel::Gemma4Metal(model) => Some(&model.tokenizer),
        }
    }

    pub fn generate_chat<F>(
        &self,
        messages: &[ChatMessage],
        options: GenerationOptions,
        cancellation: &CancellationFlag,
        on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
    {
        let options = options.validate()?;
        let tokens = match &self.model {
            ArchitectureModel::Glm52Cpu(model) => model
                .tokenizer
                .encode(&render_glm52_chat(messages, false))?,
            _ => {
                let rendered = render_chat(messages)?;
                self.gemma4_tokenizer()
                    .expect("Gemma architecture has a Gemma tokenizer")
                    .encode(&rendered, false)
            }
        };
        self.generate_tokens(&tokens, options, cancellation, on_token)
    }

    pub fn generate_tokens<F>(
        &self,
        prompt_tokens: &[u32],
        options: GenerationOptions,
        cancellation: &CancellationFlag,
        on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
    {
        let options = options.validate()?;
        let mut sampler = Sampler::new(options.sampling)?;
        match &self.model {
            ArchitectureModel::Gemma4Cpu(model) => model.generate_with_selector(
                prompt_tokens,
                options.maximum_new_tokens,
                cancellation,
                |logits| sampler.select(logits),
                on_token,
            ),
            ArchitectureModel::Glm52Cpu(model) => {
                let mut session = model.new_session();
                generate_glm52_session(
                    model,
                    &mut session,
                    prompt_tokens,
                    options,
                    cancellation,
                    &mut sampler,
                    on_token,
                )
            }
            #[cfg(target_os = "macos")]
            ArchitectureModel::Gemma4Metal(model) => model.generate_with_selector(
                prompt_tokens,
                options.maximum_new_tokens,
                cancellation,
                |logits| sampler.select(logits),
                on_token,
            ),
        }
    }

    fn new_session(&self) -> Result<ArchitectureSession, RuntimeError> {
        match &self.model {
            ArchitectureModel::Gemma4Cpu(model) => {
                Ok(ArchitectureSession::Gemma4Cpu(model.new_session()))
            }
            ArchitectureModel::Glm52Cpu(model) => {
                Ok(ArchitectureSession::Glm52Cpu(model.new_session()))
            }
            #[cfg(target_os = "macos")]
            ArchitectureModel::Gemma4Metal(model) => {
                Ok(ArchitectureSession::Gemma4Metal(model.new_session()?))
            }
        }
    }

    fn generate_chat_in_session<F>(
        &self,
        session: &mut ArchitectureSession,
        messages: &[ChatMessage],
        options: GenerationOptions,
        cancellation: &CancellationFlag,
        on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
    {
        let options = options.validate()?;
        let tokens = match &self.model {
            ArchitectureModel::Glm52Cpu(model) => model
                .tokenizer
                .encode(&render_glm52_chat(messages, false))?,
            _ => {
                let rendered = render_chat(messages)?;
                self.gemma4_tokenizer()
                    .expect("Gemma architecture has a Gemma tokenizer")
                    .encode(&rendered, false)
            }
        };
        let mut sampler = Sampler::new(options.sampling)?;
        match (&self.model, session) {
            (ArchitectureModel::Gemma4Cpu(model), ArchitectureSession::Gemma4Cpu(session)) => model
                .generate_session_with_selector(
                    session,
                    &tokens,
                    options.maximum_new_tokens,
                    cancellation,
                    |logits| sampler.select(logits),
                    on_token,
                ),
            (ArchitectureModel::Glm52Cpu(model), ArchitectureSession::Glm52Cpu(session)) => {
                generate_glm52_session(
                    model,
                    session,
                    &tokens,
                    options,
                    cancellation,
                    &mut sampler,
                    on_token,
                )
            }
            #[cfg(target_os = "macos")]
            (ArchitectureModel::Gemma4Metal(model), ArchitectureSession::Gemma4Metal(session)) => {
                model.generate_session_with_selector(
                    session,
                    &tokens,
                    options.maximum_new_tokens,
                    cancellation,
                    |logits| sampler.select(logits),
                    on_token,
                )
            }
            #[allow(unreachable_patterns)]
            _ => Err(RuntimeError::InvalidConfiguration(
                "session architecture does not match its loaded model".to_owned(),
            )),
        }
    }
}

fn generate_glm52_session<F>(
    model: &Glm52CpuModel,
    session: &mut Glm52CpuSession,
    prompt_tokens: &[u32],
    options: GenerationOptions,
    cancellation: &CancellationFlag,
    sampler: &mut Sampler,
    mut on_token: F,
) -> Result<GenerationResult, RuntimeError>
where
    F: FnMut(u32, &str),
{
    if prompt_tokens.is_empty() {
        return Err(RuntimeError::EmptyPrompt);
    }
    let requested = prompt_tokens
        .len()
        .checked_add(options.maximum_new_tokens)
        .ok_or(RuntimeError::ContextLimit {
            requested: usize::MAX,
            limit: session.maximum_context(),
        })?;
    if requested > session.maximum_context() {
        return Err(RuntimeError::ContextLimit {
            requested,
            limit: session.maximum_context(),
        });
    }

    let reusable = session
        .tokens
        .iter()
        .zip(prompt_tokens)
        .take_while(|(left, right)| left == right)
        .count();
    let stable_logits = (reusable == session.tokens.len())
        .then(|| session.logits.clone())
        .flatten();
    session.truncate(reusable);
    session.logits = stable_logits.clone();
    let cache_before = model.expert_cache_stats()?;

    let result = (|| {
        let started = Instant::now();
        let mut profile = GenerationProfile {
            prompt_tokens: prompt_tokens.len(),
            reused_prompt_tokens: reusable,
            ..GenerationProfile::default()
        };
        let mut forward_profile = Glm52ForwardProfile::default();
        let prefill_started = Instant::now();
        let mut logits = if reusable == prompt_tokens.len() {
            session.logits.clone().ok_or_else(|| {
                RuntimeError::InvalidConfiguration(
                    "GLM session is missing logits for its cached prompt".to_owned(),
                )
            })?
        } else {
            let mut final_logits = None;
            for token in prompt_tokens.iter().copied().skip(reusable) {
                if cancellation.is_cancelled() {
                    return Err(RuntimeError::Cancelled);
                }
                let output = model.forward_token_profiled(
                    session,
                    token,
                    cancellation.as_atomic(),
                    &mut forward_profile,
                )?;
                session.tokens.push(token);
                session.logits = Some(output.clone());
                final_logits = Some(output);
            }
            final_logits.expect("non-empty GLM prompt suffix produces logits")
        };
        profile.prefill_time = prefill_started.elapsed();

        let mut generated = Vec::with_capacity(options.maximum_new_tokens);
        let mut pending_utf8 = Vec::new();
        let mut stopped = false;
        for generation_index in 0..options.maximum_new_tokens {
            if cancellation.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let next = sampler.select(&logits)?;
            if next as usize >= logits.len() {
                return Err(RuntimeError::InvalidToken {
                    token: next,
                    vocabulary: logits.len(),
                });
            }
            if generation_index == 0 {
                profile.time_to_first_token = started.elapsed();
            }
            if model.config.stop_token_ids.contains(&next) || model.tokenizer.is_special(next) {
                stopped = true;
                break;
            }
            generated.push(next);
            profile.generated_tokens += 1;
            pending_utf8.extend(model.tokenizer.decode_token_bytes(next, false)?);
            let piece = drain_valid_utf8(&mut pending_utf8, false);
            if !piece.is_empty() {
                on_token(next, &piece);
            }
            if generation_index + 1 < options.maximum_new_tokens {
                let decode_started = Instant::now();
                logits = model.forward_token_profiled(
                    session,
                    next,
                    cancellation.as_atomic(),
                    &mut forward_profile,
                )?;
                session.tokens.push(next);
                session.logits = Some(logits.clone());
                profile.decode_time += decode_started.elapsed();
            }
        }
        if !pending_utf8.is_empty() {
            let piece = drain_valid_utf8(&mut pending_utf8, true);
            if !piece.is_empty() {
                on_token(
                    *generated.last().expect("pending bytes have a token"),
                    &piece,
                );
            }
        }
        profile.mapped_bytes_touched = forward_profile.mapped_bytes_touched;
        profile.resident_kv_bytes = session.resident_state_bytes();
        profile.embedding_time = forward_profile.embedding_time;
        profile.attention_time = forward_profile.attention_time;
        profile.feed_forward_time = forward_profile.feed_forward_time;
        profile.output_time = forward_profile.output_time;
        let cache_after = model.expert_cache_stats()?;
        profile.expert_cache_hits = cache_after.hits.saturating_sub(cache_before.hits);
        profile.expert_cache_misses = cache_after.misses.saturating_sub(cache_before.misses);
        profile.expert_cache_evictions =
            cache_after.evictions.saturating_sub(cache_before.evictions);
        profile.expert_reads = cache_after.reads.saturating_sub(cache_before.reads);
        profile.expert_bytes_read = cache_after
            .bytes_read
            .saturating_sub(cache_before.bytes_read);
        profile.expert_io_wait = cache_after
            .io_wait
            .checked_sub(cache_before.io_wait)
            .unwrap_or_default();
        profile.expert_resident_bytes = cache_after.resident_bytes;
        profile.total_time = started.elapsed();
        Ok(GenerationResult {
            text: model.tokenizer.decode(&generated, false)?,
            token_ids: generated,
            stopped,
            profile,
        })
    })();
    if result.is_err() {
        session.truncate(reusable);
        session.logits = stable_logits;
    }
    result
}

fn drain_valid_utf8(bytes: &mut Vec<u8>, final_piece: bool) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => {
            let output = text.to_owned();
            bytes.clear();
            output
        }
        Err(error) if error.valid_up_to() > 0 => {
            let valid = error.valid_up_to();
            let output = String::from_utf8(bytes[..valid].to_vec())
                .expect("UTF-8 validator identified a valid prefix");
            bytes.drain(..valid);
            output
        }
        Err(_) if final_piece => {
            let output = String::from_utf8_lossy(bytes).into_owned();
            bytes.clear();
            output
        }
        Err(_) => String::new(),
    }
}

#[derive(Clone)]
pub struct RuntimeService {
    inner: Arc<RuntimeServiceInner>,
}

struct RuntimeServiceInner {
    paths: Paths,
    ollama_root: Option<PathBuf>,
    backend: BackendSelection,
    limits: RuntimeLimits,
    workers: Mutex<HashMap<PathBuf, ModelWorker>>,
}

struct ModelWorker {
    name: String,
    architecture: String,
    backend: String,
    sender: mpsc::SyncSender<GenerationJob>,
    status: Arc<Mutex<WorkerStatus>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for RuntimeServiceInner {
    fn drop(&mut self) {
        let workers = self
            .workers
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let drained = workers
            .drain()
            .map(|(_, worker)| worker)
            .collect::<Vec<_>>();
        let mut threads = Vec::with_capacity(drained.len());
        for mut worker in drained {
            drop(worker.sender);
            if let Some(thread) = worker.thread.take() {
                threads.push(thread);
            }
        }
        for thread in threads {
            if thread.join().is_err() {
                tracing::warn!("a DiskMule model worker panicked during shutdown");
            }
        }
    }
}

struct GenerationJob {
    session_id: Option<String>,
    messages: Vec<ChatMessage>,
    options: GenerationOptions,
    cancellation: CancellationFlag,
    events: async_mpsc::Sender<GenerationEvent>,
}

#[derive(Debug)]
pub enum GenerationEvent {
    Token { id: u32, text: String },
    Complete(Box<GenerationResult>),
    Error { message: String, cancelled: bool },
}

pub struct GenerationTicket {
    events: async_mpsc::Receiver<GenerationEvent>,
    cancellation: CancellationFlag,
}

impl GenerationTicket {
    pub async fn recv(&mut self) -> Option<GenerationEvent> {
        self.events.recv().await
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

impl Drop for GenerationTicket {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStatus {
    Loading,
    Ready,
    Busy,
    Failed,
    Stopped,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LoadedModelInfo {
    pub name: String,
    pub architecture: String,
    pub backend: String,
    pub status: WorkerStatus,
}

impl RuntimeService {
    pub fn new(
        paths: Paths,
        ollama_root: Option<PathBuf>,
        backend: BackendSelection,
        limits: RuntimeLimits,
    ) -> Result<Self, ServiceError> {
        Ok(Self {
            inner: Arc::new(RuntimeServiceInner {
                paths,
                ollama_root,
                backend,
                limits: limits.validate()?,
                workers: Mutex::new(HashMap::new()),
            }),
        })
    }

    pub fn limits(&self) -> RuntimeLimits {
        self.inner.limits
    }

    pub fn backend(&self) -> BackendSelection {
        self.inner.backend
    }

    pub fn loaded_models(&self) -> Vec<LoadedModelInfo> {
        let workers = lock_unpoisoned(&self.inner.workers);
        let mut models = workers
            .values()
            .map(|worker| LoadedModelInfo {
                name: worker.name.clone(),
                architecture: worker.architecture.clone(),
                backend: worker.backend.clone(),
                status: *lock_unpoisoned(&worker.status),
            })
            .collect::<Vec<_>>();
        models.sort_by(|left, right| left.name.cmp(&right.name));
        models
    }

    pub fn generate(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        options: GenerationOptions,
    ) -> Result<GenerationTicket, ServiceError> {
        self.generate_with_session(model, None, messages, options)
    }

    pub fn generate_in_session(
        &self,
        model: &str,
        session_id: impl Into<String>,
        messages: Vec<ChatMessage>,
        options: GenerationOptions,
    ) -> Result<GenerationTicket, ServiceError> {
        let session_id = session_id.into();
        if session_id.is_empty()
            || session_id.len() > 128
            || !session_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:".contains(&byte))
        {
            return Err(ServiceError::InvalidRequest(
                "session ID must use 1..=128 ASCII letters, numbers, '-', '_', '.', or ':'"
                    .to_owned(),
            ));
        }
        self.generate_with_session(model, Some(session_id), messages, options)
    }

    fn generate_with_session(
        &self,
        model: &str,
        session_id: Option<String>,
        messages: Vec<ChatMessage>,
        options: GenerationOptions,
    ) -> Result<GenerationTicket, ServiceError> {
        options
            .validate()
            .map_err(|error| ServiceError::InvalidRequest(error.to_string()))?;
        if messages.is_empty() {
            return Err(ServiceError::InvalidRequest(
                "messages must contain at least one chat turn".to_owned(),
            ));
        }
        if messages.len() > 1_024 {
            return Err(ServiceError::InvalidRequest(
                "messages must contain at most 1024 chat turns".to_owned(),
            ));
        }
        let content_bytes = messages
            .iter()
            .try_fold(0_usize, |total, message| {
                total.checked_add(message.content.len())
            })
            .ok_or_else(|| {
                ServiceError::InvalidRequest("message content size overflowed".to_owned())
            })?;
        if content_bytes > 1024 * 1024 {
            return Err(ServiceError::InvalidRequest(
                "message content must be at most 1 MiB".to_owned(),
            ));
        }
        let catalog = ModelCatalog::discover(&self.inner.paths, self.inner.ollama_root.clone())?;
        let record = catalog.resolve_for_run(model)?;
        if !record.compatibility.is_metadata_compatible() {
            return Err(ModelError::NotRunnable {
                name: record.name,
                status: record.compatibility.to_string(),
            }
            .into());
        }
        let architecture = record
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".to_owned());
        if !matches!(architecture.as_str(), "gemma4" | "glm_moe_dsa") {
            return Err(ServiceError::UnsupportedArchitecture {
                name: record.name,
                architecture,
            });
        }
        let path = record
            .path
            .clone()
            .ok_or_else(|| ServiceError::MissingModelPath {
                name: record.name.clone(),
            })?;

        let mut workers = lock_unpoisoned(&self.inner.workers);
        if !workers.contains_key(&path) {
            if workers.len() >= self.inner.limits.maximum_loaded_models {
                return Err(ServiceError::ModelLimit {
                    limit: self.inner.limits.maximum_loaded_models,
                });
            }
            let worker = spawn_model_worker(
                record.name.clone(),
                path.clone(),
                architecture,
                self.inner.backend,
                self.inner.limits,
            )?;
            workers.insert(path.clone(), worker);
        }
        let worker = workers.get(&path).expect("worker inserted above");
        let cancellation = CancellationFlag::default();
        let (events, receiver) = async_mpsc::channel(self.inner.limits.token_buffer);
        let job = GenerationJob {
            session_id,
            messages,
            options,
            cancellation: cancellation.clone(),
            events,
        };
        match worker.sender.try_send(job) {
            Ok(()) => Ok(GenerationTicket {
                events: receiver,
                cancellation,
            }),
            Err(mpsc::TrySendError::Full(_)) => Err(ServiceError::QueueFull {
                model: worker.name.clone(),
            }),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(ServiceError::WorkerStopped {
                model: worker.name.clone(),
            }),
        }
    }
}

fn spawn_model_worker(
    name: String,
    path: PathBuf,
    architecture: String,
    backend: BackendSelection,
    limits: RuntimeLimits,
) -> Result<ModelWorker, ServiceError> {
    let (sender, receiver) = mpsc::sync_channel::<GenerationJob>(limits.request_queue);
    let status = Arc::new(Mutex::new(WorkerStatus::Loading));
    let worker_status = Arc::clone(&status);
    let worker_name = name.clone();
    let load_architecture = architecture.clone();
    let backend_label = backend.label();
    let thread = thread::Builder::new()
        .name("diskmule-model".to_owned())
        .spawn(move || {
            let engine = GenerationEngine::open(
                worker_name,
                path,
                &load_architecture,
                limits.context,
                backend,
            );
            match engine {
                Ok(engine) => {
                    *lock_unpoisoned(&worker_status) = WorkerStatus::Ready;
                    run_model_worker(
                        &engine,
                        receiver,
                        &worker_status,
                        limits.maximum_sessions_per_model,
                    );
                }
                Err(error) => {
                    *lock_unpoisoned(&worker_status) = WorkerStatus::Failed;
                    let message = error.to_string();
                    for job in receiver {
                        let _ = job.events.blocking_send(GenerationEvent::Error {
                            message: message.clone(),
                            cancelled: false,
                        });
                    }
                }
            }
            *lock_unpoisoned(&worker_status) = WorkerStatus::Stopped;
        })
        .map_err(|source| ServiceError::SpawnWorker {
            model: name.clone(),
            source,
        })?;
    Ok(ModelWorker {
        name,
        architecture,
        backend: backend_label,
        sender,
        status,
        thread: Some(thread),
    })
}

fn run_model_worker(
    engine: &GenerationEngine,
    receiver: mpsc::Receiver<GenerationJob>,
    status: &Mutex<WorkerStatus>,
    maximum_sessions: usize,
) {
    let mut sessions = HashMap::<String, ResidentSession>::new();
    let mut session_clock = 0_u64;
    for job in receiver {
        *lock_unpoisoned(status) = WorkerStatus::Busy;
        if job.cancellation.is_cancelled() {
            let _ = job.events.blocking_send(GenerationEvent::Error {
                message: RuntimeError::Cancelled.to_string(),
                cancelled: true,
            });
            *lock_unpoisoned(status) = WorkerStatus::Ready;
            continue;
        }
        let events = job.events.clone();
        let cancellation = job.cancellation.clone();
        let mut on_token = |id, text: &str| {
            if events
                .blocking_send(GenerationEvent::Token {
                    id,
                    text: text.to_owned(),
                })
                .is_err()
            {
                cancellation.cancel();
            }
        };
        let result = if let Some(session_id) = &job.session_id {
            session_clock = session_clock.wrapping_add(1);
            if !sessions.contains_key(session_id) {
                if sessions.len() >= maximum_sessions {
                    let oldest = sessions
                        .iter()
                        .min_by(|left, right| {
                            left.1
                                .last_used
                                .cmp(&right.1.last_used)
                                .then_with(|| left.0.cmp(right.0))
                        })
                        .map(|(id, _)| id.clone())
                        .expect("positive session limit with a full map has an oldest entry");
                    sessions.remove(&oldest);
                }
                engine.new_session().map(|state| {
                    sessions.insert(
                        session_id.clone(),
                        ResidentSession {
                            state,
                            last_used: session_clock,
                        },
                    );
                })
            } else {
                Ok(())
            }
            .and_then(|()| {
                let session = sessions
                    .get_mut(session_id)
                    .expect("session exists after successful creation");
                session.last_used = session_clock;
                engine.generate_chat_in_session(
                    &mut session.state,
                    &job.messages,
                    job.options,
                    &job.cancellation,
                    &mut on_token,
                )
            })
        } else {
            engine.generate_chat(&job.messages, job.options, &job.cancellation, &mut on_token)
        };
        match result {
            Ok(result) => {
                let _ = job
                    .events
                    .blocking_send(GenerationEvent::Complete(Box::new(result)));
            }
            Err(error) => {
                let cancelled = error.is_cancelled();
                let _ = job.events.blocking_send(GenerationEvent::Error {
                    message: error.to_string(),
                    cancelled,
                });
            }
        }
        *lock_unpoisoned(status) = WorkerStatus::Ready;
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn runtime_usize_environment(
    name: &'static str,
    default: usize,
    minimum: usize,
    maximum: usize,
) -> Result<usize, RuntimeError> {
    let Some(raw) = env::var_os(name) else {
        return Ok(default);
    };
    raw.to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            RuntimeError::InvalidConfiguration(format!(
                "{name} must be an integer in {minimum}..={maximum}"
            ))
        })
}

fn runtime_f32_environment(name: &'static str, default: f32) -> Result<f32, RuntimeError> {
    let Some(raw) = env::var_os(name) else {
        return Ok(default);
    };
    raw.to_str()
        .and_then(|value| value.parse::<f32>().ok())
        .ok_or_else(|| {
            RuntimeError::InvalidConfiguration(format!("{name} must be a floating-point number"))
        })
}

fn runtime_u64_environment(name: &'static str, default: u64) -> Result<u64, RuntimeError> {
    let Some(raw) = env::var_os(name) else {
        return Ok(default);
    };
    raw.to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| {
            RuntimeError::InvalidConfiguration(format!("{name} must be an unsigned integer"))
        })
}

struct Sampler {
    config: SamplingConfig,
    random: SplitMix64,
}

impl Sampler {
    fn new(config: SamplingConfig) -> Result<Self, RuntimeError> {
        Ok(Self {
            config: config.validate()?,
            random: SplitMix64(config.seed),
        })
    }

    fn select(&mut self, logits: &[f32]) -> Result<u32, RuntimeError> {
        if logits.is_empty() {
            return Err(RuntimeError::InvalidConfiguration(
                "cannot sample an empty logit vector".to_owned(),
            ));
        }
        let mut candidates = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(RuntimeError::InvalidConfiguration(
                "all token logits are non-finite".to_owned(),
            ));
        }
        candidates.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        if self.config.temperature == 0.0 || self.config.top_k == 1 {
            return u32::try_from(candidates[0].0).map_err(|_| {
                RuntimeError::InvalidConfiguration("vocabulary exceeds u32 token IDs".to_owned())
            });
        }
        if self.config.top_k > 0 {
            candidates.truncate(self.config.top_k.min(candidates.len()));
        }
        let maximum = candidates[0].1;
        let mut weighted = candidates
            .into_iter()
            .map(|(token, logit)| {
                let weight = f64::from((logit - maximum) / self.config.temperature).exp();
                (token, weight)
            })
            .collect::<Vec<_>>();
        let total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(RuntimeError::InvalidConfiguration(
                "sampling weights are not finite and positive".to_owned(),
            ));
        }
        let cutoff = total * f64::from(self.config.top_p);
        let mut cumulative = 0.0;
        let mut retained = weighted.len();
        for (index, (_, weight)) in weighted.iter().enumerate() {
            cumulative += *weight;
            if cumulative >= cutoff {
                retained = index + 1;
                break;
            }
        }
        weighted.truncate(retained.max(1));
        let retained_total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
        let needle = self.random.unit_f64() * retained_total;
        let mut cumulative = 0.0;
        for (token, weight) in &weighted {
            cumulative += *weight;
            if needle < cumulative {
                return u32::try_from(*token).map_err(|_| {
                    RuntimeError::InvalidConfiguration(
                        "vocabulary exceeds u32 token IDs".to_owned(),
                    )
                });
            }
        }
        u32::try_from(weighted.last().expect("retained sample is non-empty").0).map_err(|_| {
            RuntimeError::InvalidConfiguration("vocabulary exceeds u32 token IDs".to_owned())
        })
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(target_os = "macos")]
fn expert_residency_from_environment() -> Result<ExpertResidency, RuntimeError> {
    let Some(raw) = env::var_os("DISKMULE_EXPERT_SLOTS") else {
        return Ok(ExpertResidency::Mapped);
    };
    let slots_per_layer = raw
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=128).contains(value))
        .ok_or_else(|| {
            RuntimeError::InvalidConfiguration(
                "DISKMULE_EXPERT_SLOTS must be an integer in 1..=128".to_owned(),
            )
        })?;
    Ok(ExpertResidency::Streamed { slots_per_layer })
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;

    use tempfile::TempDir;

    use crate::{
        gemma4::{ChatMessage, ChatRole, cpu::CancellationFlag},
        glm52::cpu::write_test_model,
    };

    use super::{
        BackendSelection, GenerationEngine, GenerationOptions, GenerationTicket, RuntimeLimits,
        Sampler, SamplingConfig,
    };

    #[test]
    fn greedy_sampling_is_stable_and_uses_lowest_id_for_ties() {
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        assert_eq!(sampler.select(&[-2.0, 4.0, 4.0, 1.0]).unwrap(), 1);
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_bounded_by_top_k() {
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 2,
            top_p: 1.0,
            seed: 42,
        };
        let mut left = Sampler::new(config).unwrap();
        let mut right = Sampler::new(config).unwrap();
        let logits = [3.0, 2.0, 1.0, 0.0];
        let left_sequence = (0..32)
            .map(|_| left.select(&logits).unwrap())
            .collect::<Vec<_>>();
        let right_sequence = (0..32)
            .map(|_| right.select(&logits).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(left_sequence, right_sequence);
        assert!(left_sequence.iter().all(|token| *token <= 1));
    }

    #[test]
    fn sampling_bounds_are_rejected() {
        assert!(
            SamplingConfig {
                temperature: f32::NAN,
                ..SamplingConfig::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            GenerationOptions {
                maximum_new_tokens: 0,
                ..GenerationOptions::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn runtime_limits_reject_unbounded_or_empty_resources() {
        assert!(
            RuntimeLimits {
                request_queue: 0,
                ..RuntimeLimits::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            RuntimeLimits {
                maximum_loaded_models: 17,
                ..RuntimeLimits::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn dropping_a_generation_ticket_cancels_its_work() {
        let cancellation = CancellationFlag::default();
        let observed = cancellation.clone();
        let (_sender, events) = mpsc::channel(1);
        drop(GenerationTicket {
            events,
            cancellation,
        });
        assert!(observed.is_cancelled());
    }

    #[test]
    fn glm_uses_shared_sampling_and_persistent_prompt_sessions() {
        let temp = TempDir::new().unwrap();
        write_test_model(temp.path());
        let engine = GenerationEngine::open(
            "glm-tiny",
            temp.path(),
            "glm_moe_dsa",
            32,
            BackendSelection::Cpu,
        )
        .unwrap();
        assert_eq!(engine.backend(), BackendSelection::Cpu);

        let cancellation = CancellationFlag::default();
        let direct = engine
            .generate_tokens(
                &[1],
                GenerationOptions {
                    maximum_new_tokens: 1,
                    ..GenerationOptions::default()
                },
                &cancellation,
                |_, _| {},
            )
            .unwrap();
        assert_eq!(direct.token_ids, [5]);
        assert!(direct.profile.mapped_bytes_touched > 0);
        assert!(direct.profile.resident_kv_bytes > 0);
        assert!(direct.profile.embedding_time > std::time::Duration::ZERO);
        assert!(direct.profile.attention_time > std::time::Duration::ZERO);
        assert!(direct.profile.feed_forward_time > std::time::Duration::ZERO);
        assert!(direct.profile.output_time > std::time::Duration::ZERO);
        assert!(direct.profile.expert_cache_misses > 0);
        assert!(direct.profile.expert_reads > 0);

        let messages = [ChatMessage::new(ChatRole::User, "!")];
        let options = GenerationOptions {
            maximum_new_tokens: 1,
            ..GenerationOptions::default()
        };
        let mut session = engine.new_session().unwrap();
        let first = engine
            .generate_chat_in_session(&mut session, &messages, options, &cancellation, |_, _| {})
            .unwrap();
        let second = engine
            .generate_chat_in_session(&mut session, &messages, options, &cancellation, |_, _| {})
            .unwrap();
        assert_eq!(first.token_ids, second.token_ids);
        assert_eq!(first.profile.reused_prompt_tokens, 0);
        assert!(second.profile.reused_prompt_tokens > 0);
    }
}
