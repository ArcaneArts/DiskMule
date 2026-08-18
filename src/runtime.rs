//! Shared model loading, prompt rendering, sampling, and token generation.

use std::{
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc as async_mpsc;

use crate::{
    config::Paths,
    gemma4::{
        ChatMessage, Gemma4Tokenizer,
        cpu::{CancellationFlag, Gemma4CpuModel, GenerationResult, RuntimeError},
        render_chat,
    },
    model::{ModelCatalog, ModelError},
};

#[cfg(target_os = "macos")]
use crate::gemma4::metal::{ExpertResidency, Gemma4MetalModel};

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
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            context: DEFAULT_CONTEXT,
            maximum_loaded_models: 1,
            request_queue: 8,
            token_buffer: 32,
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
    Gemma4Cpu(Gemma4CpuModel),
    #[cfg(target_os = "macos")]
    Gemma4Metal(Gemma4MetalModel),
}

impl GenerationEngine {
    pub fn open_gemma4(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        context: usize,
        backend: BackendSelection,
    ) -> Result<Self, RuntimeError> {
        let path = path.as_ref().to_path_buf();
        let model = match backend {
            BackendSelection::Cpu => {
                ArchitectureModel::Gemma4Cpu(Gemma4CpuModel::open(&path, context)?)
            }
            #[cfg(target_os = "macos")]
            BackendSelection::Metal { residency } => ArchitectureModel::Gemma4Metal(
                Gemma4MetalModel::open_with_expert_residency(&path, context, residency)?,
            ),
        };
        Ok(Self {
            name: name.into(),
            path,
            backend,
            model,
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

    pub fn tokenizer(&self) -> &Gemma4Tokenizer {
        match &self.model {
            ArchitectureModel::Gemma4Cpu(model) => &model.tokenizer,
            #[cfg(target_os = "macos")]
            ArchitectureModel::Gemma4Metal(model) => &model.tokenizer,
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
        let rendered = render_chat(messages)?;
        let tokens = self.tokenizer().encode(&rendered, false);
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
}

struct GenerationJob {
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
        if architecture != "gemma4" {
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
    thread::Builder::new()
        .name("diskmule-model".to_owned())
        .spawn(move || {
            let engine = GenerationEngine::open_gemma4(worker_name, path, limits.context, backend);
            match engine {
                Ok(engine) => {
                    *lock_unpoisoned(&worker_status) = WorkerStatus::Ready;
                    run_model_worker(&engine, receiver, &worker_status);
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
        backend: backend.label(),
        sender,
        status,
    })
}

fn run_model_worker(
    engine: &GenerationEngine,
    receiver: mpsc::Receiver<GenerationJob>,
    status: &Mutex<WorkerStatus>,
) {
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
        let result =
            engine.generate_chat(&job.messages, job.options, &job.cancellation, |id, text| {
                if events
                    .blocking_send(GenerationEvent::Token {
                        id,
                        text: text.to_owned(),
                    })
                    .is_err()
                {
                    cancellation.cancel();
                }
            });
        match result {
            Ok(result) => {
                let _ = job
                    .events
                    .blocking_send(GenerationEvent::Complete(Box::new(result)));
            }
            Err(error) => {
                let cancelled = matches!(error, RuntimeError::Cancelled);
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

    use crate::gemma4::cpu::CancellationFlag;

    use super::{GenerationOptions, GenerationTicket, RuntimeLimits, Sampler, SamplingConfig};

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
}
