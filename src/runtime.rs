//! Shared model loading, prompt rendering, sampling, and token generation.

use std::{
    env,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::gemma4::{
    ChatMessage, Gemma4Tokenizer,
    cpu::{CancellationFlag, Gemma4CpuModel, GenerationResult, RuntimeError},
    render_chat,
};

#[cfg(target_os = "macos")]
use crate::gemma4::metal::{ExpertResidency, Gemma4MetalModel};

pub const DEFAULT_CONTEXT: usize = 4_096;
pub const DEFAULT_MAXIMUM_NEW_TOKENS: usize = 64;

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
    use super::{GenerationOptions, Sampler, SamplingConfig};

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
}
