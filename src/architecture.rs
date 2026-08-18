//! Shared model-family contract, built-in registry, and capability reporting.

use std::{any::Any, path::Path};

use serde::Serialize;

use crate::{
    gemma4::{cpu::Gemma4CpuModel, render_chat as render_gemma4_chat},
    generation::{CancellationFlag, GenerationResult, RuntimeError},
    glm52::{cpu::Glm52CpuModel, tokenizer::render_chat as render_glm52_chat},
    qwen35::{cpu::Qwen35CpuModel, tokenizer::render_chat as render_qwen35_chat},
    runtime::BackendSelection,
};

#[cfg(target_os = "macos")]
use crate::gemma4::metal::Gemma4MetalModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContainerFormat {
    Gguf,
    Safetensors,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetalSupport {
    FullGraph,
    MatrixOffload,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResidencySupport {
    Mapped,
    RoutedExpertCache,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct ArchitectureCapabilities {
    pub architecture: &'static str,
    pub family: &'static str,
    pub containers: &'static [ContainerFormat],
    pub text_chat: bool,
    pub multimodal: bool,
    pub persistent_sessions: bool,
    pub cpu_reference: bool,
    pub metal: MetalSupport,
    pub residency: ResidencySupport,
}

const GGUF: &[ContainerFormat] = &[ContainerFormat::Gguf];
const SAFETENSORS: &[ContainerFormat] = &[ContainerFormat::Safetensors];

pub const GEMMA4_CAPABILITIES: ArchitectureCapabilities = ArchitectureCapabilities {
    architecture: "gemma4",
    family: "Gemma 4 text",
    containers: GGUF,
    text_chat: true,
    multimodal: false,
    persistent_sessions: true,
    cpu_reference: true,
    metal: MetalSupport::FullGraph,
    residency: ResidencySupport::RoutedExpertCache,
};

pub const GLM52_CAPABILITIES: ArchitectureCapabilities = ArchitectureCapabilities {
    architecture: "glm_moe_dsa",
    family: "GLM-5.2 text",
    containers: SAFETENSORS,
    text_chat: true,
    multimodal: false,
    persistent_sessions: true,
    cpu_reference: true,
    metal: MetalSupport::MatrixOffload,
    residency: ResidencySupport::RoutedExpertCache,
};

pub const QWEN35_CAPABILITIES: ArchitectureCapabilities = ArchitectureCapabilities {
    architecture: "qwen35",
    family: "Qwen 3.5 text",
    containers: GGUF,
    text_chat: true,
    multimodal: false,
    persistent_sessions: true,
    cpu_reference: true,
    metal: MetalSupport::MatrixOffload,
    residency: ResidencySupport::Mapped,
};

pub const BUILTIN_ARCHITECTURES: &[ArchitectureCapabilities] =
    &[GEMMA4_CAPABILITIES, GLM52_CAPABILITIES, QWEN35_CAPABILITIES];

pub fn capabilities(architecture: &str) -> Option<&'static ArchitectureCapabilities> {
    BUILTIN_ARCHITECTURES
        .iter()
        .find(|capabilities| capabilities.architecture == architecture)
}

pub trait ArchitectureSession {
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any> ArchitectureSession for T {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Object-safe boundary between model-family code and the shared scheduler.
pub trait LoadedArchitecture {
    fn capabilities(&self) -> &'static ArchitectureCapabilities;
    fn backend_label(&self) -> String;
    fn tokenize_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>, RuntimeError>;
    fn new_session(&self) -> Result<Box<dyn ArchitectureSession>, RuntimeError>;
    fn generate_session(
        &self,
        session: &mut dyn ArchitectureSession,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        select: &mut dyn FnMut(&[f32]) -> Result<u32, RuntimeError>,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GenerationResult, RuntimeError>;
}

pub fn load(
    path: &Path,
    architecture: &str,
    context: usize,
    backend: BackendSelection,
) -> Result<Box<dyn LoadedArchitecture>, RuntimeError> {
    let model: Box<dyn LoadedArchitecture> = match architecture {
        "gemma4" => match backend {
            BackendSelection::Cpu => Box::new(Gemma4CpuModel::open(path, context)?),
            #[cfg(target_os = "macos")]
            BackendSelection::Metal { residency } => Box::new(
                Gemma4MetalModel::open_with_expert_residency(path, context, residency)?,
            ),
        },
        "glm_moe_dsa" => match backend {
            BackendSelection::Cpu => {
                Box::new(Glm52CpuModel::from_directory_with_context(path, context)?)
            }
            #[cfg(target_os = "macos")]
            BackendSelection::Metal { .. } => Box::new(
                Glm52CpuModel::from_directory_with_context_and_metal(path, context)?,
            ),
        },
        "qwen35" => match backend {
            BackendSelection::Cpu => Box::new(Qwen35CpuModel::open(path, context)?),
            #[cfg(target_os = "macos")]
            BackendSelection::Metal { .. } => Box::new(Qwen35CpuModel::open_metal(path, context)?),
        },
        other => {
            return Err(RuntimeError::InvalidConfiguration(format!(
                "unsupported architecture {other:?}"
            )));
        }
    };
    tracing::info!(
        model = %path.display(),
        architecture = model.capabilities().architecture,
        backend = %model.backend_label(),
        "loaded model architecture"
    );
    Ok(model)
}

impl LoadedArchitecture for Gemma4CpuModel {
    fn capabilities(&self) -> &'static ArchitectureCapabilities {
        &GEMMA4_CAPABILITIES
    }

    fn backend_label(&self) -> String {
        "CPU".to_owned()
    }

    fn tokenize_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>, RuntimeError> {
        Ok(self.tokenizer.encode(&render_gemma4_chat(messages)?, false))
    }

    fn new_session(&self) -> Result<Box<dyn ArchitectureSession>, RuntimeError> {
        Ok(Box::new(self.new_session()))
    }

    fn generate_session(
        &self,
        session: &mut dyn ArchitectureSession,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        select: &mut dyn FnMut(&[f32]) -> Result<u32, RuntimeError>,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GenerationResult, RuntimeError> {
        self.generate_session_with_selector(
            session_mut(session, self.capabilities().architecture)?,
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            select,
            on_token,
        )
    }
}

#[cfg(target_os = "macos")]
impl LoadedArchitecture for Gemma4MetalModel {
    fn capabilities(&self) -> &'static ArchitectureCapabilities {
        &GEMMA4_CAPABILITIES
    }

    fn backend_label(&self) -> String {
        format!("Metal ({})", self.device_name())
    }

    fn tokenize_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>, RuntimeError> {
        Ok(self.tokenizer.encode(&render_gemma4_chat(messages)?, false))
    }

    fn new_session(&self) -> Result<Box<dyn ArchitectureSession>, RuntimeError> {
        Ok(Box::new(self.new_session()?))
    }

    fn generate_session(
        &self,
        session: &mut dyn ArchitectureSession,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        select: &mut dyn FnMut(&[f32]) -> Result<u32, RuntimeError>,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GenerationResult, RuntimeError> {
        self.generate_session_with_selector(
            session_mut(session, self.capabilities().architecture)?,
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            select,
            on_token,
        )
    }
}

impl LoadedArchitecture for Glm52CpuModel {
    fn capabilities(&self) -> &'static ArchitectureCapabilities {
        &GLM52_CAPABILITIES
    }

    fn backend_label(&self) -> String {
        #[cfg(target_os = "macos")]
        if let Some(device) = self.metal_device_name() {
            return format!("Metal ({device})");
        }
        "CPU".to_owned()
    }

    fn tokenize_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>, RuntimeError> {
        Ok(self.tokenizer.encode(&render_glm52_chat(messages, false))?)
    }

    fn new_session(&self) -> Result<Box<dyn ArchitectureSession>, RuntimeError> {
        Ok(Box::new(self.new_session()))
    }

    fn generate_session(
        &self,
        session: &mut dyn ArchitectureSession,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        select: &mut dyn FnMut(&[f32]) -> Result<u32, RuntimeError>,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GenerationResult, RuntimeError> {
        self.generate_session_with_selector(
            session_mut(session, self.capabilities().architecture)?,
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            select,
            on_token,
        )
    }
}

impl LoadedArchitecture for Qwen35CpuModel {
    fn capabilities(&self) -> &'static ArchitectureCapabilities {
        &QWEN35_CAPABILITIES
    }

    fn backend_label(&self) -> String {
        self.backend_label()
    }

    fn tokenize_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>, RuntimeError> {
        Ok(self
            .tokenizer
            .encode(&render_qwen35_chat(messages, false)?)?)
    }

    fn new_session(&self) -> Result<Box<dyn ArchitectureSession>, RuntimeError> {
        Ok(Box::new(self.new_session()))
    }

    fn generate_session(
        &self,
        session: &mut dyn ArchitectureSession,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        select: &mut dyn FnMut(&[f32]) -> Result<u32, RuntimeError>,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GenerationResult, RuntimeError> {
        self.generate_session_with_selector(
            session_mut(session, self.capabilities().architecture)?,
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            select,
            on_token,
        )
    }
}

fn session_mut<'a, T: Any>(
    session: &'a mut dyn ArchitectureSession,
    architecture: &str,
) -> Result<&'a mut T, RuntimeError> {
    session.as_any_mut().downcast_mut().ok_or_else(|| {
        RuntimeError::InvalidConfiguration(format!(
            "session architecture does not match loaded {architecture:?} model"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BUILTIN_ARCHITECTURES, ContainerFormat, MetalSupport, ResidencySupport, capabilities,
    };

    #[test]
    fn registry_reports_each_runtime_architecture_once() {
        assert_eq!(BUILTIN_ARCHITECTURES.len(), 3);
        for (index, architecture) in BUILTIN_ARCHITECTURES.iter().enumerate() {
            assert_eq!(capabilities(architecture.architecture), Some(architecture));
            assert!(architecture.text_chat);
            assert!(architecture.persistent_sessions);
            assert!(architecture.cpu_reference);
            assert!(!architecture.multimodal);
            assert!(
                !BUILTIN_ARCHITECTURES[..index]
                    .iter()
                    .any(|other| other.architecture == architecture.architecture)
            );
        }
        assert_eq!(capabilities("unknown"), None);
        assert_eq!(
            capabilities("gemma4").unwrap().containers,
            [ContainerFormat::Gguf]
        );
        assert_eq!(
            capabilities("gemma4").unwrap().metal,
            MetalSupport::FullGraph
        );
        assert_eq!(
            capabilities("glm_moe_dsa").unwrap().residency,
            ResidencySupport::RoutedExpertCache
        );
    }
}
