use std::{io, process::ExitCode};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    Usage(String),

    #[error("{0}")]
    InvalidConfiguration(String),

    #[error("could not determine the DiskMule data directory; set DISKMULE_HOME")]
    HomeDirectoryUnavailable,

    #[error("{0}")]
    NotImplemented(&'static str),

    #[error("{operation} is not implemented yet for {context}")]
    NotImplementedWithContext {
        operation: &'static str,
        context: String,
    },

    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    Server(#[from] crate::server::ServerError),

    #[error(transparent)]
    Gemma4(#[from] crate::gemma4::cpu::RuntimeError),

    #[error(transparent)]
    Runtime(#[from] crate::runtime::ServiceError),

    #[error("generation failed: {0}")]
    GenerationFailed(String),
}

impl AppError {
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Usage(_) | Self::InvalidConfiguration(_) | Self::HomeDirectoryUnavailable => {
                ExitCode::from(2)
            }
            Self::NotImplemented(_) | Self::NotImplementedWithContext { .. } | Self::Io(_) => {
                ExitCode::FAILURE
            }
            Self::Gguf(_) => ExitCode::FAILURE,
            Self::Model(_) => ExitCode::FAILURE,
            Self::Server(_) => ExitCode::FAILURE,
            Self::Gemma4(_) => ExitCode::FAILURE,
            Self::Runtime(_) => ExitCode::FAILURE,
            Self::GenerationFailed(_) => ExitCode::FAILURE,
        }
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
