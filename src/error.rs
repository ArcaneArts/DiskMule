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
        }
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
