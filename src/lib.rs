pub mod cli;
pub mod config;
pub mod error;
pub mod gguf;

use std::io::Write;

use cli::{Action, Cli};
use config::Paths;
use error::{AppError, Result};

/// Runs one parsed DiskMule command.
///
/// The concrete model and server operations are filled in by the next vertical
/// slices. Keeping dispatch in the library lets command behavior be tested
/// without spawning a subprocess.
pub fn execute(cli: &Cli, paths: &Paths, output: &mut impl Write) -> Result<()> {
    match cli.action()? {
        Action::Serve => Err(AppError::NotImplemented(
            "the local server is not implemented yet",
        )),
        Action::Run { model } => Err(AppError::NotImplementedWithContext {
            operation: "model inference",
            context: model.to_owned(),
        }),
        Action::List => {
            writeln!(
                output,
                "No models listed yet (managed root: {}).",
                paths.models.display()
            )?;
            Ok(())
        }
        Action::Remove { model } => Err(AppError::NotImplementedWithContext {
            operation: "model removal",
            context: model.to_owned(),
        }),
    }
}

pub fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .compact()
        .try_init();
}
