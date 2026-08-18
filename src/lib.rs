pub mod cli;
pub mod config;
pub mod error;
pub mod gguf;
pub mod model;

use std::io::Write;

use cli::{Action, Cli};
use config::Paths;
use error::{AppError, Result};
use model::{ModelCatalog, format_size};

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
            let catalog = ModelCatalog::discover(paths, config::ollama_models_dir()?)?;
            writeln!(
                output,
                "NAME\tARCHITECTURE\tQUANTIZATION\tSIZE\tSOURCE\tSTATUS"
            )?;
            for record in catalog.records() {
                writeln!(
                    output,
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    record.name,
                    record.architecture.as_deref().unwrap_or("-"),
                    record.quantization.as_deref().unwrap_or("-"),
                    record.size.map(format_size).as_deref().unwrap_or("-"),
                    record.source,
                    record.compatibility
                )?;
            }
            Ok(())
        }
        Action::Remove { model } => {
            let mut catalog = ModelCatalog::discover(paths, config::ollama_models_dir()?)?;
            let removed = catalog.remove(model, &Default::default())?;
            writeln!(
                output,
                "Removed {} ({}).",
                removed.name,
                removed.path.display()
            )?;
            Ok(())
        }
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
