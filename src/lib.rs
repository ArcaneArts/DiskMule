pub mod chat;
pub mod cli;
pub mod config;
pub mod cpu;
pub mod error;
#[cfg(target_os = "macos")]
pub mod expert;
pub mod gemma4;
pub mod gguf;
pub mod glm52;
pub mod metal;
pub mod model;
pub mod runtime;
pub mod safetensors;
pub mod server;

use std::io::{self, BufRead, Write};

use cli::{Action, Cli};
use config::Paths;
use error::{AppError, Result};
use model::{ModelCatalog, format_size};
use runtime::{BackendSelection, RuntimeLimits, RuntimeService};

/// Runs one parsed DiskMule command.
///
/// Keeping dispatch in the library lets command behavior be tested without
/// spawning a subprocess.
pub async fn execute(cli: &Cli, paths: &Paths, output: &mut impl Write) -> Result<()> {
    execute_with_io(cli, paths, &mut io::stdin().lock(), output).await
}

/// Runs one parsed command with injectable terminal I/O.
pub async fn execute_with_io(
    cli: &Cli,
    paths: &Paths,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    match cli.action()? {
        Action::Serve => {
            let runtime = RuntimeService::new(
                paths.clone(),
                config::ollama_models_dir()?,
                BackendSelection::from_environment()?,
                RuntimeLimits::from_environment()?,
            )?;
            server::serve(
                server::bind_from_environment()?,
                runtime,
                server::shutdown_signal(),
            )
            .await?;
        }
        Action::Run { model } => {
            let ollama_root = config::ollama_models_dir()?;
            let catalog = ModelCatalog::discover(paths, ollama_root.clone())?;
            let record = catalog.resolve_for_run(model)?;
            writeln!(output, "Model: {}", record.name)?;
            writeln!(output, "Source: {}", record.source)?;
            if let Some(path) = &record.path {
                writeln!(output, "Path: {}", path.display())?;
            }
            writeln!(
                output,
                "Architecture: {}",
                record.architecture.as_deref().unwrap_or("-")
            )?;
            writeln!(
                output,
                "Quantization: {}",
                record.quantization.as_deref().unwrap_or("-")
            )?;
            writeln!(
                output,
                "Size: {}",
                record.size.map(format_size).as_deref().unwrap_or("-")
            )?;
            if let Some(summary) = &record.gguf {
                writeln!(output, "GGUF version: {}", summary.version)?;
                writeln!(output, "Tensors: {}", summary.tensor_count)?;
                writeln!(output, "Alignment: {} bytes", summary.alignment)?;
                writeln!(output, "Tensor data offset: {}", summary.data_offset)?;
            }
            if let Some(summary) = &record.safetensors {
                writeln!(output, "SafeTensors shards: {}", summary.shard_count)?;
                writeln!(output, "Tensors: {}", summary.tensor_count)?;
            }
            writeln!(output, "Status: {}", record.compatibility)?;
            if !record.compatibility.is_metadata_compatible() {
                return Err(model::ModelError::NotRunnable {
                    name: record.name,
                    status: record.compatibility.to_string(),
                }
                .into());
            }
            if record.path.is_none() {
                return Err(AppError::InvalidConfiguration(format!(
                    "model {:?} has no local tensor path",
                    record.name
                )));
            }
            let runtime = RuntimeService::new(
                paths.clone(),
                ollama_root,
                BackendSelection::from_environment()?,
                RuntimeLimits::from_environment()?,
            )?;
            chat::run_model(model, runtime, input, output).await?;
        }
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
        }
    };
    Ok(())
}

pub fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,diskmule=info"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .compact()
        .try_init();
}
