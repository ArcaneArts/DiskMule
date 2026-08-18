use std::{io, process::ExitCode};

use clap::Parser;
use diskmule::{cli::Cli, config::Paths};

#[tokio::main]
async fn main() -> ExitCode {
    diskmule::init_logging();

    let cli = Cli::parse();
    let paths = match Paths::discover() {
        Ok(paths) => paths,
        Err(error) => {
            eprintln!("error: {error}");
            return error.exit_code();
        }
    };

    match diskmule::execute(&cli, &paths, &mut io::stdout().lock()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            error.exit_code()
        }
    }
}
