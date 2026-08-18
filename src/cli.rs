use clap::{Parser, Subcommand};

use crate::error::{AppError, Result};

#[derive(Debug, Parser)]
#[command(
    name = "diskmule",
    version,
    about = "Run and serve local models with explicit storage control",
    arg_required_else_help = true,
    disable_help_subcommand = true
)]
pub struct Cli {
    /// Start the local HTTP inference service.
    #[arg(long)]
    pub serve: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Inspect a model and start an interactive chat.
    Run {
        /// DiskMule model, local path, or discovered Ollama model name.
        model: String,
    },
    /// List managed and compatible externally discovered models.
    #[command(visible_alias = "list")]
    Ls,
    /// Remove a model owned by DiskMule.
    #[command(visible_alias = "remove")]
    Rm {
        /// Name of a DiskMule-managed model.
        model: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action<'a> {
    Serve,
    Run { model: &'a str },
    List,
    Remove { model: &'a str },
}

impl Cli {
    pub fn action(&self) -> Result<Action<'_>> {
        match (self.serve, &self.command) {
            (true, None) => Ok(Action::Serve),
            (false, Some(Command::Run { model })) => Ok(Action::Run { model }),
            (false, Some(Command::Ls)) => Ok(Action::List),
            (false, Some(Command::Rm { model })) => Ok(Action::Remove { model }),
            (true, Some(_)) => Err(AppError::Usage(
                "--serve cannot be combined with a subcommand".to_owned(),
            )),
            (false, None) => Err(AppError::Usage("choose --serve, run, ls, or rm".to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser, error::ErrorKind};

    use super::{Action, Cli};

    #[test]
    fn clap_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_each_public_command() {
        let serve = Cli::try_parse_from(["diskmule", "--serve"]).unwrap();
        assert_eq!(serve.action().unwrap(), Action::Serve);

        let run = Cli::try_parse_from(["diskmule", "run", "gemma4:26b"]).unwrap();
        assert_eq!(
            run.action().unwrap(),
            Action::Run {
                model: "gemma4:26b"
            }
        );

        let list = Cli::try_parse_from(["diskmule", "ls"]).unwrap();
        assert_eq!(list.action().unwrap(), Action::List);

        let remove = Cli::try_parse_from(["diskmule", "rm", "gemma4:26b"]).unwrap();
        assert_eq!(
            remove.action().unwrap(),
            Action::Remove {
                model: "gemma4:26b"
            }
        );
    }

    #[test]
    fn missing_action_displays_help() {
        let error = Cli::try_parse_from(["diskmule"]).unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        let rendered = error.to_string();
        assert!(rendered.contains("Usage: diskmule"));
        assert!(rendered.contains("--serve"));
        assert!(rendered.contains("run"));
    }

    #[test]
    fn explicit_help_contains_command_contract() {
        let error = Cli::try_parse_from(["diskmule", "--help"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayHelp);
        let rendered = error.to_string();
        assert!(rendered.contains("Usage: diskmule [OPTIONS] [COMMAND]"));
        assert!(rendered.contains("--serve"));
        assert!(rendered.contains("run"));
        assert!(rendered.contains("ls"));
        assert!(rendered.contains("rm"));
    }

    #[test]
    fn serve_and_subcommand_are_rejected() {
        let cli = Cli::try_parse_from(["diskmule", "--serve", "ls"]).unwrap();
        assert!(cli.action().is_err());
    }
}
