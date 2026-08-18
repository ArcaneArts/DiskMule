//! Interactive terminal chat backed by the shared Gemma generation engine.

use std::{
    env,
    io::{BufRead, Write},
    path::Path,
};

use crate::{
    error::{AppError, Result},
    gemma4::{
        ChatMessage, ChatRole, Gemma4Tokenizer,
        cpu::{CancellationFlag, Gemma4CpuModel, GenerationResult, RuntimeError},
        render_chat,
    },
};

#[cfg(target_os = "macos")]
use crate::gemma4::metal::{ExpertResidency, Gemma4MetalModel};

const DEFAULT_CONTEXT: usize = 4_096;
const DEFAULT_MAXIMUM_NEW_TOKENS: usize = 64;

pub fn run_gemma4(path: &Path, input: &mut impl BufRead, output: &mut impl Write) -> Result<()> {
    run_gemma4_with_backend(path, input, output, BackendSelection::from_environment()?)
}

pub fn run_gemma4_cpu(
    path: &Path,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    run_gemma4_with_backend(path, input, output, BackendSelection::Cpu)
}

fn run_gemma4_with_backend(
    path: &Path,
    input: &mut impl BufRead,
    output: &mut impl Write,
    backend: BackendSelection,
) -> Result<()> {
    writeln!(
        output,
        "DiskMule {} chat. Enter /clear to reset or /bye to exit.",
        backend.label()
    )?;
    let mut model = None;
    let context = bounded_environment("DISKMULE_CONTEXT", DEFAULT_CONTEXT, 1, 262_144)?;
    let maximum_new_tokens =
        bounded_environment("DISKMULE_MAX_TOKENS", DEFAULT_MAXIMUM_NEW_TOKENS, 1, 4_096)?;
    let mut history = Vec::new();
    let mut line = String::new();
    loop {
        write!(output, ">>> ")?;
        output.flush()?;
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let prompt = line.trim();
        match prompt {
            "" => continue,
            "/bye" | "/exit" | "/quit" => break,
            "/clear" => {
                history.clear();
                writeln!(output, "Conversation cleared.")?;
                continue;
            }
            "/help" => {
                writeln!(output, "Commands: /clear, /bye, /exit, /quit, /help")?;
                continue;
            }
            _ => {}
        }

        if model.is_none() {
            writeln!(output, "Loading model...")?;
            output.flush()?;
            model = Some(ChatModel::open(path, context, backend)?);
        }
        let model = model.as_ref().expect("model initialized above");
        history.push(ChatMessage::new(ChatRole::User, prompt));
        let rendered = render_chat(&history).map_err(RuntimeError::from)?;
        let tokens = model.tokenizer().encode(&rendered, false);
        let mut stream_error = None;
        let generated = model.generate_greedy(
            &tokens,
            maximum_new_tokens,
            &CancellationFlag::default(),
            |_, piece| {
                if stream_error.is_none()
                    && let Err(error) = write!(output, "{piece}").and_then(|_| output.flush())
                {
                    stream_error = Some(error);
                }
            },
        )?;
        if let Some(error) = stream_error {
            return Err(error.into());
        }
        writeln!(output)?;
        history.push(ChatMessage::new(ChatRole::Assistant, generated.text.trim()));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum BackendSelection {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal {
        residency: ExpertResidency,
    },
}

impl BackendSelection {
    fn from_environment() -> Result<Self> {
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
                    Err(AppError::InvalidConfiguration(
                        "DISKMULE_BACKEND=metal requires macOS".to_owned(),
                    ))
                }
            }
            _ => Err(AppError::InvalidConfiguration(
                "DISKMULE_BACKEND must be auto, metal, or cpu".to_owned(),
            )),
        }
    }

    fn label(self) -> String {
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

enum ChatModel {
    Cpu(Gemma4CpuModel),
    #[cfg(target_os = "macos")]
    Metal(Gemma4MetalModel),
}

impl ChatModel {
    fn open(path: &Path, context: usize, backend: BackendSelection) -> Result<Self> {
        Ok(match backend {
            BackendSelection::Cpu => Self::Cpu(Gemma4CpuModel::open(path, context)?),
            #[cfg(target_os = "macos")]
            BackendSelection::Metal { residency } => Self::Metal(
                Gemma4MetalModel::open_with_expert_residency(path, context, residency)?,
            ),
        })
    }

    fn tokenizer(&self) -> &Gemma4Tokenizer {
        match self {
            Self::Cpu(model) => &model.tokenizer,
            #[cfg(target_os = "macos")]
            Self::Metal(model) => &model.tokenizer,
        }
    }

    fn generate_greedy<F>(
        &self,
        tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        mut on_token: F,
    ) -> std::result::Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
    {
        match self {
            Self::Cpu(model) => {
                model.generate_greedy(tokens, maximum_new_tokens, cancellation, &mut on_token)
            }
            #[cfg(target_os = "macos")]
            Self::Metal(model) => {
                model.generate_greedy(tokens, maximum_new_tokens, cancellation, &mut on_token)
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn expert_residency_from_environment() -> Result<ExpertResidency> {
    let Some(raw) = env::var_os("DISKMULE_EXPERT_SLOTS") else {
        return Ok(ExpertResidency::Mapped);
    };
    let slots_per_layer = raw
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=128).contains(value))
        .ok_or_else(|| {
            AppError::InvalidConfiguration(
                "DISKMULE_EXPERT_SLOTS must be an integer in 1..=128".to_owned(),
            )
        })?;
    Ok(ExpertResidency::Streamed { slots_per_layer })
}

fn bounded_environment(
    name: &'static str,
    default: usize,
    minimum: usize,
    maximum: usize,
) -> Result<usize> {
    let Some(raw) = env::var_os(name) else {
        return Ok(default);
    };
    let value = raw
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            AppError::InvalidConfiguration(format!(
                "{name} must be an integer in {minimum}..={maximum}"
            ))
        })?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::run_gemma4_cpu;

    #[test]
    fn eof_and_local_commands_do_not_load_the_model() {
        for input in ["", "/help\n/clear\n/bye\n"] {
            let mut output = Vec::new();
            run_gemma4_cpu(
                Path::new("this-model-must-not-be-opened.gguf"),
                &mut Cursor::new(input),
                &mut output,
            )
            .unwrap();
            let output = String::from_utf8(output).unwrap();
            assert!(output.contains("DiskMule CPU chat"));
            assert!(!output.contains("Loading model"));
        }
    }

    use std::path::Path;
}
