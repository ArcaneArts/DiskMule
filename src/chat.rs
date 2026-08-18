//! Interactive terminal chat backed by the shared Gemma generation engine.

use std::{
    env,
    io::{BufRead, Write},
    path::Path,
};

use crate::{
    error::{AppError, Result},
    gemma4::{ChatMessage, ChatRole, cpu::CancellationFlag},
    runtime::{BackendSelection, GenerationEngine, GenerationOptions},
};

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
    let mut engine = None;
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

        if engine.is_none() {
            writeln!(output, "Loading model...")?;
            output.flush()?;
            engine = Some(GenerationEngine::open_gemma4(
                path.display().to_string(),
                path,
                context,
                backend,
            )?);
        }
        let engine = engine.as_ref().expect("model initialized above");
        history.push(ChatMessage::new(ChatRole::User, prompt));
        let mut stream_error = None;
        let generated = engine.generate_chat(
            &history,
            GenerationOptions {
                maximum_new_tokens,
                ..GenerationOptions::default()
            },
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
