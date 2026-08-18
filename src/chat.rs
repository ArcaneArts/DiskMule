//! Interactive terminal chat backed by the shared Gemma generation engine.

use std::{
    io::{BufRead, Write},
    path::Path,
};

use crate::{
    architecture::{ChatMessage, ChatRole},
    error::{AppError, Result},
    generation::CancellationFlag,
    runtime::{
        BackendSelection, GenerationEngine, GenerationEvent, GenerationOptions, RuntimeService,
    },
};

const DEFAULT_CONTEXT: usize = 4_096;
const DEFAULT_MAXIMUM_NEW_TOKENS: usize = 64;

pub async fn run_model(
    model: &str,
    runtime: RuntimeService,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    let options = GenerationOptions::from_environment()?;
    writeln!(
        output,
        "DiskMule {} chat. Enter /clear to reset or /bye to exit.",
        runtime.backend().label()
    )?;
    if options.sampling.temperature == 0.0 {
        writeln!(output, "Sampling: deterministic greedy")?;
    } else {
        writeln!(
            output,
            "Sampling: temperature {}, top-k {}, top-p {}, seed {}",
            options.sampling.temperature,
            options.sampling.top_k,
            options.sampling.top_p,
            options.sampling.seed
        )?;
    }
    let mut history = Vec::new();
    let mut line = String::new();
    let mut loading_announced = false;
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

        if !loading_announced {
            writeln!(output, "Loading model...")?;
            output.flush()?;
            loading_announced = true;
        }
        history.push(ChatMessage::new(ChatRole::User, prompt));
        let mut ticket = runtime.generate_in_session(model, "cli", history.clone(), options)?;
        let interrupt = tokio::signal::ctrl_c();
        tokio::pin!(interrupt);
        let mut completed_text = None;
        let mut cancelled = false;
        loop {
            tokio::select! {
                signal = &mut interrupt => {
                    ticket.cancel();
                    cancelled = true;
                    match signal {
                        Ok(()) => writeln!(output, "\nGeneration cancelled.")?,
                        Err(error) => return Err(error.into()),
                    }
                    break;
                }
                event = ticket.recv() => match event {
                    Some(GenerationEvent::Token { text, .. }) => {
                        write!(output, "{text}")?;
                        output.flush()?;
                    }
                    Some(GenerationEvent::Complete(result)) => {
                        completed_text = Some(result.text.trim().to_owned());
                        writeln!(output)?;
                        break;
                    }
                    Some(GenerationEvent::Error { message, cancelled: true }) => {
                        cancelled = true;
                        writeln!(output, "\nGeneration cancelled: {message}")?;
                        break;
                    }
                    Some(GenerationEvent::Error { message, cancelled: false }) => {
                        return Err(AppError::GenerationFailed(message));
                    }
                    None => {
                        return Err(AppError::GenerationFailed(
                            "generation worker closed without a final response".to_owned(),
                        ));
                    }
                }
            }
        }
        if cancelled {
            history.pop();
        } else if let Some(text) = completed_text {
            history.push(ChatMessage::new(ChatRole::Assistant, text));
        }
    }
    Ok(())
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
    let context = DEFAULT_CONTEXT;
    let maximum_new_tokens = DEFAULT_MAXIMUM_NEW_TOKENS;
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
