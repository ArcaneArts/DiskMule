//! Qwen3.5 GPT-2 byte-BPE tokenizer and text-only chat framing.

use std::collections::HashMap;

use crate::{
    gemma4::{ChatMessage, ChatRole},
    gguf::{GgufFile, MetadataArray, MetadataValue},
};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen35TokenizerError {
    #[error("missing Qwen3.5 tokenizer metadata {0:?}")]
    MissingMetadata(&'static str),

    #[error("Qwen3.5 tokenizer metadata {key:?} has the wrong type; expected {expected}")]
    MetadataType {
        key: &'static str,
        expected: &'static str,
    },

    #[error("invalid Qwen3.5 tokenizer: {0}")]
    Invalid(String),

    #[error("Qwen3.5 tokenizer cannot encode byte-level symbol {0:?}")]
    UnknownSymbol(String),

    #[error("token ID {id} is outside the Qwen3.5 vocabulary")]
    InvalidTokenId { id: u32 },

    #[error("Qwen3.5 chat system message must be first")]
    SystemMessageOrder,
}

#[derive(Debug, Clone)]
struct Token {
    text: String,
    control: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen35Tokenizer {
    vocabulary: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    atomized: Vec<(u32, String)>,
    tokens: Vec<Token>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    stop_ids: Vec<u32>,
}

impl Qwen35Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, Qwen35TokenizerError> {
        require_string(gguf, "tokenizer.ggml.model", "gpt2")?;
        require_string(gguf, "tokenizer.ggml.pre", "qwen35")?;
        let texts = string_array(gguf, "tokenizer.ggml.tokens")?;
        let types = i32_array(gguf, "tokenizer.ggml.token_type")?;
        let merges = string_array(gguf, "tokenizer.ggml.merges")?;
        if texts.len() != types.len() {
            return Err(Qwen35TokenizerError::Invalid(format!(
                "token_type has {} entries for {} tokens",
                types.len(),
                texts.len()
            )));
        }
        let mut vocabulary = HashMap::with_capacity(texts.len());
        let mut tokens = Vec::with_capacity(texts.len());
        let mut atomized = Vec::new();
        for (id, (text, kind)) in texts.iter().zip(types).enumerate() {
            let id = u32::try_from(id).map_err(|_| {
                Qwen35TokenizerError::Invalid("vocabulary exceeds u32 token IDs".to_owned())
            })?;
            if vocabulary.insert(text.clone(), id).is_some() {
                return Err(Qwen35TokenizerError::Invalid(format!(
                    "duplicate token text {text:?}"
                )));
            }
            let control = *kind == 3;
            if matches!(*kind, 3 | 4) {
                atomized.push((id, text.clone()));
            }
            tokens.push(Token {
                text: text.clone(),
                control,
            });
        }
        atomized.sort_by(|left, right| {
            right
                .1
                .len()
                .cmp(&left.1.len())
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let Some((left, right)) = merge.split_once(' ') else {
                return Err(Qwen35TokenizerError::Invalid(format!(
                    "merge {rank} has no separator"
                )));
            };
            if left.is_empty() || right.is_empty() {
                return Err(Qwen35TokenizerError::Invalid(format!(
                    "merge {rank} contains an empty side"
                )));
            }
            merge_ranks.insert((left.to_owned(), right.to_owned()), rank);
        }
        let byte_to_char = byte_to_unicode();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(byte, character)| (*character, byte as u8))
            .collect();
        let eos = u32_metadata(gguf, "tokenizer.ggml.eos_token_id")?;
        if eos as usize >= tokens.len() {
            return Err(Qwen35TokenizerError::Invalid(format!(
                "EOS token {eos} is outside the vocabulary"
            )));
        }
        let mut stop_ids = [
            Some(eos),
            vocabulary.get("<|endoftext|>").copied(),
            vocabulary.get("<|im_end|>").copied(),
            vocabulary.get("<|fim_pad|>").copied(),
            vocabulary.get("<|repo_name|>").copied(),
            vocabulary.get("<|file_sep|>").copied(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        stop_ids.sort_unstable();
        stop_ids.dedup();
        Ok(Self {
            vocabulary,
            merges: merge_ranks,
            atomized,
            tokens,
            byte_to_char,
            char_to_byte,
            stop_ids,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Qwen35TokenizerError> {
        let mut output = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let mut next: Option<(usize, u32, &str)> = None;
            for (id, token) in &self.atomized {
                if let Some(relative) = text[offset..].find(token) {
                    let position = offset + relative;
                    if next.is_none_or(|(best, _, current)| {
                        position < best || (position == best && token.len() > current.len())
                    }) {
                        next = Some((position, *id, token));
                    }
                }
            }
            let end = next.map_or(text.len(), |(position, _, _)| position);
            if end > offset {
                self.encode_chunk(&text[offset..end], &mut output)?;
            }
            let Some((position, id, control)) = next else {
                break;
            };
            output.push(id);
            offset = position + control.len();
        }
        Ok(output)
    }

    pub fn decode(
        &self,
        ids: &[u32],
        include_special: bool,
    ) -> Result<String, Qwen35TokenizerError> {
        let mut bytes = Vec::new();
        for id in ids {
            bytes.extend(self.decode_token_bytes(*id, include_special)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn decode_token_bytes(
        &self,
        id: u32,
        include_special: bool,
    ) -> Result<Vec<u8>, Qwen35TokenizerError> {
        let token = self
            .tokens
            .get(id as usize)
            .ok_or(Qwen35TokenizerError::InvalidTokenId { id })?;
        if token.control {
            return Ok(if include_special {
                token.text.as_bytes().to_vec()
            } else {
                Vec::new()
            });
        }
        token
            .text
            .chars()
            .map(|character| {
                self.char_to_byte
                    .get(&character)
                    .copied()
                    .ok_or_else(|| Qwen35TokenizerError::UnknownSymbol(character.to_string()))
            })
            .collect()
    }

    pub fn vocabulary_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.vocabulary.get(text).copied()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.tokens
            .get(id as usize)
            .is_some_and(|token| token.control)
    }

    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    fn encode_chunk(&self, chunk: &str, output: &mut Vec<u32>) -> Result<(), Qwen35TokenizerError> {
        for (start, end) in pretoken_ranges(chunk) {
            self.encode_piece(&chunk.as_bytes()[start..end], output)?;
        }
        Ok(())
    }

    fn encode_piece(
        &self,
        bytes: &[u8],
        output: &mut Vec<u32>,
    ) -> Result<(), Qwen35TokenizerError> {
        let encoded = bytes
            .iter()
            .map(|byte| self.byte_to_char[*byte as usize])
            .collect::<String>();
        if let Some(id) = self.vocabulary.get(&encoded) {
            output.push(*id);
            return Ok(());
        }
        let mut symbols = encoded
            .chars()
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for index in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = self
                    .merges
                    .get(&(symbols[index].clone(), symbols[index + 1].clone()))
                    .copied()
                    && best.is_none_or(|(_, best_rank)| rank < best_rank)
                {
                    best = Some((index, rank));
                }
            }
            let Some((index, _)) = best else {
                break;
            };
            let right = symbols.remove(index + 1);
            symbols[index].push_str(&right);
        }
        for symbol in symbols {
            output.push(
                *self
                    .vocabulary
                    .get(&symbol)
                    .ok_or_else(|| Qwen35TokenizerError::UnknownSymbol(symbol.clone()))?,
            );
        }
        Ok(())
    }
}

pub fn render_chat(
    messages: &[ChatMessage],
    enable_thinking: bool,
) -> Result<String, Qwen35TokenizerError> {
    let mut output = String::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System && index != 0 {
            return Err(Qwen35TokenizerError::SystemMessageOrder);
        }
        match message.role {
            ChatRole::System => {
                output.push_str("<|im_start|>system\n");
                output.push_str(message.content.trim());
                output.push_str("<|im_end|>\n");
            }
            ChatRole::User => {
                output.push_str("<|im_start|>user\n");
                output.push_str(message.content.trim());
                output.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                output.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
                output.push_str(message.content.trim());
                output.push_str("<|im_end|>\n");
            }
        }
    }
    output.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        output.push_str("<think>\n");
    } else {
        output.push_str("<think>\n\n</think>\n\n");
    }
    Ok(output)
}

fn string_array<'a>(
    gguf: &'a GgufFile,
    key: &'static str,
) -> Result<&'a [String], Qwen35TokenizerError> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::ArrayValues(MetadataArray::String(values))) => Ok(values),
        Some(_) => Err(Qwen35TokenizerError::MetadataType {
            key,
            expected: "captured string array",
        }),
        None => Err(Qwen35TokenizerError::MissingMetadata(key)),
    }
}

fn i32_array<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a [i32], Qwen35TokenizerError> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::ArrayValues(MetadataArray::I32(values))) => Ok(values),
        Some(_) => Err(Qwen35TokenizerError::MetadataType {
            key,
            expected: "captured I32 array",
        }),
        None => Err(Qwen35TokenizerError::MissingMetadata(key)),
    }
}

fn require_string(
    gguf: &GgufFile,
    key: &'static str,
    expected: &'static str,
) -> Result<(), Qwen35TokenizerError> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::String(value)) if value == expected => Ok(()),
        Some(MetadataValue::String(value)) => Err(Qwen35TokenizerError::Invalid(format!(
            "{key} is {value:?}; expected {expected:?}"
        ))),
        Some(_) => Err(Qwen35TokenizerError::MetadataType {
            key,
            expected: "string",
        }),
        None => Err(Qwen35TokenizerError::MissingMetadata(key)),
    }
}

fn u32_metadata(gguf: &GgufFile, key: &'static str) -> Result<u32, Qwen35TokenizerError> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::U32(value)) => Ok(*value),
        Some(_) => Err(Qwen35TokenizerError::MetadataType {
            key,
            expected: "U32",
        }),
        None => Err(Qwen35TokenizerError::MissingMetadata(key)),
    }
}

fn pretoken_ranges(text: &str) -> Vec<(usize, usize)> {
    let characters = text.char_indices().collect::<Vec<_>>();
    let mut ranges = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let start = index;
        let character = characters[index].1;
        if character == '\'' && index + 1 < characters.len() {
            let tail = characters[index + 1].1.to_ascii_lowercase();
            if index + 2 < characters.len() {
                let pair = [tail, characters[index + 2].1.to_ascii_lowercase()];
                if matches!(pair, ['r', 'e'] | ['v', 'e'] | ['l', 'l']) {
                    index += 3;
                    push_range(&characters, text.len(), start, index, &mut ranges);
                    continue;
                }
            }
            if matches!(tail, 's' | 't' | 'm' | 'd') {
                index += 2;
                push_range(&characters, text.len(), start, index, &mut ranges);
                continue;
            }
        }
        let mut cursor = index;
        if !is_letter(character) && !is_newline(character) && !is_number(character) {
            if cursor + 1 < characters.len() && is_letter(characters[cursor + 1].1) {
                cursor += 1;
            } else {
                cursor = usize::MAX;
            }
        }
        if cursor != usize::MAX && is_letter(characters[cursor].1) {
            while cursor < characters.len() && is_letter(characters[cursor].1) {
                cursor += 1;
            }
            index = cursor;
            push_range(&characters, text.len(), start, index, &mut ranges);
            continue;
        }
        if is_number(character) {
            index += 1;
            while index < characters.len() && index - start < 3 && is_number(characters[index].1) {
                index += 1;
            }
            push_range(&characters, text.len(), start, index, &mut ranges);
            continue;
        }
        cursor = index;
        if character == ' '
            && cursor + 1 < characters.len()
            && is_punctuation_group(characters[cursor + 1].1)
        {
            cursor += 1;
        }
        if cursor < characters.len() && is_punctuation_group(characters[cursor].1) {
            cursor += 1;
            while cursor < characters.len() && is_punctuation_group(characters[cursor].1) {
                cursor += 1;
            }
            while cursor < characters.len() && is_newline(characters[cursor].1) {
                cursor += 1;
            }
            index = cursor;
            push_range(&characters, text.len(), start, index, &mut ranges);
            continue;
        }
        if character.is_whitespace() {
            cursor = index;
            while cursor < characters.len() && characters[cursor].1.is_whitespace() {
                cursor += 1;
            }
            if let Some(last_newline) = (index..cursor)
                .rev()
                .find(|candidate| is_newline(characters[*candidate].1))
            {
                index = last_newline + 1;
            } else if cursor < characters.len() {
                index = cursor.saturating_sub(1).max(index + 1);
            } else {
                index = cursor;
            }
            push_range(&characters, text.len(), start, index, &mut ranges);
            continue;
        }
        index += 1;
        push_range(&characters, text.len(), start, index, &mut ranges);
    }
    ranges
}

fn push_range(
    characters: &[(usize, char)],
    text_length: usize,
    start: usize,
    end: usize,
    ranges: &mut Vec<(usize, usize)>,
) {
    ranges.push((
        characters[start].0,
        characters.get(end).map_or(text_length, |item| item.0),
    ));
}

fn is_letter(character: char) -> bool {
    character.is_alphabetic()
}

fn is_number(character: char) -> bool {
    character.is_numeric()
}

fn is_newline(character: char) -> bool {
    matches!(character, '\r' | '\n')
}

fn is_punctuation_group(character: char) -> bool {
    !character.is_whitespace() && !is_letter(character) && !is_number(character)
}

fn byte_to_unicode() -> [char; 256] {
    let mut direct = [false; 256];
    direct[33..=126].fill(true);
    direct[161..=172].fill(true);
    direct[174..=255].fill(true);
    let mut mapped = ['\0'; 256];
    let mut extra = 0_u32;
    for byte in 0..256 {
        let codepoint = if direct[byte] {
            byte as u32
        } else {
            let codepoint = 256 + extra;
            extra += 1;
            codepoint
        };
        mapped[byte] = char::from_u32(codepoint).expect("GPT-2 byte map is valid Unicode");
    }
    mapped
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        gemma4::{ChatMessage, ChatRole},
        gguf::{GgufFile, MetadataArray, MetadataValue},
    };

    use super::{Qwen35Tokenizer, byte_to_unicode, render_chat};

    #[test]
    fn encodes_merges_special_tokens_and_unicode_bytes() {
        let gguf = tokenizer_gguf();
        let tokenizer = Qwen35Tokenizer::from_gguf(&gguf).unwrap();
        let ids = tokenizer.encode("<|im_start|>hex 世界").unwrap();
        assert_eq!(ids[0], 258);
        assert_eq!(ids[1], 257);
        assert_eq!(
            tokenizer.decode(&ids, true).unwrap(),
            "<|im_start|>hex 世界"
        );
        assert_eq!(tokenizer.decode(&[258], false).unwrap(), "");
        assert!(tokenizer.stop_token_ids().contains(&259));
    }

    #[test]
    fn renders_text_chat_without_claiming_multimodal_support() {
        let messages = [
            ChatMessage::new(ChatRole::System, "Be terse."),
            ChatMessage::new(ChatRole::User, " Hi "),
            ChatMessage::new(ChatRole::Assistant, "Hello"),
        ];
        assert_eq!(
            render_chat(&messages, false).unwrap(),
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    fn tokenizer_gguf() -> GgufFile {
        let mapping = byte_to_unicode();
        let mut tokens = mapping
            .into_iter()
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        tokens.extend([
            "he".to_owned(),
            "hex".to_owned(),
            "<|im_start|>".to_owned(),
            "<|im_end|>".to_owned(),
            "<|endoftext|>".to_owned(),
        ]);
        let mut types = vec![1; tokens.len()];
        types[258..].fill(3);
        GgufFile {
            version: 3,
            alignment: 32,
            data_offset: 0,
            file_len: 0,
            metadata: BTreeMap::from([
                (
                    "tokenizer.ggml.model".to_owned(),
                    MetadataValue::String("gpt2".to_owned()),
                ),
                (
                    "tokenizer.ggml.pre".to_owned(),
                    MetadataValue::String("qwen35".to_owned()),
                ),
                (
                    "tokenizer.ggml.tokens".to_owned(),
                    MetadataValue::ArrayValues(MetadataArray::String(tokens)),
                ),
                (
                    "tokenizer.ggml.token_type".to_owned(),
                    MetadataValue::ArrayValues(MetadataArray::I32(types)),
                ),
                (
                    "tokenizer.ggml.merges".to_owned(),
                    MetadataValue::ArrayValues(MetadataArray::String(vec![
                        "h e".to_owned(),
                        "he x".to_owned(),
                    ])),
                ),
                (
                    "tokenizer.ggml.eos_token_id".to_owned(),
                    MetadataValue::U32(259),
                ),
            ]),
            tensors: Vec::new(),
        }
    }
}
