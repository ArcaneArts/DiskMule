//! GLM-5.2 byte-level BPE tokenizer and text-only chat template.

use std::{collections::HashMap, fs, path::Path, path::PathBuf};

use serde::Deserialize;

use crate::gemma4::{ChatMessage, ChatRole};

const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TOKEN_ID: usize = 1 << 21;

#[derive(Debug, thiserror::Error)]
pub enum Glm52TokenizerError {
    #[error("could not read GLM-5.2 tokenizer {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("GLM-5.2 tokenizer {path} is {size} bytes; maximum is {maximum}")]
    TooLarge {
        path: PathBuf,
        size: u64,
        maximum: u64,
    },

    #[error("invalid GLM-5.2 tokenizer {path}: {message}")]
    InvalidJson { path: PathBuf, message: String },

    #[error("unsupported GLM-5.2 tokenizer contract: {0}")]
    Unsupported(String),

    #[error("tokenizer contains duplicate token ID {0}")]
    DuplicateId(u32),

    #[error("tokenizer contains duplicate added-token text {0:?}")]
    DuplicateAdded(String),

    #[error("tokenizer cannot encode byte-level symbol {0:?}")]
    UnknownSymbol(String),

    #[error("token ID {id} is outside the tokenizer vocabulary")]
    InvalidTokenId { id: u32 },
}

#[derive(Debug, Deserialize)]
struct TokenizerFile {
    model: TokenizerModel,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Debug, Deserialize)]
struct TokenizerModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<serde_json::Value>,
    #[serde(default)]
    ignore_merges: bool,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Clone)]
struct AddedTokenInfo {
    id: u32,
    content: String,
}

#[derive(Debug, Clone)]
struct DecodedToken {
    text: String,
    added: bool,
    special: bool,
}

#[derive(Debug, Clone)]
pub struct Glm52Tokenizer {
    vocabulary: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    rank_bpe: bool,
    added: Vec<AddedTokenInfo>,
    decoded: Vec<Option<DecodedToken>>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl Glm52Tokenizer {
    pub fn from_directory(directory: impl AsRef<Path>) -> Result<Self, Glm52TokenizerError> {
        Self::from_file(directory.as_ref().join("tokenizer.json"))
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Glm52TokenizerError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|source| Glm52TokenizerError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        if metadata.len() > MAX_TOKENIZER_BYTES {
            return Err(Glm52TokenizerError::TooLarge {
                path: path.to_path_buf(),
                size: metadata.len(),
                maximum: MAX_TOKENIZER_BYTES,
            });
        }
        let bytes = fs::read(path).map_err(|source| Glm52TokenizerError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let raw: TokenizerFile =
            serde_json::from_slice(&bytes).map_err(|error| Glm52TokenizerError::InvalidJson {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: TokenizerFile) -> Result<Self, Glm52TokenizerError> {
        if raw.model.kind != "BPE" {
            return Err(Glm52TokenizerError::Unsupported(format!(
                "model.type must be BPE, got {:?}",
                raw.model.kind
            )));
        }
        if !raw.model.ignore_merges || raw.model.byte_fallback {
            return Err(Glm52TokenizerError::Unsupported(
                "expected ignore_merges=true and byte_fallback=false".to_owned(),
            ));
        }
        let mut maximum_id = 0_usize;
        for id in raw
            .model
            .vocab
            .values()
            .copied()
            .chain(raw.added_tokens.iter().map(|token| token.id))
        {
            let id = id as usize;
            if id > MAX_TOKEN_ID {
                return Err(Glm52TokenizerError::Unsupported(format!(
                    "token ID {id} exceeds {MAX_TOKEN_ID}"
                )));
            }
            maximum_id = maximum_id.max(id);
        }
        let mut decoded = vec![None; maximum_id + 1];
        for (text, id) in &raw.model.vocab {
            let slot = &mut decoded[*id as usize];
            if slot.is_some() {
                return Err(Glm52TokenizerError::DuplicateId(*id));
            }
            *slot = Some(DecodedToken {
                text: text.clone(),
                added: false,
                special: false,
            });
        }
        let mut added = Vec::with_capacity(raw.added_tokens.len());
        let mut added_names = std::collections::HashSet::new();
        for token in raw.added_tokens {
            if !added_names.insert(token.content.clone()) {
                return Err(Glm52TokenizerError::DuplicateAdded(token.content));
            }
            let slot = &mut decoded[token.id as usize];
            if slot.is_some()
                && raw
                    .model
                    .vocab
                    .get(&token.content)
                    .is_none_or(|id| *id != token.id)
            {
                return Err(Glm52TokenizerError::DuplicateId(token.id));
            }
            *slot = Some(DecodedToken {
                text: token.content.clone(),
                added: true,
                special: token.special,
            });
            added.push(AddedTokenInfo {
                id: token.id,
                content: token.content,
            });
        }
        added.sort_by(|left, right| {
            right
                .content
                .len()
                .cmp(&left.content.len())
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut merges = HashMap::new();
        for (rank, merge) in raw.model.merges.iter().enumerate() {
            let pair = match merge {
                serde_json::Value::Array(parts) if parts.len() == 2 => {
                    let left = parts[0].as_str();
                    let right = parts[1].as_str();
                    match (left, right) {
                        (Some(left), Some(right)) => (left.to_owned(), right.to_owned()),
                        _ => {
                            return Err(Glm52TokenizerError::Unsupported(format!(
                                "merge {rank} is not a string pair"
                            )));
                        }
                    }
                }
                serde_json::Value::String(pair) => {
                    let Some((left, right)) = pair.split_once(' ') else {
                        return Err(Glm52TokenizerError::Unsupported(format!(
                            "merge {rank} has no separator"
                        )));
                    };
                    if left.is_empty() || right.is_empty() {
                        return Err(Glm52TokenizerError::Unsupported(format!(
                            "merge {rank} contains an empty side"
                        )));
                    }
                    (left.to_owned(), right.to_owned())
                }
                _ => {
                    return Err(Glm52TokenizerError::Unsupported(format!(
                        "merge {rank} is malformed"
                    )));
                }
            };
            merges.insert(pair, rank);
        }
        let byte_to_char = byte_to_unicode();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(byte, character)| (*character, byte as u8))
            .collect();
        Ok(Self {
            vocabulary: raw.model.vocab,
            rank_bpe: merges.is_empty(),
            merges,
            added,
            decoded,
            byte_to_char,
            char_to_byte,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Glm52TokenizerError> {
        let mut output = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let mut next: Option<(usize, &AddedTokenInfo)> = None;
            for token in &self.added {
                if let Some(relative) = text[offset..].find(&token.content) {
                    let position = offset + relative;
                    if next.is_none_or(|(best, current)| {
                        position < best
                            || (position == best && token.content.len() > current.content.len())
                    }) {
                        next = Some((position, token));
                    }
                }
            }
            let end = next.map_or(text.len(), |(position, _)| position);
            if end > offset {
                self.encode_chunk(&text[offset..end], &mut output)?;
            }
            let Some((position, token)) = next else {
                break;
            };
            output.push(token.id);
            offset = position + token.content.len();
        }
        Ok(output)
    }

    pub fn decode(
        &self,
        ids: &[u32],
        include_special: bool,
    ) -> Result<String, Glm52TokenizerError> {
        let mut bytes = Vec::new();
        for id in ids {
            let token = self
                .decoded
                .get(*id as usize)
                .and_then(Option::as_ref)
                .ok_or(Glm52TokenizerError::InvalidTokenId { id: *id })?;
            if token.added {
                if include_special || !token.special {
                    bytes.extend_from_slice(token.text.as_bytes());
                }
            } else {
                for character in token.text.chars() {
                    let byte = self
                        .char_to_byte
                        .get(&character)
                        .ok_or_else(|| Glm52TokenizerError::UnknownSymbol(character.to_string()))?;
                    bytes.push(*byte);
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|token| token.content == text)
            .map(|token| token.id)
            .or_else(|| self.vocabulary.get(text).copied())
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.decoded
            .get(id as usize)
            .and_then(Option::as_ref)
            .is_some_and(|token| token.special)
    }

    fn encode_chunk(&self, chunk: &str, output: &mut Vec<u32>) -> Result<(), Glm52TokenizerError> {
        for (start, end) in pretoken_ranges(chunk) {
            self.encode_piece(&chunk.as_bytes()[start..end], output)?;
        }
        Ok(())
    }

    fn encode_piece(&self, bytes: &[u8], output: &mut Vec<u32>) -> Result<(), Glm52TokenizerError> {
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
                let rank = if self.rank_bpe {
                    let combined = format!("{}{}", symbols[index], symbols[index + 1]);
                    self.vocabulary.get(&combined).map(|id| *id as usize)
                } else {
                    self.merges
                        .get(&(symbols[index].clone(), symbols[index + 1].clone()))
                        .copied()
                };
                if let Some(rank) = rank
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
                    .ok_or_else(|| Glm52TokenizerError::UnknownSymbol(symbol.clone()))?,
            );
        }
        Ok(())
    }
}

pub fn render_chat(messages: &[ChatMessage], enable_thinking: bool) -> String {
    let mut output = String::from("[gMASK]<sop>");
    for message in messages {
        match message.role {
            ChatRole::System => {
                output.push_str("<|system|>");
                output.push_str(&message.content);
            }
            ChatRole::User => {
                output.push_str("<|user|>");
                output.push_str(&message.content);
            }
            ChatRole::Assistant => {
                output.push_str("<|assistant|><think></think>");
                output.push_str(message.content.trim());
            }
        }
    }
    if enable_thinking {
        output.push_str("<|assistant|><think>");
    } else {
        output.push_str("<|assistant|><think></think>");
    }
    output
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
    let byte_start = characters[start].0;
    let byte_end = characters.get(end).map_or(text_length, |item| item.0);
    ranges.push((byte_start, byte_end));
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
    use std::fs;

    use serde_json::{Map, json};
    use tempfile::TempDir;

    use crate::gemma4::{ChatMessage, ChatRole};

    use super::{Glm52Tokenizer, byte_to_unicode, render_chat};

    #[test]
    fn atomizes_added_tokens_and_round_trips_unicode_bytes() {
        let temp = TempDir::new().unwrap();
        write_tokenizer(temp.path());
        let tokenizer = Glm52Tokenizer::from_directory(temp.path()).unwrap();
        let text = "[gMASK]<sop>Hello, 世界! 1234";
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(&ids[..2], &[300, 301]);
        assert_eq!(tokenizer.decode(&ids, true).unwrap(), text);
        assert_eq!(tokenizer.token_id("<|assistant|>"), Some(304));
        assert!(tokenizer.is_special(304));
        assert_eq!(tokenizer.decode(&[304], false).unwrap(), "");
    }

    #[test]
    fn applies_ranked_merges_when_a_piece_is_not_already_a_token() {
        let temp = TempDir::new().unwrap();
        write_tokenizer(temp.path());
        let tokenizer = Glm52Tokenizer::from_directory(temp.path()).unwrap();
        let ids = tokenizer.encode("hex").unwrap();
        assert_eq!(ids, [310, byte_to_unicode()[b'x' as usize] as u32]);
    }

    #[test]
    fn renders_the_official_text_chat_envelope() {
        let messages = [
            ChatMessage::new(ChatRole::System, "Be terse."),
            ChatMessage::new(ChatRole::User, "Hi"),
            ChatMessage::new(ChatRole::Assistant, " Hello "),
            ChatMessage::new(ChatRole::User, "Again"),
        ];
        assert_eq!(
            render_chat(&messages, false),
            "[gMASK]<sop><|system|>Be terse.<|user|>Hi<|assistant|><think></think>Hello<|user|>Again<|assistant|><think></think>"
        );
        assert!(render_chat(&messages, true).ends_with("<|assistant|><think>"));
    }

    fn write_tokenizer(directory: &std::path::Path) {
        let mapping = byte_to_unicode();
        let mut vocabulary = Map::new();
        for (id, character) in mapping.into_iter().enumerate() {
            vocabulary.insert(character.to_string(), json!(id));
        }
        vocabulary.insert("he".to_owned(), json!(310));
        let added = [
            (300, "[gMASK]"),
            (301, "<sop>"),
            (302, "<|system|>"),
            (303, "<|user|>"),
            (304, "<|assistant|>"),
            (305, "<think>"),
            (306, "</think>"),
        ]
        .map(|(id, content)| json!({"id": id, "content": content, "special": true}));
        let file = json!({
            "model": {
                "type": "BPE",
                "vocab": vocabulary,
                "merges": [["h", "e"]],
                "ignore_merges": true,
                "byte_fallback": false
            },
            "added_tokens": added
        });
        fs::write(
            directory.join("tokenizer.json"),
            serde_json::to_vec(&file).unwrap(),
        )
        .unwrap();
    }
}
