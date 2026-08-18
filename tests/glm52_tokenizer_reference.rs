use std::{env, fs, path::PathBuf};

use diskmule::{
    architecture::{ChatMessage, ChatRole},
    glm52::{
        Glm52Config,
        tokenizer::{Glm52Tokenizer, render_chat},
    },
};

/// Metadata-only oracle for the recommended Colibri grouped-INT4 checkpoint.
///
/// The expected token IDs were captured independently with Hugging Face
/// `tokenizers` 0.21.1 and `add_special_tokens=false`. No tensor shard is
/// needed: point `DISKMULE_GLM52_METADATA` at a directory containing the four
/// small JSON files from the pinned repository.
#[test]
#[ignore = "requires the pinned GLM-5.2 Colibri tokenizer/config metadata"]
fn pinned_colibri_metadata_and_tokenizer_match_hugging_face() {
    let Some(directory) = env::var_os("DISKMULE_GLM52_METADATA").map(PathBuf::from) else {
        eprintln!("skipping: DISKMULE_GLM52_METADATA is unavailable");
        return;
    };

    let config = Glm52Config::from_directory(&directory).unwrap();
    assert_eq!(config.vocabulary_size, 154_880);
    assert_eq!(config.hidden_size, 6_144);
    assert_eq!(config.layer_count, 78);
    assert_eq!(config.attention_heads, 64);
    assert_eq!(config.routed_experts, 256);
    assert_eq!(config.experts_per_token, 8);
    assert_eq!(config.routed_intermediate_size, 2_048);
    assert_eq!(config.first_sparse_layer, 3);
    assert_eq!(config.stop_token_ids, [154_820, 154_827, 154_829]);

    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join("tokenizer_config.json")).unwrap())
            .unwrap();
    assert_eq!(tokenizer_config["model_max_length"], 1_048_576);
    assert_eq!(tokenizer_config["eos_token"], "<|endoftext|>");
    assert_eq!(tokenizer_config["pad_token"], "<|endoftext|>");
    assert!(tokenizer_config["chat_template"].is_null());

    let tokenizer = Glm52Tokenizer::from_directory(&directory).unwrap();
    assert_eq!(tokenizer.vocabulary_size(), 154_856);
    tokenizer
        .validate_model_vocabulary(config.vocabulary_size)
        .unwrap();
    assert_eq!(tokenizer.token_id("[gMASK]"), Some(154_822));
    assert_eq!(tokenizer.token_id("<sop>"), Some(154_824));
    assert_eq!(tokenizer.token_id("<|assistant|>"), Some(154_828));

    let cases: &[(&str, &[u32])] = &[
        ("Hello, world!", &[9_703, 11, 1_879, 0]),
        (
            "Hello, 世界! café 🦀",
            &[9_703, 11, 111_085, 0, 51_609, 11_157, 99, 222],
        ),
        (
            "  lead\nline\ttrail  ",
            &[220, 2_990, 198, 1_056, 25_504, 604, 256],
        ),
        (
            "I'm we've he'll can't I'M",
            &[40, 2_776, 582, 3_003, 566, 3_278, 646, 944, 358, 27_511],
        ),
        (
            "...?!—[]{}<>",
            &[1_112, 25_903, 2_293, 1_294, 6_257, 21_075],
        ),
        (
            "[gMASK]<sop><|user|>Hello<|assistant|><think></think>",
            &[154_822, 154_824, 154_827, 9_703, 154_828, 154_841, 154_842],
        ),
    ];
    for (text, expected) in cases {
        let actual = tokenizer.encode(text).unwrap();
        assert_eq!(&actual, expected, "token mismatch for {text:?}");
        assert_eq!(tokenizer.decode(&actual, true).unwrap(), *text);
    }

    let chat = render_chat(
        &[
            ChatMessage::new(ChatRole::System, "Be terse."),
            ChatMessage::new(ChatRole::User, "Hello, 世界!"),
        ],
        false,
    );
    assert_eq!(
        tokenizer.encode(&chat).unwrap(),
        [
            154_822, 154_824, 154_826, 3_430, 50_205, 13, 154_827, 9_703, 11, 111_085, 0, 154_828,
            154_841, 154_842,
        ]
    );
    assert_eq!(
        tokenizer
            .decode(&tokenizer.encode(&chat).unwrap(), true)
            .unwrap(),
        chat
    );
}
