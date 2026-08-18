# Adding a model architecture

DiskMule supports model families explicitly. A model is supported only when its
container, tensor contract, tokenizer, graph, generation state, and at least
one execution backend have correctness evidence. Recognizing an architecture
name or producing plausible text is not conformance.

## Runtime contract

The extension boundary is `LoadedArchitecture` in `src/architecture.rs`. The
shared scheduler sees only this contract:

- immutable capability metadata;
- the actual loaded backend label;
- architecture-owned chat rendering and tokenization;
- creation of an opaque persistent session;
- generation with a shared selector, cancellation flag, token sink, result,
  and profile.

The erased session is downcast only inside the architecture adapter. The
scheduler, CLI, and server do not inspect KV, recurrent, routing, or tokenizer
state. Shared generation types live in `src/generation.rs`, while model-family
semantics stay under `src/<architecture>/`.

Adding an architecture with existing capabilities should require changes only
to its new module, the registry/capability table and loader in
`src/architecture.rs`, and model inspection when its container needs new
metadata validation. It must not add branches to `src/runtime.rs`,
`src/server.rs`, or `src/chat.rs`. A new modality, distributed backend, or
other genuinely new capability may require an explicit contract extension.

## Implementation checklist

- Record the exact upstream architecture identifier, container revision,
  checkpoint identity, license, quantization, and reference runtime version.
- Add strict configuration and tensor-name/shape validation before payload
  execution. Reject missing, duplicate, truncated, overlapping, unaligned, or
  unsupported encodings with structured errors.
- Reuse `TensorSource` for GGUF or `SafeTensorSource` for safetensors. Add a
  format reader only when neither contract can express the container safely.
- Keep discovery metadata-only. Listing a model must not map or read its tensor
  payloads.
- Implement the exact tokenizer, special-token partitioning, byte fallback,
  chat envelope, stop IDs, and supported modalities. Report text-only honestly
  for multimodal checkpoints whose projector is not implemented.
- Implement a scalar CPU reference first, including context bounds,
  architecture-specific state, prompt-prefix behavior, transactional failure,
  cancellation, stable greedy selection, and UTF-8-safe streaming.
- Add only proven shared CPU or Metal operators. Validate every new encoding at
  real nonzero offsets and every GPU operation against CPU with a declared
  tolerance.
- Implement `LoadedArchitecture`, register one `ArchitectureCapabilities`
  record, and add the loader case in `architecture::load`.
- Declare Metal as `full_graph` only when all material graph operations execute
  there. Use `matrix_offload` for a hybrid path.
- Declare `routed_expert_cache` only when misses preserve tensor precision and
  semantics through explicit bounded slots, pinning, eviction, I/O, and
  cancellation. Otherwise declare `mapped`.
- Add bounded allocation checks for vocabulary, context, KV/recurrent state,
  scratch, caches, and any model-specific history.
- Update capability output, README, PLAN, NOTES, and a reproducible benchmark
  artifact. Do not put model blobs, caches, secrets, or user-specific paths in
  git.

## Required conformance evidence

Every new family needs fast synthetic coverage and a pinned real-checkpoint
oracle. The test names can differ, but the evidence may not be skipped:

1. Container metadata, tensor shapes, encodings, bounds, and real nonzero
   offsets.
2. Token IDs, round-trip decoding, chat framing, special tokens, and stops.
3. Deterministic CPU operators, final logits, greedy tokens, and multi-token
   decode.
4. Session extension reuse, divergent-prompt behavior, context boundaries,
   cancellation rollback, and UTF-8 streaming.
5. CPU/Metal operator and full-generation parity for every claimed Metal path.
6. Resident/streamed parity and cache metrics for claimed expert residency.
7. Loading and generation through `GenerationEngine`, not only the raw model
   type.
8. Named or direct-path CLI chat plus streaming and non-streaming HTTP requests
   through `RuntimeService`.
9. A pinned real model identity compared with an authoritative local reference
   at fixed prompt, seed, sampling, and output length.
10. A release-profile record of TTFT, prefill, decode, mapped/read bytes,
    resident state, process memory evidence, cache state, and correctness.

The routine gate is:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

Hardware/model oracles stay ignored so the routine gate remains portable. Run
the relevant ignored tests explicitly before claiming support. The current
third-family Qwen suite is the concrete example:

```bash
cargo test --test qwen35_cpu_reference
cargo test --test qwen35_reference
cargo test --release --test qwen35_reference -- --ignored --nocapture
cargo test --release --test qwen35_server_reference -- --ignored --nocapture
```

## Current conformance matrix

| Contract | Gemma 4 | GLM-5.2 | Qwen 3.5/Qwen3.8 |
| --- | --- | --- | --- |
| Container | GGUF | sharded safetensors (FP8 or Colibri flat-U8) | GGUF |
| CPU reference | Real pinned model | deterministic full-graph fixture | deterministic full-graph fixture |
| Metal | full graph, real parity | FP8/INT8/grouped-INT4 matrix and expert offload, fixture parity | matrix offload, real token oracle |
| Persistent state | KV | MLA KV + DSA history | recurrent + convolution + KV |
| Residency | mapped or explicit expert LRU | explicit expert LRU | mapped |
| CLI/server | real pinned model | shared fixture path; real model unavailable | real pinned model |

GLM-5.2 remains intentionally unclaimed at full-model scale until an authorized
hundreds-of-gigabytes checkpoint is available for the required out-of-core
oracle and performance sweep.
