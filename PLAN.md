# DiskMule implementation plan

Last updated: 2026-08-18

## Product goal

DiskMule will be an original Rust inference runtime for Apple Silicon. It will
support multiple model families through explicit architecture modules while
sharing model storage, tensor formats, quantization, caching, scheduling,
Metal execution, sampling, and serving infrastructure.

"Any model" means any model architecture and tensor encoding for which
DiskMule has an implementation. Model-specific attention, routing, norms,
tokenization, prompt rendering, multimodal handling, and tensor naming must not
be hidden behind an inaccurate universal model graph.

The first supported model will be the locally available `gemma4:26b` GGUF.
It is small enough to establish correctness and Metal performance without
storage pressure. GLM-5.2 will be the second architecture and the primary test
of out-of-core expert streaming on a 128 GB M4 Max.

Colibri and TurboFieldfare are implementation and performance references only.
DiskMule will not be a wrapper around either project or around Ollama.

## User-facing CLI

The initial command surface is:

```text
diskmule --serve
diskmule run <model>
diskmule ls
diskmule rm <model>
```

### `diskmule run <model>`

- Resolve a model by DiskMule name, local path, or an optionally discovered
  Ollama model name such as `gemma4:26b`.
- Start an interactive terminal chat similar to `ollama run`.
- Apply the model's tokenizer, chat template, stop tokens, and default sampling
  configuration.
- Stream generated text to the terminal as tokens become available.
- Preserve conversation history until the user exits or clears the session.
- Support clean interruption and terminal restoration.
- Later accept non-interactive prompts and generation overrides without
  changing the basic chat experience.

### `diskmule --serve`

- Start a foreground local HTTP inference server.
- Bind to `127.0.0.1:11435` by default so the first pass remains local-only and
  does not conflict with Ollama's usual port.
- Reuse exactly the same model registry, loader, sessions, and generation
  engine as the CLI.
- Stream generation responses and expose health and loaded-model information.
- Begin with a small documented API, then add OpenAI- and/or Ollama-compatible
  endpoints where compatibility does not distort the runtime design.
- Bound concurrent sessions, KV-cache growth, queued requests, and model memory.
- Shut down cleanly without leaving model locks or temporary files behind.

### `diskmule ls`

- List models managed by DiskMule.
- When Ollama discovery is enabled, also list compatible models found in the
  local Ollama store.
- Show at least name, architecture, quantization, on-disk size, source, and
  compatibility status.
- Clearly distinguish `diskmule` models from read-only `ollama` models.

### `diskmule rm <model>`

- Remove only models owned by DiskMule.
- Never delete or modify an Ollama-owned blob, manifest, or model alias.
- If the name resolves only to an Ollama model, explain that it is externally
  managed and refuse the removal.
- Resolve and display the exact target before deletion, and prevent deletion of
  a model that is currently loaded.

## Ollama interoperability

Ollama model discovery is useful but optional. The runtime must continue to
work when Ollama is absent or its storage layout changes.

The first implementation should:

1. Detect Ollama through its local model manifests or CLI metadata.
2. Treat discovered models as read-only external entries in DiskMule's model
   registry.
3. Resolve manifest names to their underlying GGUF blobs without copying large
   files.
4. Parse and validate the GGUF itself before declaring it compatible.
5. Ignore unsupported or incomplete models with a useful reason in `ls`.
6. Keep Ollama prompt metadata separate from DiskMule's canonical architecture
   implementation and verify template behavior with deterministic tests.

The existing local `gemma4:26b` resolves to a roughly 17 GB Q4_K_M GGUF with
25.8B parameters. It will be the first interoperability fixture, but tests must
not hard-code its user-specific absolute blob path.

DiskMule-managed models use a versioned `registry.json` containing names and
paths relative to the managed model root. Ollama entries and direct local paths
remain external, read-only catalog records and are never adopted implicitly.

## Runtime architecture

Use Rust for the host runtime and Metal Shading Language for GPU kernels. Access
Metal from Rust through maintained Objective-C framework bindings; introduce a
small Objective-C++ shim only where bindings are incomplete or make ownership
unclear.

Suggested workspace boundaries:

```text
crates/
  diskmule-cli        command parsing and interactive chat
  diskmule-server     local HTTP server and streaming responses
  diskmule-runtime    generation loop, sessions, KV cache, and scheduling
  diskmule-model      architecture traits and shared model types
  diskmule-gemma4     Gemma 4 graph, tokenizer, and tensor mapping
  diskmule-glm52      GLM-5.2 graph, tokenizer, and tensor mapping
  diskmule-format     GGUF and safetensors readers
  diskmule-storage    range reads, residency, expert cache, and model registry
  diskmule-metal      Metal device management and kernel dispatch
  diskmule-cpu        reference kernels and correctness fallback
metal/                MSL kernels
```

Keep these boundaries conceptual until the first vertical slice proves which
ones warrant separate crates. Avoid creating a large abstraction framework
before Gemma 4 inference works end to end.

## Model and backend contracts

Architecture modules own:

- Configuration validation and tensor-name mapping
- Forward-pass ordering and model-specific operations
- Attention, positional encoding, norms, and residual behavior
- Dense, routed, and shared-expert topology
- Tokenizer, chat template, stop conditions, and special tokens
- Vision or other modality preprocessing when supported

Shared runtime components own:

- Tensor metadata, shapes, views, and quantization dispatch
- GGUF and safetensors indexing and bounded range reads
- CPU and Metal operator implementations
- Resident-weight placement and explicit expert-cache slot ownership
- KV-cache allocation and session limits
- Sampling, token streaming, cancellation, and profiling
- Model naming, discovery, ownership, and lifecycle

Placement is allowed to change performance only. A cache miss must load the same
weights at the same precision and produce results within the backend's declared
numerical tolerance.

## Delivery phases

### Phase 0: repository and executable skeleton

- Create the Rust workspace and a single `diskmule` binary.
- Implement command parsing and stable help text for `--serve`, `run`, `ls`, and
  `rm`.
- Define error reporting, logging, configuration paths, and model ownership
  rules.
- Add unit tests for parsing and destructive-command safeguards.

Exit criterion: every command runs and reports intentional placeholder state;
`rm` cannot affect external files.

### Phase 1: model registry and Ollama discovery

- Define DiskMule's model registry and managed storage layout.
- Parse GGUF headers, metadata, tensor descriptors, types, shapes, and offsets.
- Discover local Ollama aliases without taking ownership of their files.
- Make `diskmule ls` show `gemma4:26b` with source and compatibility details.
- Resolve `diskmule run gemma4:26b` to a validated external GGUF.

Exit criterion: model listing and resolution work without loading tensor data or
copying the Ollama blob.

### Phase 2: Gemma 4 CPU correctness slice

Status: complete on 2026-08-18 for the pinned local `gemma4:26b` GGUF.

- Implement the Gemma 4 tokenizer and chat rendering.
- Add only the CPU operators required for a deterministic single-token forward
  pass.
- Compare tokenization, selected intermediate tensors, final logits, and greedy
  tokens against trusted Ollama/llama.cpp reference runs.
- Extend to a short multi-token generation with KV-cache reuse.

Exit criterion: deterministic Gemma 4 prompts match the reference within stated
numeric tolerances and choose the same expected tokens.

Evidence: DiskMule's read-only mapped CPU graph executes the complete Gemma 4
text block and reuses a bounded KV cache across decode. Against Ollama 0.32.14
and exact GGUF digest
`7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df`, the
four-token greedy sequence is exactly `47610, 1852, 2624, 1852`; the first
token's normalized log probability is `-4.0262756` versus Ollama's
`-4.0249696`. A live `diskmule run gemma4:26b` CPU session streamed a bounded
response and handled clear and exit commands.

### Phase 3: Metal Gemma 4 inference

Status: complete on 2026-08-18 for the pinned local `gemma4:26b` GGUF.

- Establish Metal device, command queue, pipeline, and shared-buffer ownership.
- Implement and validate quantized Q4_K_M GEMV, normalization, attention, router,
  expert, and output kernels as required by the model.
- Compare every new GPU operation against the CPU reference before integrating
  it into full generation.
- Profile prefill, time to first token, decode throughput, memory, and power.

Exit criterion: `diskmule run gemma4:26b` provides a stable interactive chat on
the M4 Max using DiskMule's own Rust runtime and Metal kernels.

Evidence: DiskMule owns the Metal device, runtime-compiled shader library,
pipelines, command queue, page-aligned shared allocations, mapped-weight view,
and synchronous command-buffer lifetimes. F32, F16, Q5_0, Q8_0, Q4_K, and Q6_K
GEMV plus Gemma norms, RoPE, attention, softmax, routing, experts, reductions,
and output operations have CPU parity fixtures. On the real 17 GB model, the
maximum CPU/Metal error across all logits is `0.00014209747` under a `0.001`
tolerance, and four greedy tokens match exactly. Metal weight residency reduced
the observed four-token run from roughly 2.43 seconds with transient weight
copies to roughly 1.31 seconds with the retained no-copy GGUF mapping.

### Phase 4: forced expert streaming on Gemma 4

Status: complete on 2026-08-18 for mapped and forced one-slot Metal execution.

- Separate always-resident tensors from routed-expert tensors.
- Implement reusable page-aligned, Metal-visible expert slots filled through
  bounded parallel range reads.
- Add explicit per-layer cache ownership, pinning, eviction, cancellation, and
  profiling.
- Force a cache too small for full residency and compare outputs with the fully
  resident path.

Exit criterion: resident and streamed Gemma 4 execution remain numerically
equivalent while metrics expose hits, misses, bytes read, and I/O wait.

Evidence: each layer owns a deterministic LRU cache of reusable page-aligned
Metal-visible gate/up and down slots. Misses issue two bounded parallel
positional reads, slots remain pinned through synchronous GPU consumption, and
cancellation is checked before and after I/O. Fully mapped and explicit
streaming modes use identical precision and kernels. A real one-slot-per-layer
run produced zero resident/streamed logit error and the same four greedy tokens;
its profile recorded hits, misses, evictions, reads, bytes, I/O wait, expert
residency, and bounded ring-buffer KV residency. Detailed conditions and raw
figures are retained in `benchmarks/2026-08-18-gemma4-m4-max.md`.

### Phase 5: chat polish and local server

Status: complete on 2026-08-18 for Gemma 4 CLI and local HTTP clients.

- Finish the interactive terminal UX, history management, streaming, and clean
  cancellation.
- Implement `diskmule --serve` using the proven runtime API.
- Add request/session limits and streaming HTTP responses.
- Test simultaneous requests, cancellation, malformed input, and graceful
  shutdown.

Exit criterion: both CLI chat and server clients use the same generation engine
and produce equivalent configured results.

Evidence: CLI and HTTP both submit the same `ChatMessage` and
`GenerationOptions` types to bounded, model-owning worker threads. Named and
direct-path resolution share the catalog. CPU and Metal sessions retain KV
state only across an exact token prefix, roll back after failure or
cancellation, and are capped by deterministic per-model LRU eviction. The CLI
streams text, preserves history, supports clear/help/exit/EOF, and returns to a
clean prompt after Ctrl-C cancellation. Sampling supports deterministic greedy
or seeded temperature/top-k/top-p selection. The loopback-default server
provides health, loaded-model status, and Ollama-style streaming or
non-streaming `/api/chat`, with bounded model, context, request, token, and
session resources. Fast tests cover malformed requests and shutdown. A pinned
real-model oracle passed simultaneous JSON and NDJSON clients, disconnect
cancellation and recovery, loaded-model reporting, and worker-joining graceful
shutdown. CLI and both server modes produced the same configured greedy
`Hello` token.

### Phase 6: GLM-5.2 and genuine out-of-core operation

Status: safe implementation complete for deterministic tiny CPU/Metal fixtures,
including the recommended Colibri grouped-INT4 container; full-model validation
and performance sweeps remain blocked on an authorized GLM-5.2 checkpoint
acquisition with adequate storage headroom.

- Add safetensors indexing and the GLM-5.2 architecture module.
- Port only the GLM-specific semantics required for correct inference, using
  Colibri as a behavioral reference.
- Apply the explicit cache and I/O system at hundreds-of-gigabytes scale.
- Sweep safe RAM budgets, cache policy, direct versus buffered I/O, queue depth,
  and Metal scheduling on the M4 Max.

Exit criterion: GLM-5.2 generates correct output without fitting the complete
expert pool in unified memory, with reproducible performance logs.

Current evidence: strict single-file and sharded safetensors indexing validates
the configuration and complete tensor contract without loading payloads. The
architecture module implements tokenizer/chat framing, MLA compressed KV,
full/shared DSA, adjacent-pair RoPE semantics, dense and MoE SwiGLU, correction
bias routing, shared experts, residual ordering, stops, prefix reuse, and
cancellation. CPU supports F32/F16/BF16/block-scaled FP8 plus the flat-U8
row-INT8 and grouped-INT4 layouts used by Colibri (`.qs` F32 scales, including
gs64). The loader derives and strictly validates logical matrix geometry from
the expected GLM tensor contract, packed bytes, and scale cardinality. On
macOS, sharded dense tensors execute from retained no-copy Metal mappings and
selected routed experts execute through Metal after bounded parallel reads
into an explicit per-layer LRU. Owned and mapped Metal fixtures cover nonzero,
unaligned payload offsets. A tiny graph with every eligible matrix quantized
matches its separately dequantized F32 CPU oracle within `1e-6`; CPU/Metal
logits and greedy choices, including cache-resident routed experts, match
within `1e-4`. Profiles
expose phase timing, logical tensor bytes, resident generation state, expert
residency, reads, bytes, cache events, and I/O wait. Buffered expert reads are
the default and `DISKMULE_GLM_DIRECT=1` enables macOS `F_NOCACHE` for controlled
comparison. This is necessary fixture evidence, not the exit criterion's
required full-model evidence.

A read-only HTTP range audit of all 141 recommended-checkpoint headers found
116,915 unique entries and no duplicates. Every one of DiskMule's 58,689
ordinary-decode tensors is present with the expected logical geometry: 463 F32,
two row-INT8, and 58,224 grouped-INT4/gs64 matrices. The repository intentionally
contains no DSA indexer tensors. DiskMule therefore treats the indexer sidecar as
strictly all-or-none: a completely absent sidecar uses exact dense attention,
matching Colibri's behavior, while a partial sidecar is rejected.

The recommended checkpoint's real 19 MB tokenizer and configuration metadata
at revision `fd9b461ac7cae4b921470d0db12230c6505bd03c` also pass a pinned
metadata-only oracle. DiskMule matches Hugging Face `tokenizers` 0.21.1 exactly
across ordinary text, Unicode, whitespace, contractions, punctuation, special
tokens, and the complete chat envelope. This audit found that the checkpoint
correctly pads its embedding/logit vocabulary to 154,880 rows while defining
tokenizer IDs only through 154,855. The loader now accepts that strict subset
and masks undefined padded rows before sampling instead of rejecting the valid
checkpoint or attempting to decode undefined IDs.

Acquisition audit on 2026-08-18: the official `zai-org/GLM-5.2-FP8` repository
reports `761,025,363,709` bytes (about 709 GiB), which cannot fit in the current
466 GiB of free space. The recommended
`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp` repository reports
`429,276,080,522` bytes (about 399.8 GiB). Ordinary decode needs its 141 main
safetensors shards, `419,296,541,560` bytes (about 390.5 GiB), plus small
configuration/tokenizer files; the optional MTP shard adds `9,959,321,520`
bytes (about 9.3 GiB) and is not part of the first validation. Even a selective
main-model acquisition would leave only about 75 GiB free on the current
volume, while the whole repository would leave about 66 GiB and take the
volume to roughly 98% utilization. DiskMule will not make that commitment
without explicit authorization or a suitably sized external target. The exact
acquisition and validation sequence is retained in
`benchmarks/2026-08-18-glm52-acquisition-plan.md`.

### Phase 7: broaden model support

Status: complete on 2026-08-18 for the pinned local `qwen3.8:latest` Q4_K_M
GGUF and the shared architecture contract.

- Add another materially different dense or MoE architecture.
- Extract shared interfaces only after Gemma 4 and GLM-5.2 reveal genuinely
  common behavior.
- Publish a documented checklist and conformance suite for adding models,
  tensor encodings, and backends.

Exit criterion: adding a model does not require changes to unrelated storage,
server, CLI, or session code unless it introduces a genuinely new capability.

Evidence: Qwen3.5/Qwen3.8 adds a hybrid graph with 48 recurrent gated-delta
layers, 16 full-attention layers, causal depthwise convolution, recurrent
matrix state, text MRoPE, GPT-2 byte BPE, and dense SwiGLU. Its state and graph
are materially different from Gemma's sliding/full attention plus routed MoE
and GLM's MLA/DSA plus streamed MoE. For pinned digest
`f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d`,
the Metal matrix-offload path matches Ollama's four greedy token IDs exactly:
`11, 353, 2688, 264`. Real CLI and HTTP chat paths match Ollama's configured
one-token `Hello` response.

After the third graph exposed the common boundary, the runtime replaced its
model/session enums with the object-safe `LoadedArchitecture` contract and an
opaque architecture-owned session. The shared scheduler now handles sampling,
cancellation, queues, sessions, streaming, and profiling without model-family
branches. A capability registry reports containers, text/multimodal scope,
CPU/Metal level, persistent sessions, and residency through
`GET /api/capabilities`. `CONTRIBUTING_MODELS.md` defines the extension
checklist, fast conformance requirements, real-model oracle, and current
three-family matrix. Qwen synthetic tests cover the tensor contract,
tokenizer, complete CPU graph, recurrent/session behavior, cancellation,
UTF-8 streaming, CPU/Metal parity, and shared engine. Pinned ignored tests cover
the real graph and CLI/server worker paths.

## Verification rules

- Use fixed prompts, seeds, sampling settings, and exact model identities.
- Retain raw benchmark and profiler output.
- Compare tokenization and logits, not only plausible generated prose.
- Test tensor offsets and alignment using real GGUF data, not just synthetic
  offset-zero buffers.
- Record cold and warm cache behavior separately.
- Track time to first token and steady-state decode independently.
- Record RSS, physical footprint, compression, expert hit rate, disk bytes per
  token, I/O wait, and GPU timing.
- Change one performance variable at a time.
- Never claim support based only on successful metadata parsing.

## Current milestone

All safe Phase 6 implementation work, including Colibri grouped-INT4 CPU,
Metal, and streamed-expert execution, and Phase 7 are complete. The remaining
goal blocker is the unavailable full GLM-5.2 checkpoint and the authorization
required for its hundreds-of-gigabytes storage commitment, out-of-core oracle,
and controlled performance sweep. These commands work:

```text
diskmule ls
diskmule run gemma4:26b
diskmule run qwen3.8:latest
diskmule rm gemma4:26b
diskmule --serve
```

`ls` identifies the Ollama-owned models, `run` provides cancellable Metal chat
on supported Apple Silicon with an explicit CPU fallback, `rm` safely refuses
external deletion, and `--serve` provides bounded streaming and non-streaming
chat. Qwen3.8 is verified through the same CLI and server. GLM fixture-backed
safetensors, CPU, Metal, session, profiling, and storage-backed expert
semantics are implemented. A full Phase 6 claim still requires the real
out-of-core model and explicit authorization for any large download or disk
commitment.
