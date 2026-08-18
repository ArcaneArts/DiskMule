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

- Implement the Gemma 4 tokenizer and chat rendering.
- Add only the CPU operators required for a deterministic single-token forward
  pass.
- Compare tokenization, selected intermediate tensors, final logits, and greedy
  tokens against trusted Ollama/llama.cpp reference runs.
- Extend to a short multi-token generation with KV-cache reuse.

Exit criterion: deterministic Gemma 4 prompts match the reference within stated
numeric tolerances and choose the same expected tokens.

### Phase 3: Metal Gemma 4 inference

- Establish Metal device, command queue, pipeline, and shared-buffer ownership.
- Implement and validate quantized Q4_K_M GEMV, normalization, attention, router,
  expert, and output kernels as required by the model.
- Compare every new GPU operation against the CPU reference before integrating
  it into full generation.
- Profile prefill, time to first token, decode throughput, memory, and power.

Exit criterion: `diskmule run gemma4:26b` provides a stable interactive chat on
the M4 Max using DiskMule's own Rust runtime and Metal kernels.

### Phase 4: forced expert streaming on Gemma 4

- Separate always-resident tensors from routed-expert tensors.
- Implement reusable page-aligned, Metal-visible expert slots filled through
  bounded parallel range reads.
- Add explicit per-layer cache ownership, pinning, eviction, cancellation, and
  profiling.
- Force a cache too small for full residency and compare outputs with the fully
  resident path.

Exit criterion: resident and streamed Gemma 4 execution remain numerically
equivalent while metrics expose hits, misses, bytes read, and I/O wait.

### Phase 5: chat polish and local server

- Finish the interactive terminal UX, history management, streaming, and clean
  cancellation.
- Implement `diskmule --serve` using the proven runtime API.
- Add request/session limits and streaming HTTP responses.
- Test simultaneous requests, cancellation, malformed input, and graceful
  shutdown.

Exit criterion: both CLI chat and server clients use the same generation engine
and produce equivalent configured results.

### Phase 6: GLM-5.2 and genuine out-of-core operation

- Add safetensors indexing and the GLM-5.2 architecture module.
- Port only the GLM-specific semantics required for correct inference, using
  Colibri as a behavioral reference.
- Apply the explicit cache and I/O system at hundreds-of-gigabytes scale.
- Sweep safe RAM budgets, cache policy, direct versus buffered I/O, queue depth,
  and Metal scheduling on the M4 Max.

Exit criterion: GLM-5.2 generates correct output without fitting the complete
expert pool in unified memory, with reproducible performance logs.

### Phase 7: broaden model support

- Add another materially different dense or MoE architecture.
- Extract shared interfaces only after Gemma 4 and GLM-5.2 reveal genuinely
  common behavior.
- Publish a documented checklist and conformance suite for adding models,
  tensor encodings, and backends.

Exit criterion: adding a model does not require changes to unrelated storage,
server, CLI, or session code unless it introduces a genuinely new capability.

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

## Immediate milestone

Build the Phase 0 and Phase 1 vertical slice so that these commands work:

```text
diskmule ls
diskmule run gemma4:26b
diskmule rm gemma4:26b
diskmule --serve
```

At this milestone, `ls` should identify the Ollama-owned Gemma model,
`run` should resolve and inspect it before reporting that inference is not yet
implemented, `rm` should safely refuse to delete it, and `--serve` should start
a minimal health endpoint. This establishes the public contract and ownership
rules before expensive inference work begins.
