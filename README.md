# DiskMule

DiskMule is an original Rust inference runtime for Apple
Silicon. Its long-term purpose is to run multiple explicitly supported model
families while keeping very large routed-expert pools on storage. Rust owns the
host runtime and GPU kernels are written in Metal Shading Language.

The current implementation includes the model-management boundary, complete
deterministic Gemma 4 26B CPU and Metal paths, full-checkpoint-validated
GLM-5.2 out-of-core Metal execution over sharded safetensors, and the installed
`qwen3.8:latest` Q4_K_M model through a Qwen3.5 hybrid recurrent-attention
graph with CPU correctness execution and Metal matrix offload. It can inspect
and map GGUF tensors, discover local Ollama models without copying them, safely
manage DiskMule-owned entries, run streaming terminal chat, retain bounded
generation state, and force routed experts through explicit storage-backed
caches. The CLI and Ollama-compatible HTTP server share bounded model workers,
persistent sessions, deterministic sampling, streaming, and cancellation. The
complete roughly 400 GiB GLM-5.2 checkpoint matches Colibri's greedy tokens
while retaining less than 1.6 GB of routed experts in forced one-slot mode.

See [PLAN.md](PLAN.md) for the staged implementation plan and [NOTES.md](NOTES.md)
for the Colibri and TurboFieldfare research behind it.

## Build

DiskMule currently requires a recent stable Rust toolchain:

```bash
cargo build --release
./target/release/diskmule --help
```

During development, replace `./target/release/diskmule` with
`cargo run --quiet --` in the examples below.

## Commands

List DiskMule-managed models and any models discovered in Ollama's local store:

```bash
diskmule ls
```

The output identifies each model's architecture, quantization, size, owner, and
status. `metadata-compatible` means the GGUF metadata and tensor ranges passed
bounded inspection. The architecture runtime performs its stricter complete
configuration and tensor-shape validation when a prompt first loads the model.

Start an interactive chat with a named model, direct local GGUF path, or local
GLM-5.2 safetensors snapshot directory:

```bash
diskmule run gemma4:26b
diskmule run qwen3.8:latest
diskmule run /path/to/model.gguf
diskmule run /path/to/glm-5.2-snapshot
```

`run` prints the resolved model details and starts an EOF-safe terminal loop.
Enter `/clear` to reset history, `/help` for local commands, or `/bye` to exit.
Generation streams through DiskMule's own Metal runtime by default on macOS.
Other platforms and `DISKMULE_BACKEND=cpu` use the explicit CPU correctness
fallback. Ctrl-C cancels an in-flight turn and returns to the prompt. Sampling
defaults to deterministic greedy decoding; temperature, top-k, top-p, and seed
are independently configurable. `DISKMULE_EXPERT_SLOTS` forces routed experts
through that many storage-backed slots per layer; leaving it unset uses the
retained GGUF mapping:

```bash
DISKMULE_CONTEXT=4096 DISKMULE_MAX_TOKENS=64 diskmule run gemma4:26b
DISKMULE_TEMPERATURE=0.8 DISKMULE_TOP_K=40 DISKMULE_TOP_P=0.9 DISKMULE_SEED=42 diskmule run gemma4:26b
DISKMULE_BACKEND=cpu diskmule run gemma4:26b
DISKMULE_EXPERT_SLOTS=4 diskmule run gemma4:26b
DISKMULE_GLM_EXPERT_SLOTS=8 diskmule run /path/to/glm-5.2-snapshot
DISKMULE_GLM_DIRECT=1 diskmule run /path/to/glm-5.2-snapshot
```

GLM dense, attention, router, shared-expert, and output matrices are mapped
directly into Metal on macOS. Selected routed experts are read on demand into a
deterministic bounded per-layer LRU; `DISKMULE_GLM_EXPERT_SLOTS` accepts 1–256
slots per sparse layer and defaults to eight. Expert reads are buffered by
default. `DISKMULE_GLM_DIRECT=1` applies macOS `F_NOCACHE` to the safetensors
descriptors so direct and buffered behavior can be compared under identical
model semantics. This setting is an experiment knob, not an assumed speedup.

Remove a DiskMule-owned model:

```bash
diskmule rm my-model:latest
```

Removal is intentionally strict. DiskMule refuses external Ollama models,
direct local files, ambiguous names, loaded models, absolute or traversing
registry paths, symlinks, paths resolving outside the managed root, and files
shared by multiple registry entries. It never modifies Ollama manifests or
blobs.

Start the foreground local service:

```bash
diskmule --serve
curl http://127.0.0.1:11435/health
```

The server binds to `127.0.0.1:11435`, one port above Ollama's usual local port,
unless `DISKMULE_BIND` explicitly selects another socket. It exposes:

```text
GET /health
GET /api/capabilities
GET /api/loaded
POST /api/chat
```

`GET /api/capabilities` reports each registered architecture's container,
text/multimodal boundary, session support, CPU reference, Metal offload level,
and residency policy. It distinguishes a full Metal graph from hybrid matrix
offload rather than presenting every backend as equivalent.

`POST /api/chat` accepts the Ollama chat request shape. Streaming is the
default and returns newline-delimited JSON; set `"stream": false` for one JSON
response. The optional DiskMule extension `"session": "client-id"` retains and
validates KV prefixes across calls. Clients still send the complete message
history, so an evicted session is reconstructed without changing semantics:

```bash
curl -N http://127.0.0.1:11435/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:26b","session":"demo","messages":[{"role":"user","content":"Hello"}],"options":{"temperature":0,"num_predict":32}}'
```

The service bounds model count, request queues, token buffers, sessions, and
context with `DISKMULE_MAX_MODELS`, `DISKMULE_REQUEST_QUEUE`,
`DISKMULE_TOKEN_BUFFER`, `DISKMULE_MAX_SESSIONS`, and `DISKMULE_CONTEXT`.
Defaults are one loaded model, eight queued requests, 32 buffered token events,
two sessions per model, and a 4096-token context. Session eviction is a
deterministic LRU. Disconnects cancel generation, and Ctrl-C or SIGTERM drains
the HTTP server and joins model-owning worker threads.

## Model locations and ownership

DiskMule uses the platform's application-data directory. On macOS the default
root is:

```text
~/Library/Application Support/com.ArcaneArts.DiskMule
```

Override it with `DISKMULE_HOME`. The managed model root and registry are:

```text
$DISKMULE_HOME/models/
$DISKMULE_HOME/registry.json
```

The version-one registry is intentionally small and explicit. Paths are
relative to the managed model root:

```json
{
  "version": 1,
  "models": [
    {
      "name": "my-model:latest",
      "path": "my-model.gguf"
    }
  ]
}
```

Ollama discovery uses `OLLAMA_MODELS` when set and otherwise checks
`~/.ollama/models`. Its manifests and content-addressed blobs are always
read-only external catalog entries. DiskMule works normally with an empty
external catalog when Ollama is not installed.

Set `RUST_LOG` to override logging, for example:

```bash
RUST_LOG=diskmule=debug diskmule --serve
```

## GGUF inspection boundary

The reader supports GGUF versions 2 and 3. Discovery reads only the header,
metadata, and tensor descriptors; it never loads payloads. Runtime loading adds
a validated read-only mapping for zero-copy tensor views while retaining
bounded positional reads for explicit expert streaming. Validation covers
bounded counts and strings, UTF-8, metadata and tensor types, quantized block
shapes, dimensions, checked size arithmetic, alignment, file bounds, duplicate
names, and overlapping ranges.

The CPU reference implements the tensor encodings used by the local
`gemma4:26b` Q4_K_M model: F32, F16, Q4_K, Q5_0, Q6_K, and Q8_0. Its Gemma
module owns proportional/SWA RoPE, grouped-query attention, Q/K/V norms,
parallel shared and routed experts, sandwich residuals, tied output projection,
softcapping, greedy selection, KV reuse, cancellation checks, and phase timing.
A syntactically valid file using an unimplemented architecture or tensor type
is reported as unsupported rather than being presented as runnable.

The Qwen3.5 module validates hybrid recurrent/full-attention layer schedules,
GPT-2 byte-BPE metadata, Q4_K_M tensors, gated-delta recurrent state, causal
depthwise convolution, partial MRoPE, gated Q/K normalization, dense SwiGLU,
stop IDs, and non-thinking text chat framing. The installed Qwen3.8 checkpoint
is multimodal upstream, but DiskMule currently advertises and implements its
text path only; it does not claim projector or vision support.

The safetensors reader validates single-file or indexed sharded snapshots
without loading tensor payloads. The GLM-5.2 module owns tokenizer/chat framing,
MLA compressed KV state, full/shared DSA selection history, RoPE, routing,
shared and streamed experts, residual order, and stop tokens. CPU execution
supports F32, F16, BF16, block-scaled FP8, and the flat row-INT8 and packed
grouped-INT4 plus `.qs` layouts used by the recommended Colibri checkpoint.
Metal dispatches the same dense and cached-expert encodings directly from shard
mappings or bounded resident slots. Generation profiles include logical tensor
bytes, resident KV/DSA bytes, cache reads/hits/misses/evictions/I/O wait, and
phase timings. Tiny deterministic fixtures prove CPU/dequantized and CPU/Metal
parity. The separate real-model oracle also proves exact Colibri greedy-token
parity and nonzero bounded expert-cache reads, misses, evictions, and residency
on the complete checkpoint.

DSA indexer weights are an optional all-or-none sidecar. A snapshot without
them uses exact dense attention; a partial sidecar is rejected. This permits the
recommended grouped-INT4 checkpoint, which intentionally omits indexer weights,
without pretending sparse long-context selection is active.

On macOS, the Metal backend implements the same encodings and Gemma operations
with a retained no-copy view of the GGUF mapping. Its KV cache is resident and
sliding layers use window-bounded ring buffers. Routed experts can use either
the mapped model or a deterministic per-layer LRU whose reusable aligned slots
are filled by bounded parallel positional reads. Placement does not change
precision or model semantics.

## Verification

Run the complete local gate with:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

The test suite generates tiny GGUF fixtures; it does not copy real model data.
When the local Ollama `gemma4:26b` manifest is present, an environment-aware
smoke test also confirms tensor execution at real offsets, listing, inspection,
refusal of external removal, and unchanged blob metadata.

The expensive pinned correctness oracle is separate from the fast gate:

```bash
cargo test --release --test gemma4_cpu_reference -- --ignored --nocapture
cargo test --release --test gemma4_metal_reference -- --ignored --nocapture
cargo test --release --test gemma4_server_reference -- --ignored --nocapture
DISKMULE_GLM52_METADATA=/path/to/glm-metadata \
  cargo test --test glm52_tokenizer_reference -- --ignored --nocapture
DISKMULE_GLM52_SNAPSHOT=/path/to/glm-snapshot \
DISKMULE_GLM_EXPERT_SLOTS=1 DISKMULE_GLM_DIRECT=1 \
  cargo test --release --test glm52_reference -- --ignored --nocapture
cargo test --release --test qwen35_reference -- --ignored --nocapture
cargo test --release --test qwen35_server_reference -- --ignored --nocapture
```

The GLM metadata oracle needs only `config.json`, `generation_config.json`,
`tokenizer.json`, and `tokenizer_config.json` from recommended Colibri revision
`fd9b461ac7cae4b921470d0db12230c6505bd03c`; it does not download or open tensor
shards. It pins exact Hugging Face tokenizer IDs and validates the checkpoint's
154,880-row padded model vocabulary against its 154,856 defined tokenizer IDs.
Undefined padded output rows are excluded from sampling.

The GLM snapshot oracle first indexes all real shards, checks the exact audited
tensor/dtype counts, and applies the complete ordinary-decode tensor contract
without loading tensor payloads. Its second ignored test runs the exact
seven-token Colibri chat prompt through forced-streaming Metal, requires the
same four greedy IDs `13041, 0, 358, 2776`, and verifies cache and KV metrics.
It is intentionally separate from the routine gate because the checkpoint is
about 400 GiB and the correctness-first dense-attention path is slow. Full
conditions and measurements are in
[`benchmarks/2026-08-18-glm52-m4-max.md`](benchmarks/2026-08-18-glm52-m4-max.md).

For GGUF digest
`7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df`, it
compares DiskMule with a temperature-zero Ollama 0.32.14 run. The four generated
token IDs match exactly (`47610, 1852, 2624, 1852`), and the first token's
normalized log probability differs by about 0.00131 under the documented 0.02
tolerance. The Metal oracle additionally compares every logit with the CPU
reference under a 0.001 tolerance and forces the real model through a one-slot
expert cache. It also verifies token-for-token equality across a persistent KV
session. The server oracle runs simultaneous streaming and non-streaming
clients, disconnect recovery, loaded-model reporting, and graceful shutdown.
Conditions and local M4 Max results are retained in
[`benchmarks/2026-08-18-gemma4-m4-max.md`](benchmarks/2026-08-18-gemma4-m4-max.md).

For Qwen GGUF digest
`f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d`,
DiskMule matches Ollama's four greedy IDs exactly (`11, 353, 2688, 264`) and
matches the configured CLI/server chat token `Hello`. The Qwen server oracle
covers both JSON and NDJSON over the same persistent runtime session. Exact
conditions and first-observed-versus-warm measurements are in
[`benchmarks/2026-08-18-qwen35-m4-max.md`](benchmarks/2026-08-18-qwen35-m4-max.md).

The architecture extension boundary and required conformance evidence are
documented in [CONTRIBUTING_MODELS.md](CONTRIBUTING_MODELS.md).
