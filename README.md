# DiskMule

DiskMule is an original Rust inference runtime for Apple
Silicon. Its long-term purpose is to run multiple explicitly supported model
families while keeping very large routed-expert pools on storage. Rust owns the
host runtime and GPU kernels will be written in Metal Shading Language.

The current implementation includes the model-management boundary, complete
deterministic Gemma 4 26B CPU and Metal paths, and fixture-validated GLM-5.2
CPU/Metal execution over sharded safetensors. It can inspect and map GGUF
tensors, discover local Ollama models without copying them, safely manage
DiskMule-owned entries, run streaming terminal chat, retain bounded Metal KV
state, and force routed experts through explicit storage-backed caches. The
CLI and Ollama-compatible HTTP server share bounded model workers, persistent
KV sessions, deterministic sampling, streaming, and cancellation. A real
hundreds-of-gigabytes GLM-5.2 checkpoint is still required before claiming
full-model correctness or out-of-core performance.

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
GET /api/loaded
POST /api/chat
```

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

The safetensors reader validates single-file or indexed sharded snapshots
without loading tensor payloads. The GLM-5.2 module owns tokenizer/chat framing,
MLA compressed KV state, full/shared DSA selection history, RoPE, routing,
shared and streamed experts, residual order, and stop tokens. CPU execution
supports F32, F16, BF16, and block-scaled FP8; Metal dispatches the same dense
and cached-expert encodings. Generation profiles include logical tensor bytes,
resident KV/DSA bytes, cache reads/hits/misses/evictions/I/O wait, and phase
timings. Tiny deterministic fixtures prove CPU/Metal parity but are not treated
as full GLM evidence.

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
```

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
