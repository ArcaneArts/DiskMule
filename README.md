# DiskMule

DiskMule is an original Rust inference runtime for Apple
Silicon. Its long-term purpose is to run multiple explicitly supported model
families while keeping very large routed-expert pools on storage. Rust owns the
host runtime and GPU kernels will be written in Metal Shading Language.

The current implementation includes the model-management boundary plus complete
deterministic Gemma 4 26B CPU and Metal paths. It can inspect and map GGUF
tensors, discover local Ollama models without copying them, safely manage
DiskMule-owned entries, run streaming terminal chat, retain bounded Metal KV
state, and force routed experts through an explicit storage-backed cache. The
shared CLI/server generation surface is the next active phase.

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

Start an interactive chat with a named model or direct local GGUF path:

```bash
diskmule run gemma4:26b
diskmule run /path/to/model.gguf
```

`run` prints the resolved model details and starts an EOF-safe terminal loop.
Enter `/clear` to reset history, `/help` for local commands, or `/bye` to exit.
Generation is greedy and streams through DiskMule's own Metal runtime by
default on macOS. Other platforms and `DISKMULE_BACKEND=cpu` use the explicit
CPU correctness fallback. `DISKMULE_EXPERT_SLOTS` forces routed experts through
that many storage-backed slots per layer; leaving it unset uses the retained
GGUF mapping. Temporary development controls bound generation while the richer
Phase 5 CLI surface is being built:

```bash
DISKMULE_CONTEXT=4096 DISKMULE_MAX_TOKENS=64 diskmule run gemma4:26b
DISKMULE_BACKEND=cpu diskmule run gemma4:26b
DISKMULE_EXPERT_SLOTS=4 diskmule run gemma4:26b
```

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

The first-pass server binds only to `127.0.0.1:11435`, one port above Ollama's
usual local port, and currently exposes:

```text
GET /health
```

It returns JSON containing the health status, service name, and DiskMule
version. Ctrl-C and SIGTERM trigger graceful shutdown.

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
```

For GGUF digest
`7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df`, it
compares DiskMule with a temperature-zero Ollama 0.32.14 run. The four generated
token IDs match exactly (`47610, 1852, 2624, 1852`), and the first token's
normalized log probability differs by about 0.00131 under the documented 0.02
tolerance. The Metal oracle additionally compares every logit with the CPU
reference under a 0.001 tolerance and forces the real model through a one-slot
expert cache. Conditions and local M4 Max results are retained in
[`benchmarks/2026-08-18-gemma4-m4-max.md`](benchmarks/2026-08-18-gemma4-m4-max.md).
