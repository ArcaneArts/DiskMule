# DiskMule

DiskMule is an original Rust inference runtime under construction for Apple
Silicon. Its long-term purpose is to run multiple explicitly supported model
families while keeping very large routed-expert pools on storage. Rust owns the
host runtime and future GPU kernels will be written in Metal Shading Language.

The current first pass implements the model-management boundary only. It can
inspect GGUF metadata, discover local Ollama models without copying them,
safely manage DiskMule-owned entries, and serve a health endpoint. It does not
perform inference yet.

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
status. `metadata-compatible; inference pending` means the complete GGUF
metadata and tensor ranges passed inspection; it is deliberately not a claim
that inference is implemented.

Inspect a named model or a direct local GGUF path:

```bash
diskmule run gemma4:26b
diskmule run /path/to/model.gguf
```

For this milestone, `run` prints the resolved source, path, architecture,
quantization, GGUF version, tensor count, alignment, and tensor-data offset,
then exits with an explicit `model inference is not implemented yet` error.
Interactive chat is scheduled for the later inference phases in `PLAN.md`.

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

The reader supports GGUF versions 2 and 3 and reads only the header, metadata,
and tensor descriptors. It validates bounded counts and string lengths, UTF-8,
metadata types, tensor types and quantized block shapes, dimensions, checked
size arithmetic, alignment, file bounds, duplicate names, and overlapping
ranges. Tensor payload bytes are not loaded during discovery or `run`.

The first pass recognizes the tensor encodings needed by the local
`gemma4:26b` Q4_K_M model. A syntactically valid model using an unimplemented
architecture or tensor type is reported as unsupported or invalid rather than
being presented as runnable.

## Verification

Run the complete local gate with:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

The test suite generates tiny GGUF fixtures; it does not copy real model data.
When the local Ollama `gemma4:26b` manifest is present, an environment-aware
smoke test also confirms listing, inspection, refusal of external removal, and
unchanged blob metadata.
