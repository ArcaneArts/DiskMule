# DiskMule research notes

Last updated: 2026-08-18

## Local reference repositories

The `projects/` directory contains independent, intentionally gitignored clones used as implementation references:

- `projects/colibri` — <https://github.com/JustVugg/colibri>
  - Inspected commit: `4ef9a9920dd4290b4ac26f74e73913fd18379bc9`
- `projects/turbo-fieldfare` — <https://github.com/drumih/turbo-fieldfare>
  - Inspected commit: `9df7b67748b18afdf33ae37626980e0a582c9167`

These are not vendored dependencies. Do not edit them as though their changes belong to DiskMule. Pull or reclone them deliberately when newer upstream behavior needs to be studied.

## Research objective

The motivating target is GLM-5.2 inference on an M4 Max MacBook Pro with 128 GB of unified memory. GLM-5.2 is a 744B-parameter MoE model with about 40B parameters active per token. The complete model cannot fit in unified memory, but its routed experts can be treated as a storage-backed working set.

The current best starting point is Colibri, not a GLM port of TurboFieldfare and not a new generic runtime built from scratch.

Colibri already supplies the high-risk GLM-specific work: model graph, routing semantics, MLA/DSA attention, tokenizer and prompt behavior, quantized model conversion, streamed expert loading, KV persistence, MTP, correctness tests, an explicit expert cache, and an experimental Metal backend. TurboFieldfare should be treated primarily as an optimization playbook and a source of measured Apple Silicon design ideas.

Once GLM works well, a second model can be implemented and genuinely common pieces can then be extracted into a model-agnostic streaming runtime. Designing an "any MoE" abstraction before that risks hiding model-specific differences in attention, router behavior, expert topology, quantization, KV layout, tokenizer, and tensor formats.

## Colibri findings

Useful entry points:

- `projects/colibri/c/colibri.c` — GLM inference, expert cache, placement, I/O pipeline, routing, pinning, and orchestration.
- `projects/colibri/c/backend_metal.mm` — Metal kernels and zero-copy unified-memory integration.
- `projects/colibri/c/expert_store.h` — expert storage bookkeeping.
- `projects/colibri/docs/metal.md` — Apple Silicon build and backend notes.
- `projects/colibri/docs/ENVIRONMENT.md` — tuning controls.
- `projects/colibri/docs/METAL-M5MAX-PERF-REPORT.md` — particularly useful 128 GB tuning data.
- `projects/colibri/docs/benchmarks.md` — throughput and quality data.
- `projects/colibri/docs/FORMATS.md` — quantization/container formats.

Important facts:

- Current Colibri already has a native, experimental Metal backend for GLM-5.2. It runs routed-expert SwiGLU, fused decode attention, and eligible prefill GEMMs on the Apple GPU.
- Apple Silicon lets Metal consume application-owned expert slabs through shared unified-memory buffers, avoiding a discrete-GPU PCIe transfer.
- Resident dense weights are approximately 9.9 GB at INT4.
- The routed-expert pool is approximately 370 GB at INT4. An expert is roughly 19 MB, with 19,456 routed experts across the model.
- The recommended container is grouped INT4/gs64 with an INT8 MTP head. The current Hugging Face repository reports 429,276,080,522 bytes total (about 399.8 GiB): 141 ordinary-decode shards occupy 419,296,541,560 bytes (about 390.5 GiB), and the optional MTP shard occupies 9,959,321,520 bytes (about 9.3 GiB). Avoid the older per-row INT4 containers: Colibri reports a material quality regression (roughly nine percentage points in its evaluation).
- The original FP8 source currently reports 761,025,363,709 bytes (about 709 GiB). The converter is resumable and processes shards without requiring both full FP8 and INT4 copies at once.
- The Metal source in the inspected revision includes grouped-INT4 handling. Some older performance reports still describe grouped Metal support as unfinished, so trust current source and tests over stale prose.
- `make metal-test` should be run before model benchmarks. Grouped-INT4 Metal uses a different accumulation order; near-tie logits may diverge even when numerical error is small.
- Published Metal results vary significantly by cache state, SSD, flags, and model format. An M4 Max 128 GB result in `docs/metal.md` reports about 0.42 tok/s, while a tuned M5 Max 128 GB report reaches about 2.24 tok/s. Do not transfer a headline number across machines without reproducing its exact conditions.

## TurboFieldfare findings

Useful entry points:

- `projects/turbo-fieldfare/docs/SYSTEM_DESIGN.md` — the clearest explanation of ownership, cache slots, I/O, and command-buffer phases.
- `projects/turbo-fieldfare/docs/OPTIMIZATION_JOURNEY.md` — measured wins and failed experiments.
- `projects/turbo-fieldfare/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift` — fixed-slot, parallel `pread` expert streaming.
- `projects/turbo-fieldfare/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift` — decode/prefill orchestration.
- `projects/turbo-fieldfare/Sources/TurboFieldfare/Kernels/MoE/` — routed and shared expert kernels.
- `projects/turbo-fieldfare/Sources/TurboFieldfare/Metal/` — Metal kernels and fusions.
- `projects/turbo-fieldfare/Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` — evidence of the deliberately fixed Gemma architecture contract.

TurboFieldfare is highly optimized but intentionally specialized for Gemma 4 26B-A4B. It hard-codes or validates Gemma-specific dimensions and behavior: 30 layers, 128 experts, top-8 routing, attention layout, norms, RoPE, softcaps, KV layout, quantization, and kernel shapes. Porting GLM into it would replace much of its inference core.

Notable transferable ideas and measurements:

- Use explicit, bounded `pread` into reusable, aligned Metal-visible slots for routed experts.
- Its production expert cache is per-layer LFU with recency as a tie-breaker. Colibri currently combines learned pins with an adaptive per-layer LRU, so LFU is a good controlled experiment rather than an assumed win.
- Execute the always-resident shared-expert branch while routed-expert misses are read.
- Run cached routed experts early, but keep synchronization and buffer ownership coarse and explicit.
- Chunk prompt prefill to bound activation and expert working sets.
- Persistent GPU workgroups materially improved routed MoE throughput in TurboFieldfare.
- TensorOps/Metal matrix primitives can be useful for sufficiently large prefill work, while custom GEMV kernels are better for single-token decode.
- Alignment assumptions must be tested at real tensor offsets, not only offset-zero fixtures.
- End-to-end measurements rejected several plausible optimizations: demand-paged `mmap` for cold experts, unstable read-ahead hints, speculative cross-layer reads, MTLIO under cold conditions, overly fine-grained overlap, and excessive kernel fusion.
- TurboFieldfare measured a cold expert access at about 9.88 ms through demand paging versus about 2.79 ms using explicit `pread`.

## DiskMule Gemma 4 correctness findings

The Phase 2 implementation validated the actual Ollama GGUF rather than relying
only on the reference projects' repacked formats. Important container-specific
details are:

- The text tower is 30 layers with 25 sliding-window and five full-attention
  layers, 2,816 hidden features, 16 query heads, 128 routed experts, top-8
  routing, a 2,112-wide shared expert, and 704-wide routed experts.
- Full-attention layers omit `attn_v.weight` and derive the unnormalized V
  projection from K before applying an unweighted per-head RMSNorm.
- `rope_freqs.weight` contains 128 factors of `1.0` followed by 128 factors of
  `1e30`. This represents Gemma 4's 0.25 proportional partial-RoPE behavior;
  treating all 512 full-attention head dimensions as rotated changes output.
- The shared expert output uses `post_ffw_norm_1.weight`; the combined shared
  plus routed output then uses `post_ffw_norm.weight`. Swapping these still
  produced finite plausible output but failed the exact token oracle.
- The GGUF tokenizer advertises `llama` plus the `gemma4` pre-tokenizer but is
  scored SentencePiece in llama.cpp behavior; the stored merge table is not the
  runtime merge algorithm.

For pinned digest
`7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df`, Ollama
0.32.14 generated token IDs `47610, 1852, 2624, 1852` from a raw BOS prompt at
temperature zero. DiskMule matches all four IDs. Ollama reported a first-token
log probability of `-4.0249696`; DiskMule computes `-4.0262756`, an absolute
difference of about `0.00131` under the Phase 2 tolerance of `0.02`.

The authoritative llama.cpp sources used to verify GGUF quantization,
tokenization, tensor naming, and Gemma graph ordering include
`ggml-quants.c`, `ggml-common.h`, `src/llama-vocab.cpp`,
`src/llama-arch.cpp`, `src/llama-graph.cpp`, and `src/models/gemma4.cpp`.
Reference source is not copied into DiskMule.

## Expert caching strategy

Do not rely exclusively on macOS cached/inactive RAM. The OS file cache is useful as a best-effort second-chance layer, but it is not an expert cache:

- It does not understand layer boundaries, expert IDs, routing frequency, or predicted reuse.
- It is dynamically reclaimed under pressure and gives the runtime no hit-rate guarantee.
- Buffered `pread` may leave an OS-owned file-cache copy in addition to the application-owned Metal-visible slot.
- It cannot provide the stable slot ownership needed while a GPU command buffer is consuming an expert.

The intended hierarchy should be:

1. Always-resident dense weights, router, attention state, KV cache, and reusable scratch.
2. An explicitly pinned hot-expert tier learned from representative routing history.
3. A bounded per-layer adaptive expert cache in Metal-visible unified memory.
4. Remaining experts on NVMe, loaded by explicit parallel reads.
5. The macOS file cache only as an opportunistic bonus when buffered I/O is enabled.

## DiskMule Metal and streamed-expert findings

The Phase 3 Metal backend uses the maintained `objc2-metal` bindings and runtime
MSL compilation. The command-line Metal compiler is not installed on this host,
but framework compilation and dispatch work on the Apple M4 Max. DiskMule's
shared allocations use host-page-aligned no-copy buffers with explicit retained
deallocators. The immutable GGUF is wrapped once through a lifetime-owning
no-copy Metal view, and real tensors dispatch at validated nonzero 64-bit file
offsets.

Metal's fast `tanh` returned non-finite values for the large cubic argument in
Gemma's tanh-GELU approximation even though the activation itself was valid.
Clamping the tanh argument to `[-10, 10]` matches saturated FP32 CPU behavior
and restored full-graph parity. The regression fixture includes gate values up
to magnitude 90.

For routed experts, `ExpertResidency::Mapped` uses the retained GGUF mapping.
`ExpertResidency::Streamed` instead reads the selected expert's exact encoded
gate/up and down ranges into reusable per-layer Metal-visible slots. The two
reads run in bounded parallel, the deterministic LRU chooses only unpinned
slots, and the slot is pinned until both synchronous GPU projections finish.
The one-slot real-model oracle is intentionally smaller than top-8 routing and
therefore forces eviction and storage reads without changing precision or
semantics.

The retained benchmark at commit `72cf6d0` records exact conditions and results
in `benchmarks/2026-08-18-gemma4-m4-max.md`. Warm mapped Metal produced a
326.6 ms TTFT and about 3.12 decode tokens/s for the tiny BOS fixture. Forced
one-slot streaming produced a 380.2 ms TTFT and about 2.80 decode tokens/s,
with zero resident/streamed logit error. These are warm local measurements, not
SSD or cross-machine claims.

## DiskMule GLM-5.2 implementation findings

DiskMule now has an independent GLM architecture path over strict sharded
safetensors metadata. Deterministic fixtures cover the complete ordinary-decode
graph: native byte-level BPE and chat framing, MLA compressed KV, the GLM
adjacent-pair-to-contiguous-halves RoPE layout, full and shared DSA, router
correction bias, normalized top-k routed experts, shared experts, and residual
ordering. CPU readers support F32, F16, BF16, 128-by-128 block-scaled E4M3FN,
flat row-INT8, and packed grouped-INT4 with F32 `.qs` scales. The integer
layouts match Colibri's low-nibble-even convention and scale geometry; logical
shapes come from the GLM tensor contract rather than the intentionally
flattened U8 storage shape. The original F32 tiny fixture predicts
`5, 9, 10, 5`; its final logit zero is approximately `-0.12922081`. A tiny
graph with every eligible matrix quantized matches a separately dequantized F32
oracle within `1e-6`.

The recommended repository's 141 headers were also audited remotely without
downloading payloads. They contain 116,915 unique entries, no duplicates, and
all 58,689 required ordinary-decode tensors: 463 F32, two row-INT8, and 58,224
grouped-INT4/gs64 matrices. They contain no DSA indexer tensors. This is
intentional in Colibri: the optional separately converted indexer sidecar
enables sparse DSA beyond `index_topk`, while its absence keeps exact dense
attention. DiskMule accepts only the complete sidecar or none of it, rejecting
partial indexer state that could otherwise change long-context behavior
silently.

The real tokenizer/config metadata at recommended repository revision
`fd9b461ac7cae4b921470d0db12230c6505bd03c` was downloaded separately from
the tensor payload and compared with Hugging Face `tokenizers` 0.21.1. Exact
IDs match for ASCII, multilingual Unicode, whitespace, contractions,
punctuation, special-token literals, and the complete chat envelope. The audit
also exposed an intentional padded-vocabulary distinction: `config.json`
declares 154,880 embedding/logit rows, while `tokenizer.json` defines the
contiguous IDs 0–154,855. DiskMule accepts tokenizer IDs that are a strict
subset of model rows and masks the 24 undefined padded logits before sampling.
It still rejects any tokenizer ID outside the model vocabulary.

On Metal, dense tensors remain in retained no-copy whole-shard mappings.
Safetensors payload offsets are not assumed to be naturally aligned: F32, F16,
and BF16 kernels reconstruct little-endian values from bytes, avoiding silent
misreads observed with typed loads from an unaligned fixture. Dedicated
row-INT8 and grouped-INT4 kernels consume mapped `.qs` scales without expanding
weights. Owned and mapped operator tests include deliberately nonzero,
unaligned tensor and scale offsets. Routed experts remain governed by the
explicit per-layer bounded LRU. On a miss, exact encoded gate/up/down ranges
and their FP8 or integer scales are read in bounded parallel; cached matrices
and SiLU gating then execute through Metal. Tiny full-graph CPU and Metal
logits and DSA selections, including integer streamed experts, match within
`1e-4`.

GLM generation uses the same runtime, sampling, CLI, HTTP workers, sessions,
prefix reuse, cancellation, and bounded queues as Gemma. Its profile reports
logical tensor bytes, resident MLA/DSA state, expert cache and I/O metrics, and
embedding/attention/feed-forward/output timing. Buffered reads remain the
default. `DISKMULE_GLM_DIRECT=1` applies `F_NOCACHE` to each safetensors shard
descriptor on macOS for a controlled direct-versus-buffered experiment. No
performance conclusion has been drawn from the tiny fixture.

This closes safe implementation work but not Phase 6. There is no full GLM-5.2
checkpoint in the local model stores. The official FP8 repository is about 709
GiB. The recommended grouped-INT4 repository is about 399.8 GiB total;
excluding the unneeded first-pass MTP shard still requires about 390.5 GiB.
With 474 GiB currently free, using the internal volume would leave only about
83 GiB and is not a safe implicit action. Full out-of-core correctness,
cache-budget sweeps, memory-pressure measurements, and reproducible performance
logs require an explicitly authorized acquisition target and storage
commitment.

## DiskMule Qwen3.5/Qwen3.8 findings

The installed Ollama `qwen3.8:latest` model is a suitable third architecture:
its GGUF identifies as `qwen35`, has 64 ordinary layers plus one next-token
prediction layer omitted from ordinary decode, and alternates three recurrent
gated-delta layers with one full-attention layer. The implemented text graph
therefore has 48 recurrent layers and 16 full-attention layers. It uses GPT-2
byte BPE, partial text MRoPE, per-head Q/K normalization, causal depthwise
convolution, recurrent matrix state, output gating, and dense SwiGLU. This is
materially different from both existing graphs.

The upstream checkpoint includes a projector, but DiskMule intentionally
advertises text only. The projector is a distinct Ollama layer and no vision
preprocessor or multimodal contract has been implemented.

The most important graph discrepancy found during the real oracle was the
recurrent key-head broadcast. llama.cpp's fused gated-delta operation maps a
value head to `value_head % key_heads`; using the contiguous repetition common
in grouped-query attention produced stable but incorrect output. With cyclic
mapping, pinned digest
`f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d`
matches Ollama's greedy IDs `11, 353, 2688, 264` exactly. Metal currently
offloads mapped matrices while recurrent and attention vector operations remain
scalar Rust, so capability reporting correctly calls this matrix offload rather
than a full Metal graph.

The Qwen result established the common architecture boundary. Chat messages,
cancellation, results, and profiling are shared, but tokenizer/framing, opaque
session type, stop behavior, and generation remain architecture owned. The
model worker no longer contains an architecture enum or model-specific
branches. Capabilities and the sole load-time registry are in
`src/architecture.rs`; the extension and conformance rules are recorded in
`CONTRIBUTING_MODELS.md`.

## Shared runtime and server findings

Metal model objects remain on dedicated owning threads. CLI and HTTP requests
reach them through the same bounded synchronous request queue and bounded token
channel; an HTTP body or terminal turn dropping its generation ticket flips the
same cancellation flag checked by the graph and expert reader. Model count,
context, queue capacity, buffered token events, and per-model sessions all have
validated upper bounds.

Persistent CPU and Metal sessions compare the next rendered prompt with the
exact cached token IDs and reuse only the common prefix. A changed prompt
truncates every KV layer to the same prefix. Failed and cancelled turns roll
back to the last stable prefix. The final un-forwarded output token is simply
re-evaluated on the next turn, avoiding a hidden extra decode. Session capacity
uses deterministic LRU eviction; because clients retain complete message
history, eviction changes performance rather than prompt semantics. The pinned
Metal oracle split the four-token BOS fixture across two session calls, reused
two prompt tokens on the second call, and matched the stateless IDs exactly.

The real HTTP oracle sends simultaneous streaming and non-streaming Gemma
requests, drops a streaming connection, verifies a successful recovery request,
checks loaded-model status, and confirms graceful shutdown joins the model
worker. The server remains loopback-only by default; a non-loopback bind must be
an explicit `DISKMULE_BIND` configuration.

Colibri already has a learned hot store (`PIN=auto`, `PIN_GB`, `AUTOPIN`) and per-layer cache slots. On macOS it can `mlock` the application-owned cache so the memory compressor does not unexpectedly reclaim/compress the hot working set.

Compare buffered reads with `DIRECT=1`. On macOS, Colibri maps the direct setting to `F_NOCACHE`. Direct I/O gives the application more control and can avoid duplicate file-cache residency, but performance is SSD- and workload-dependent. Measure both paths instead of assuming direct I/O always wins.

## Initial M4 Max operating point

Do not allocate all 128 GB. CPU, GPU, Metal buffers, cache slabs, KV state, applications, and macOS share the same physical memory and bandwidth. Memory compression is especially harmful because it consumes CPU and SoC power while the GPU is active.

Start with `--ram 96`, then sweep approximately 105 and 110 GB while watching `memory_pressure`, process physical footprint, compressor activity, cache hit rate, disk bytes per token, and GPU idle time. An M5 Max report found 110 GB safe and 120 GB over the compression knee; the M4 Max must be measured independently.

A useful starting command is:

```bash
cd projects/colibri/c
make colibri METAL=1
make metal-test

COLI_MODEL=/path/to/glm52_i4 \
COLI_METAL=1 \
COLI_NO_OMP_TUNE=1 \
PIPE=1 \
PIPE_WORKERS=8 \
DIRECT=1 \
MTP=0 \
PROF=1 \
./coli run "A fixed benchmark prompt" --ram 96 --ngen 256
```

The M5 Max report found that active OpenMP spinning could steal the shared SoC power budget from Metal and cause a large regression. `COLI_NO_OMP_TUNE=1` plus `PIPE=1 PIPE_WORKERS=8` was the successful pairing there. Re-measure on the M4 Max rather than treating it as a permanent universal default.

## Benchmark discipline

Change one variable at a time and retain raw logs. Record:

- Exact Colibri commit and model container/format
- macOS version and build flags
- Prompt, context length, requested output length, temperature, and random seed
- Cold versus warm state
- Frozen `.coli_usage` routing history or a deliberately fresh history
- `--ram`, `--cap`, direct/buffered I/O, `PIPE` and worker count
- MTP state and acceptance rate
- Time to first token and steady-state decode throughput separately
- Expert hit rate, misses/token, disk GB/token, read bandwidth, and I/O wait
- CPU/GPU phase times, power/thermal state, RSS, physical footprint, and compression
- Output/correctness comparison, not merely kernel microbenchmark speed

Suggested first experiment matrix:

1. `DIRECT=0` versus `DIRECT=1`.
2. RAM budgets of 96, 105, and 110 GB.
3. Cache caps around 16, 24, 32, and 33 slots per layer.
4. Fresh routing history versus a frozen representative learned history.
5. Existing LRU plus pins versus an LFU-with-recency implementation.
6. MTP off versus on only after ordinary decode is understood.

Use deterministic/greedy generation for performance comparisons, but remember that parallel floating-point reductions can still change near-tie token choices. Throughput comparisons should not require byte-identical long generations unless the tested path claims that stronger invariant.

## Likely optimization order

Use Colibri's profiler to select work rather than porting optimizations speculatively:

- Disk-bound: improve expert hit rate, explicit cache policy, read queue depth, file layout, read coalescing, direct I/O, or multi-drive splitting/mirroring.
- GPU-matmul-bound: test persistent MoE workgroups, vectorization aligned to the real format, and batched resident-expert dispatch.
- Metal-submission-bound: reduce CPU/GPU synchronization and combine compatible work into fewer command buffers.
- Prefill-bound: test chunking, staged Metal matrix primitives, and TensorOps-style attention.
- Memory-pressure-bound: eliminate duplicate residency and cap every slot/scratch arena before adding more caching.

The main system invariant should be that placement changes speed, not model semantics: a cache miss must load the same expert at the same precision rather than substitute, skip, or silently requantize it.
