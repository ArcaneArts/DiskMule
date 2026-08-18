# GLM-5.2 acquisition and out-of-core validation plan

Date: 2026-08-18

Host: Apple M4 Max, 128 GB unified memory

DiskMule commit: record `git rev-parse HEAD` immediately before acquisition

## Checkpoint decision

Use
[`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)
for the first full-model run. DiskMule implements its ordinary-decode row-INT8
and grouped-INT4/gs64 layouts without wrapping Colibri or converting precision
at load time.

The Hugging Face model API reported this inventory on 2026-08-18:

| Component | Bytes | Approximate GiB | Required initially |
| --- | ---: | ---: | --- |
| 141 `out-NNNNN.safetensors` shards | 419,296,541,560 | 390.5 | yes |
| `out-mtp-00000.safetensors` | 9,959,321,520 | 9.3 | no |
| Whole repository | 429,276,080,522 | 399.8 | no |

The official
[`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8)
repository reported 761,025,363,709 bytes (about 709 GiB), so it cannot fit on
the current volume. The current volume has 474 GiB free. A selective
grouped-INT4 download would leave only about 83 GiB; the whole repository would
leave about 74 GiB and bring the volume to roughly 98% utilization. Do not
start either download without explicit authorization. Prefer a dedicated APFS
volume with at least 550 GiB free so the checkpoint, partial downloads, logs,
and healthy filesystem headroom do not compete.

A 141-shard HTTP range audit read only safetensors headers. It found 116,915
unique entries, zero duplicates, and all 58,689 tensors required for ordinary
decode: 463 F32, two row-INT8, and 58,224 grouped-INT4/gs64 matrices, all with
valid flat payload and `.qs` scale geometry. There are zero DSA indexer tensors.
DiskMule's all-or-none indexer contract therefore selects exact dense attention
for this checkpoint; sparse DSA is not claimed unless a complete compatible
sidecar is acquired later.

MTP is deliberately excluded from the first acquisition and validation.
Ordinary decode must be correct and profiled before DiskMule adds or claims MTP.

The experimental
[`mastouri/GLM-5.2-colibri-E8-IQ3-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-E8-IQ3-with-int8-mtp)
alternative reports 299,042,372,874 bytes total. Its 140 ordinary shards are
289,062,833,912 bytes (about 269.2 GiB), which would leave about 205 GiB on the
current volume. It is not an acquisition substitute today: DiskMule does not
implement its 98-byte E8/IQ3 lattice blocks or required activation rotation.
Choose it only as a separate format-development decision; do not misdecode it
as grouped INT4 merely because it also uses flat U8 storage.

## Pre-acquisition safeguards

1. Record `git rev-parse HEAD`, `sw_vers`, `system_profiler SPHardwareDataType`,
   `df -h`, and the target-volume filesystem.
2. Record the Hugging Face API `usedStorage`, file list, and every required
   shard size. Require exactly 141 ordinary shards and reject unexpected names.
3. Capture a read-only fingerprint of Ollama manifests and blobs. Never place
   the GLM snapshot under Ollama's model root.
4. Confirm at least 550 GiB is free on the selected target and that no download
   tool will retain a second full cache copy elsewhere.
5. Download into a dedicated partial directory with resumable transfers, then
   verify every size before one atomic rename to the final snapshot directory.

## Correctness sequence

1. Run `diskmule ls` and open the snapshot metadata. This must index all shards
   and validate the complete GLM tensor contract without reading payloads.
2. Run tokenizer and chat-template fixtures against the checkpoint metadata.
3. Generate one greedy token through the CPU reference and retain token ID,
   logits summary, timings, and tensor/cache byte counters. CPU is a correctness
   oracle, not a throughput target.
4. Run the same prompt through mapped Metal and compare logits under the
   declared `1e-4` fixture tolerance initially; widen a real-model tolerance
   only with recorded numerical evidence, never to excuse a token mismatch.
5. Run at least four greedy tokens and compare exact IDs with an authoritative
   Colibri checkpoint run using identical tokenizer, prompt, and ordinary
   decode settings.
6. Force one expert slot per sparse layer, repeat the prompt, and require the
   same greedy IDs plus Metal/resident numerical parity. Confirm nonzero misses,
   evictions, reads, bytes read, and I/O wait.
7. Exercise CLI streaming, cancellation, clear/exit, non-streaming HTTP,
   streaming HTTP, disconnect cancellation, concurrency bounds, and graceful
   shutdown through the shared runtime.

## Out-of-core and performance sequence

Begin with a total application target near 96 GB, but derive the expert-slot
count from measured `resident_bytes` instead of assuming an expert size. Leave
headroom for macOS, Metal scratch, MLA/DSA state, and file-cache duplication;
stop on rising memory pressure or compression.

Change one variable per run:

1. Buffered reads versus `DISKMULE_GLM_DIRECT=1`.
2. Expert slot count / measured resident budget.
3. Request queue depth and I/O parallelism, if exposed by the measured
   bottleneck.
4. Cold versus warm state, using a documented reboot or cache-conditioning
   procedure rather than claiming that a second run is cold.

For each run retain exact commit and checkpoint inventory, prompt/token IDs,
context and output length, sampling settings, cache mode, TTFT, prefill and
decode throughput, CPU/GPU phase time, hits/misses/evictions, bytes and reads per
token, I/O wait, RSS, physical footprint, memory pressure, compression, and
thermal/power caveats. Record correctness beside performance.

## Completion condition

Phase 6 remains incomplete until a real run proves ordinary GLM-5.2 generation
with the complete expert pool larger than unified memory, explicit bounded
expert residency, correct forced-streaming parity, and reproducible performance
logs. Re-run the full repository verification gate and the Ollama fingerprint
afterward; all work must be committed and the tree clean.
