# 1Cat v37 attention dataflow

The `NINFER_VOLTA_PACKED_PREFILL` implementation is an adaptation of
the packed-GQA, blocked-prefix, FP32 online-state design documented by
[1Cat-vLLM v37](https://github.com/1CatAI/1Cat-vLLM/tree/a3498d463db549c17a58c04f4455694e0416ca54/csrc/attention/sm70_v37)
at commit `a3498d463db549c17a58c04f4455694e0416ca54`.
Its source declares BSD-3-Clause; the original Flash-V100 license is retained
in `LICENSE`. NVIDIA CUTLASS remains the separately pinned CMake dependency
and retains its own license.

The NInfer adapter is `src/ops/launcher/volta_packed_prefill.cu`. It preserves
the design's six-head packing, SM70 M128/N256/K32 GEMM geometry, FP32 QK/PV,
FP32 softmax maxima/sums and online numerator merge. FP16 probabilities remain
Tensor Core operands. The independent attention oracle, not another kernel,
determines numerical acceptance.

This is not an exact copy or a claim of the upstream kernel's performance.
The adaptation batches four KV groups in the grid's z dimension. A private
CUTLASS QK epilogue computes each N256 tile's exact masked FP32 maximum,
emits tile-relative FP16 probabilities, and records FP32 tile maxima/sums.
A PV CTA covers all 256 value channels for 128 packed query rows; it reduces
the tile statistics, rescales probabilities while loading the MMA operand,
and updates the FP32 online numerator directly in its epilogue. Each key
block needs two kernel launches across all four groups. Query packing and
final BF16 output conversion each require one additional launch.

There are two FP16 probability boundaries: conversion of tile-relative
probabilities in the QK epilogue, then conversion after rescaling them to the
key-block maximum in the PV load transform. Tile sums follow v37 by summing
the first emitted FP16 probabilities in FP32. Subsequent tile/block rescaling,
maxima, denominator state, PV accumulation and online numerator state remain
FP32. As in the existing Volta route, Q and gathered K/V are also FP16 MMA
operands. These boundaries require independent numerical qualification; an
FP32 accumulator alone does not imply FP32 attention accuracy.

Every key block, including the tail, receives the exact device-position
causal mask before maxima or probabilities are computed. Fully masked tiles
produce zero mass; the MMA pipeline's masked look-ahead cannot read beyond
the valid tile metadata. The implementation does not use the separate 79T
implementation's sampled maxima or exponent clipping.

There are no extra device allocations, retained pointers, constant-symbol
metadata, private streams, or duplicate gathered caches. The existing mask
workspace stores all groups' packed FP16 Q, FP16 probabilities, FP32 tile
statistics and FP32 online maxima/denominators. The existing output staging
stores the four groups' FP32 numerators. Only a bounded, per-CTA shared-memory
table of tile scales is added; it fits V100's 96 KiB opt-in limit together
with the MMA/epilogue storage. No FP32 score or temporary PV-output matrix is
materialized. Call-specific pointers and extents are passed by value instead
of the upstream implementation's mutable device globals.
The original INT8-G64 codec, normalized-Hadamard Q/K transform and FP16 KV
gather remain unchanged. Key-block width is bounded by available scratch and
8192 keys. The host execution envelope must be exact; graph capture and
unsupported shapes retain the existing implementation.

The CMake option defaults to ON. It admits Hq24/Hkv4/D256, query
blocks of 64–1024 tokens and key extents 32768–262144. Unsupported shapes,
graph capture and loose execution envelopes retain the FP32 flash-attention
fallback. Dispatch comparisons must use that numerically corrected fallback:
packed prefill is faster at T64/T128 as well as full T1024 blocks. The earlier
FP16-numerator fallback was faster on small widths but failed independent
numerical checks, so its timing cannot justify excluding those widths.
Setting the option OFF is a diagnostic fallback control, not a restoration of
the old baseline: the FP32 fallback repair remains active independently.

Broad numerical qualification covered T64/T128, partial M128 tiles at T65,
and T1024. The focused `--volta-prefill-tail-only` suite adds four sampled
FP64 cases on production INT8-G64: T127/T256/T512 at 32K and T877 at 128K,
with changing maxima and biased values. On the final 64–1024-gate binary, the
13-case dispatch/fallback subset passed in 241.94 seconds and the four tail
cases passed in 30.90 seconds, both exit0. Large cases sample first/middle/last
queries against all heads, components and visible keys. The independent FP64
reference remains unrounded; the causal criterion now includes a general BF16
half-ULP output floor. Base arithmetic constants remain, but the acceptance
envelope changed. These checks do not claim a measured gain at every
intermediate width or establish model reasoning quality from kernel timing.
The separate selected-model six-case quality suite and three matched fixed-output
performance pairs also passed. The release profile is qualified within the
documented scope; publication and deployment remain pending. See the
[October evidence report](../../deploy/v100/results/2026-10-03/README.md).
