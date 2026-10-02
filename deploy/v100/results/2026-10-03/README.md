# V100 / Qwen3.8-27B optimization evidence, 2026-10-03

**Status: published, installed and production-smoke verified. The prior GPU service is restored; Qwen is currently stopped and not GPU-resident.**
The final MTP4 profile passed 13 focused and four tail GPU cases, all six
reasoning-quality cases on the first attempt, and three valid fixed-output
performance requests. At 229245 input tokens, full-request latency fell 29.0%.
At 16394 input tokens it increased 6.4%; this is a measured long-context
improvement with a short-context tradeoff, not a universal speedup.

Machine-readable measurements, seeds, fixture hashes, verdicts and limitations:
[metrics.json](metrics.json). This report contains technical evidence only; raw
responses and private operational records are not included.

## Hardware, artifact and scope

One NVIDIA Tesla V100-SXM2-32GB, SM70, CUDA 12.9.86, Release build; 16 vCPUs and
64 GiB host RAM. The selected model is **Huihui abliterated Qwen3.8-27B NVFP4**, in
`.ninfer` v2 format, with SHA256
`02c0c80616e2dd353133355d840aa6418d83f4c523369ad93b426e6c5bbc83c8`.
All model comparisons retain that artifact, INT8 group64 KV, Vision enabled,
concurrency one, prefill chunk 2048 and disabled prefix reuse. Artifact identity
is attested by a file hash; the HTTP protocol does not verify it.

NVIDIA driver: 580.178.04, power limit 300 W. Application clocks were requested at
1530/877 MHz, but an under-load snapshot showed **1477/877 MHz**, 68 °C, 292.7 W
and 31959 MiB used. These are observed values, not a claim of locked 1530 MHz
throughout the campaign.

The configured context and KV capacity are **262144 total tokens**, including
input and output. Quality requests target about 229K input to leave room for a
32768-token output budget. They do not measure 262144 input tokens plus output.
The 2048-token service prefill chunk is internally divided into two query blocks
of at most 1024 tokens.

The control is a separate build of commit
`ce67fc9cd7fb58a82b57e9e7d27f4943ede7ce8a`. Setting packed prefill `OFF` in the new
tree does **not** recreate it: the FP32 fallback repair is unconditional on SM70.

## What was taken from the three projects

| Source | Pinned revision | Applied work |
|---|---|---|
| [ninfer-v100](https://github.com/geoffwatts/ninfer-v100) | `b37d0dd3e1163b9d802d8bccfa89918bf68d793e` | Existing C++/CUDA SM70 engine, registered artifacts, INT8-G64 KV, MTP, serving and Vision; repaired FP32 fallback and evaluated decode layouts. |
| [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | `a3498d463db549c17a58c04f4455694e0416ca54` | v37 packed GQA with six query heads per KV group, adapted to batched SM70 QK/PV GEMMs and FP32 online softmax. |
| [HyperQwen](https://github.com/syv-ai/HyperQwen) | `e1459c7631774f56de2f9425437d54e7e72ea688` | Reviewed KV compression, MTP, head quantization and lookup ideas against capabilities already in the engine; no KVarN or Ampere-only kernel is claimed as a V100 port. |

V100 provides FP16 Tensor Cores, without native BF16 or INT8 Tensor Cores. Speedups
reported on an RTX 3090 therefore do not predict V100 performance. All three
repositories have Apache-2.0 top-level licenses; the adapted v37 code carries
BSD-3-Clause attribution in [the integration notes](../../../../third_party/onecat_v37/UPSTREAM.md).

## Model measurement protocol

The corpus and fixture hashes match across profiles. Corpus SHA256:
`5433c64364b1803c3550225bf59f75b3b932a01d40187d7c5337d50ccd43fa8a`.
All profiles used the same frozen campaign harness, SHA256
`2919b416c8640437589b8d2bf1222a816aed9cabab30aeb6d3a802cc1a174cf9`.
Its original metadata predates the `harness_sha256` field; this separately recorded
hash was not backfilled into those records. The current local comparator requires
complete metadata, so these legacy pairs were matched by trial IDs, fixture and
metadata checks plus the preserved harness identity.

Performance uses one Russian continuation prompt at each length, thinking
disabled, zero cached input, seed-repeat 20261002 and derived per-case seeds.
Each valid request reaches **1024 actual output tokens** with `finish_reason=length`.
Early EOS is invalid for fixed-output comparison and is not retried until it passes.
Rates below are native engine timings. Wall time and first-visible-token TTFT are
measured by the HTTP client and include frontend overhead.

The Anthropic count endpoint sizes fixtures; actual OpenAI usage confirms the
input length. String length is never used as a token count. The perf fixtures
are labeled `heldout` in the corpus but were used for MTP selection, so their final
performance is not independent of tuning. The six quality cases are separate tasks.

## Final matched performance: MTP4, gate64–1024

All three pairs have identical actual input, 1024 actual output tokens,
`finish_reason=length`, zero reasoning tokens and zero cached input. The final
app SHA256 is `36b0f3638a11791c5fdded71e11b20facf2ba575a2dc99624ec8be8ab70d8e27`.

| Actual input | Baseline → final prefill, s | Prefill latency change | Baseline → final decode, tok/s | Decode rate change | Baseline → final wall, s | Wall latency change |
|---:|---:|---:|---:|---:|---:|---:|
| 1006 | 0.900 → 0.950 | +5.5% | 58.003 → 60.407 | +4.1% | 18.541 → 17.890 | −3.5% |
| 16394 | 14.589 → 18.812 | +28.9% | 56.215 → 63.702 | +13.3% | 32.799 → 34.886 | +6.4% |
| 229245 | 593.763 → 417.192 | −29.7% | 32.768 → 38.947 | +18.9% | 625.142 → 443.619 | −29.0% |

At about 229K, native prefill throughput is 386.089 → 549.495 tok/s (**+42.3%**);
the corresponding latency reduction is 29.7%, a different percentage. The full
request is about 181.5 seconds shorter. First-visible TTFT changes from
593.921 to 417.351 seconds. The 16K prefill regression is retained explicitly:
faster decode does not offset the added prefill time there.

This is one prompt/seed per length, with no repeated interleaved A/B confidence
interval. Long-prompt draft acceptance changed from 710/1249 (56.8%) to 767/1079
(71.1%); decode improvement combines runtime and generated-path effects. The
separate six-case quality gate passed on both builds. These observations support
this release profile for the stated long-context workload within the limits below.

## Intermediate MTP tuning: historical T1024 gate

These candidates use the repaired FP32 fallback, the **temporary T1024-only
packed-prefill gate** and QPN tail enabled. They are not measurements of the final
64–1024 gate. The MTP4 app hash is
`41f3c5b86afae38e9e57bb10ef3393ef716d4691d86b0474d9ed75f512004da4`;
MTP3/7 use the later normal build
`d06815d42f15093d41aba9959adf10460f2a28496bf26a403c3d3ebe749f7518`.
Their intended production math is the same, but they are different binaries.
The original candidate metadata's `qualified` label is an inherited driver label,
not proof of whole-model qualification.

| Actual input | Profile | Prefill, s | Native decode, tok/s | First-visible TTFT, s | Request wall, s | Draft acceptance |
|---:|---|---:|---:|---:|---:|---:|
| 1006 | Baseline MTP4 | 0.900 | 58.003 | 0.903 | 18.541 | 36.9% |
| 1006 | Candidate MTP3 | 0.935 | 68.883 | 0.940 | 15.791 | 55.6% |
| 1006 | Candidate MTP4 | 0.931 | 60.415 | 0.935 | 17.868 | 39.3% |
| 1006 | Candidate MTP7 | 0.953 | 52.848 | 0.957 | 20.314 | 22.5% |
| 16394 | Baseline MTP4 | 14.589 | 56.215 | 14.601 | 32.799 | 40.3% |
| 16394 | Candidate MTP3 | 18.600 | 52.910 | 18.615 | 37.950 | 41.3% |
| 16394 | Candidate MTP4 | 18.572 | 63.736 | 18.586 | 34.637 | 49.0% |
| 16394 | Candidate MTP7 | 19.414 | 52.911 | 19.429 | 38.764 | 27.1% |
| 229245 | Baseline MTP4 | 593.763 | 32.768 | 593.921 | 625.142 | 56.8% |
| 229245 | Candidate MTP3 | 426.578 | 40.067 | 426.742 | 452.274 | 85.6% |
| 229245 | Candidate MTP4 | 426.472 | 39.415 | 426.636 | 452.591 | 72.1% |
| 229245 | Candidate MTP7 | 428.959 | 21.574 | 429.122 | 476.540 | 28.3% |

The baseline 131059-input request stopped after 702 output tokens. Its raw timing
is preserved in `metrics.json`, but excluded from these fixed-output comparisons.

**MTP4 is selected for final evaluation.** MTP3 is fastest on the short prompt and
has 1.65% higher long-context decode than MTP4, while MTP4 is 20.46% faster in
decode at 16K. MTP7 drops to 21.57 tok/s at about 229K despite its faster attention
Op: low draft acceptance and extra draft work matter. This is a balanced choice
on three samples, not a universal optimum or a statistically established ranking.

Against baseline MTP4, the intermediate MTP4 long request reduces prefill latency
by 28.2%, improves decode rate by 20.3% and reduces whole-request wall time by
27.6%. At 16K its prefill regresses and the whole request is **5.6% slower**.
The broader gate was measured separately in the final table above; these tuning
measurements remain distinct from those release-profile results.

## Attention Op measurements and gate correction

These are public attention Op microbenchmarks: Hq24/Hkv4/D256, INT8-G64,
fragmented mapping, batch one, warm cache, five warmups and 20 repeats. Prefill
uses eager execution and decode uses graphs. They establish kernel/Op effects,
not full-model speedups.

| History tokens, T1024 | Baseline median, ms | Fused median, ms | Op speedup |
|---:|---:|---:|---:|
| 32768 | 26.272 | 19.645 | 1.337× |
| 131072 | 130.223 | 68.812 | 1.892× |
| 261120 | 275.557 | 130.890 | 2.105× |

The initial T64/T128 comparison favored the old fallback and prompted the
T1024-only gate. That fallback subsequently proved numerically defective.
Against the repaired FP32 fallback, the broad fused path is much faster:

| History tokens | T | Repaired fallback median, ms | Broad fused median, ms | Fallback/fused latency |
|---:|---:|---:|---:|---:|
| 32768 | 64 | 10.875 | 3.510 | 3.10× |
| 32768 | 128 | 17.988 | 4.857 | 3.70× |
| 131072 | 64 | 43.726 | 11.337 | 3.86× |
| 131072 | 128 | 73.921 | 14.415 | 5.13× |
| 261120 | 64 | 87.862 | 22.428 | 3.92× |
| 261120 | 128 | 152.253 | 27.773 | 5.48× |

These were separate timing series, but the difference greatly exceeds their
within-series min–p95 spread. Source now restores the 64–1024 gate, with an exact
execution envelope and 32768–262144 keys. Graph capture and unsupported shapes
retain fallback. The final binary passed the additional tail coverage. Its T64
and T128 medians at 32K are 3.488 and 4.824 ms; T1024 is 19.538 ms. Full final
prefill/tail microbenchmark rows are retained in `metrics.json`.

QPN tail improves the T7/T8 attention Op by **26–29%** across 32K, 128K and near
256K. T1/T5 vary by about 0.3%. It remains a startup-fixed opt-in control,
`NINFER_V100_INT8_QPN_TAIL=1`, with default zero. Split-KV tuning did not justify
changing the existing split size 480: small long-context gains cost larger
regressions elsewhere. Neither result by itself chooses the model's draft window.

## Numerical contract and evidence

Causal A1 append, A3 cached and batch are checked against an independent FP64
oracle evaluated from represented public cache/query inputs. The reference
remains **unrounded**, and raw output errors and relative L2 are measured.
The repaired SM70 fallback accumulates PV in FP32 and normalizes by emitted
FP16 probability mass.

The acceptance criterion **changed** to include a general BF16 output-rounding
floor. The original arithmetic profile constants remain:

| Storage profile | Relative L2 | Gross absolute | Gross relative to maximum reference |
|---|---:|---:|---:|
| BF16 | 2.8e−3 | 1.0e−3 | 2.7e−3 |
| INT8-G64 | 3.15e−3 | 1.1e−3 | 3.0e−3 |

For `h[i] = halfULP_BF16(reference[i])`, the decision bounds are:

```text
gross_limit[i] = profile.absolute + max(profile.relative * max(abs(reference)), h[i])
raw_relative_L2_limit = max(profile.L2, norm(h) / max(norm(reference), 1e-30))
```

This is an expanded acceptance envelope, not unchanged tolerances. It applies
uniformly across registered storage profiles and causal routes, without a
shape-specific exception. Other plain/packed/context attention suites retain
their own criteria. The CPU criterion checks accept correct BF16 rounding at
signed binade boundaries and reject a sparse, representable 4.69% outlier. CPU
criterion checks are distinct from GPU kernel qualification.

| Evidence | Result and scope |
|---|---|
| QPN tail independent FP64 GPU suite | PASS, 532.2 s, under the original criterion. |
| Broad fused prefill GPU suite | 477.9 s; all fused cases passed, but the suite failed one BF16 graph-fallback case, seed974, under the original criterion. |
| Repaired fallback and revised criterion, T1024 gate | Focused 13-case GPU suite PASS, exit0, including the seed974 regression. |
| Final 64–1024 gate: focused13 | **PASS**, exit0, 241.94 s, under the revised BF16 output criterion. |
| Final 64–1024 gate: four tail cases | **PASS**, exit0, 30.90 s: INT8-G64 T127/256/512 at 32768 and T877 at 131072, with changing maxima and biased values. |

Large-prefill checks sample first/middle/last queries, covering all heads,
all 256 output components and all visible keys for each selected query. These
tests qualify those causal Op cases, not every input. Passing tolerance does not
establish bitwise parity, equality of model output distributions or general
reasoning quality. The separate model gate below remains necessary.

## Reasoning quality and final evaluation

Baseline passed **6/6 on the first attempt**, with `xhigh`, thinking budget 16384
and total output budget 32768. All outputs ended with `stop`, zero cached input
and no truncation. Code assertions are withheld from the prompt.

| Baseline quality case | Actual input | Output / reasoning tokens | Prefill, s | Native decode, tok/s | Verdict |
|---|---:|---:|---:|---:|---|
| Russian strict-JSON ledger | 1028 | 268 / 254 | 1.020 | 74.832 | PASS |
| Python interval merging | 1043 | 2045 / 1905 | 1.024 | 62.417 | PASS |
| Logical schedule | 1013 | 917 / 892 | 0.908 | 69.391 | PASS |
| Three-needle retrieval, ~16K | 16382 | 456 / 392 | 14.636 | 83.748 | PASS |
| Three-needle retrieval, ~128K | 131092 | 364 / 286 | 232.953 | 52.107 | PASS |
| Three-needle retrieval, ~229K | 229233 | 383 / 319 | 592.669 | 38.029 | PASS |

The final MTP4 build with the restored 64–1024 gate also passed **6/6 on the first
attempt**, using the same fixture hashes, seeds, `xhigh` effort and 16384/32768
thinking/output budgets. Every response ended with `stop`, without retries,
truncation or cached input:

| Final quality case | Actual input | Output / reasoning tokens | Prefill, s | Native decode, tok/s | Verdict |
|---|---:|---:|---:|---:|---|
| Russian strict-JSON ledger | 1028 | 232 / 218 | 1.048 | 70.675 | PASS |
| Python interval merging | 1043 | 1600 / 1477 | 1.055 | 63.738 | PASS |
| Logical schedule | 1013 | 598 / 546 | 0.934 | 65.500 | PASS |
| Three-needle retrieval, ~16K | 16382 | 397 / 333 | 18.607 | 78.384 | PASS |
| Three-needle retrieval, ~128K | 131092 | 279 / 215 | 197.725 | 51.589 | PASS |
| Three-needle retrieval, ~229K | 229233 | 383 / 319 | 417.289 | 39.273 | PASS |

Output lengths differ on several tasks, so these quality timings are observations,
not the fixed-output performance comparison. Passing this small suite establishes
the stated regression gate, not general reasoning parity.

| Final MTP4 with 64–1024 gate | Status |
|---|---|
| Final binary identity | `36b0f3638a11791c5fdded71e11b20facf2ba575a2dc99624ec8be8ab70d8e27` |
| Focused13 and four tail GPU checks | PASS, exit0 for both |
| Same six quality cases and budgets | PASS 6/6, first attempt |
| Matched 1K/16K/~229K fixed-output performance | PASS 3/3 valid fixed-output pairs |
| Release-profile qualification | PASS within the stated numerical, quality and performance scope |
| Publication | Commit `4cb0ba78b6a64b3ee0a6f8200ec8aab5d474608a` on GitHub `master` |
| Installation and production verification | Exact qualified binary installed; 3/3 smoke cases PASS, first attempt |
| Current GPU residency | Prior service restored and GPU-resident; Qwen stopped and not resident |

Build and test commands, including `--volta-prefill-tail-only`, are in the
[deployment guide](../../README.md). Packed prefill is now **enabled by default**
with `NINFER_VOLTA_PACKED_PREFILL=ON` for qualified SM70 shapes. The measured
binary already explicitly used `ON`; changing the CMake default introduces no
GPU arithmetic change. `OFF` is a diagnostic fallback control and does not
recreate the historical baseline. QPN tail remains optional with runtime default
zero; this selected profile uses `NINFER_V100_INT8_QPN_TAIL=1`.

## Publication and production verification

The release source is published as
[4cb0ba78b6a64b3ee0a6f8200ec8aab5d474608a](https://github.com/VladimirKraswov/ninfer-v100/commit/4cb0ba78b6a64b3ee0a6f8200ec8aab5d474608a)
on `master`. The installed source manifest identifies that commit, and the
installed server hash is the exact qualified
`36b0f3638a11791c5fdded71e11b20facf2ba575a2dc99624ec8be8ab70d8e27`.
The service was active/running; the model endpoint advertised `qwen-v100` with
`max_model_len=262144`.

Production uses MTP4, chunk2048, INT8-G64 KV, QPN tail enabled and split480, with
Vision and preserved thinking. Prefix reuse is **enabled** in production; the
controlled performance campaign disabled it. Production defaults are
`medium`/8192 thinking and 32768 output, while the quality gate explicitly
requested `xhigh`/16384. These differences are preserved, not presented as a
controlled comparison of cached and uncached requests.

| Production smoke | Attempts | Finish reason | Input / output / reasoning tokens | Verdict |
|---|---:|---|---:|---|
| First calculator tool selection | 1 | `tool_calls` | 337 / 66 / 33 | PASS |
| Second calculator tool selection | 1 | `tool_calls` | 320 / 59 / 23 | PASS |
| Simple image-color case | 1 | `stop` | 269 / 77 / 72 | PASS |

All three used medium/8192 thinking, a 32768 output ceiling and zero cached input.
The two tool checks verify first-turn tool selection and arguments, not a complete
multi-turn tool-execution loop. One image case is a smoke check, not broad Vision
qualification. After verification, the GPU was restored to its prior service.
At 2026-10-02 22:35 UTC, that service was healthy, production-ready, accepted
jobs/realtime requests and had a model resident on V100 (30665 MiB used). The
maintenance state was released. **Qwen remains installed and verified, but is
currently stopped and not GPU-resident.** This restoration receipt establishes
service readiness and GPU use; it does not assert an unchanged container image
after independent service updates.

## Limits of the evidence


- One performance prompt/seed per length, without repeated interleaved A/B or a
  confidence interval. Generated paths and acceptance can change despite equal
  sampling seeds; decode gains are not a pure kernel attribution.
- About 30 seconds of CPU diagnostic compilation overlapped part of intermediate
  MTP4 prefill. No GPU job competed. MTP3/7 had no compilation overlap. Early
  baseline short quality timings also overlapped CPU compilation.
- The six quality cases are a narrow regression gate. A second seed-repeat,
  performance code templates, general reasoning parity, arbitrary long-document
  reasoning and image quality are not established by this campaign.
- Vision remains enabled; the six October quality cases are text-only and the
  separate production smoke adds one simple image case. About
  229K actual input validates that scenario, not every use of the full capacity
  or generation of all 32768 reserved output tokens.
- Model, quantization and artifact remain fixed. These results cannot be used to
  claim a comparison against the original non-abliterated checkpoint, a different
  quantization, or an unrelated engine's headline throughput.
