# V100 deployment qualification, 2026-09-23

This campaign qualifies the owner's migration from llama.cpp to the V100 NInfer
fork. It is not a claim that quantized 27B matches a frontier model, or that a small
smoke suite establishes general quality parity.

## Hardware and artifact

- Proxmox VM 5100, Ubuntu 24.04, 16 vCPU Intel Xeon E5-2698 v4, 64 GiB assigned RAM.
- Tesla V100-SXM2-32GB, 32768 MiB, SM70, 300 W limit, graphics/memory 1530/877 MHz,
  PCIe 3 x16; NVIDIA 580.178.04, CUDA 12.9.86; CMake Release, arch 70.
- Qwen3.8-27B NVFP4 `.ninfer` v2; 23,719,496,192 bytes. Model provenance/hash and
  service command are in [deployment guide](../../README.md).
- Original baseline: Qwen3.8-27B-Abliterated Q4_K_M GGUF + BF16/F16 visual projector,
  llama.cpp, MTP4, KV q8_0, context 262144. Different weight/quantization artifacts:
  quality differences cannot be attributed solely to the inference engine.

## Proven arithmetic issue

The initial complete CTest run had 96 passes, 7 artifact-dependent skips and one
failure. `ninfer_gdn_gating_proj_test` reproduced an out-of-tolerance Qwen3.8 27B
control value after a CUDA Graph replay: actual -4.05658 vs FP64 -4.04687, T=16,
mode=4, phase=1. The composed route rounded normalized input to BF16 before the
control projection, although this is not an observable required cast boundary.

Commit `7cac899a` selects the existing SIMT kernel on SM70, keeping normalization
and control accumulation in FP32 and independently writing BF16 `h`. The original
test and extra tile-boundary graph replays pass with unchanged tolerances. Input
widths include every T=1..128, replay boundaries and prefill extents through 4097;
split and contiguous-parent weight forms are covered. The speculative-round test
also passes. Six affected serving/CLI/schema/logging tests pass. The seven skipped
real-artifact tests require other specific model/draft fixtures; they do not count
as passing. We did not rerun all unchanged tests after the targeted fix.

## Historical comparison (unmodified NInfer, 2048 output-token ceiling)

These are the previous campaign's decode medians of three runs, not new fork
numbers. Names describe nominal case sizes, not exact tokenizer counts.

| Case | llama.cpp tok/s | Original NInfer tok/s | Ratio |
|---|---:|---:|---:|
| ~2K | 58.5 | 83.9 | 1.43x |
| ~8K | 52.9 | 80.3 | 1.52x |
| ~32K | 44.4 | 63.1 | 1.42x |
| ~96K | 31.1 | 49.5 | 1.59x |
| ~160K | 22.1 | 43.1 | 1.95x |
| ~180K | 21.3 | 41.5 | 1.95x |

Previous cold 225810-token text probe: llama.cpp TTFT 944 s, end-to-end 1105 s,
2832 completion tokens; original NInfer TTFT 587 s, end-to-end 683 s, 3206 completion
tokens. Prompt-tokens/TTFT approximates 239 vs 385 tok/s, including frontend and
first-token overhead; this is not isolated CUDA prefill throughput. The old report's
quality totals and pass@1 labels are not a valid gate: see the methodology errata
in the deployment guide. Original raw data is retained in the owner's benchmark
workspace; it has not been overwritten.

## New campaign protocol

Serial requests only, one resident model. Compare MTP1/chunk1024, MTP2/chunk2048
and MTP4/chunk1024, 3 runs each at two context sizes, output ceiling 512, seed1001,
xhigh effort. TTFT is time to the first content **or reasoning** token. Decode rate
is usage completion_tokens / (end-to-end - TTFT), including thinking and final text.
This shorter ceiling is for configuration selection, not a controlled comparison
against the 2048-token historical baseline. Prefix-cache hits are recorded; distinguish
cold prompt ingestion from warm repetitions.

Quality uses the frozen 108-case smoke suite, root seed20260923 hashed with case ID,
no retries, Medium, production 32768 output/8192 thinking cap. Old per-case output
limits are overridden and that difference must remain explicit. All responses and
finish reasons are retained locally. A separate cold ~225K prompt plus image checks
three scattered facts, OCR and output completion together. Fine-detail Vision and
exact word counting remain model limitations until independently demonstrated.

## Measured configuration choice

All entries below use the **corrected FP32 control route**, 262144 KV capacity and
Vision enabled. Actual prompts are **1267** and **29961** tokens (the historical file
names say 2K/32K). Three serial runs per cell, median decode tok/s:

| Profile | Output ceiling / seed protocol | 1267 input | 29961 input |
|---|---|---:|---:|
| MTP1 / prefill1024 | 512 / fixed1001 | 52.35 | 40.91 |
| MTP2 / prefill2048 | 512 / fixed1001 | 62.62 | 53.31 |
| MTP4 / prefill1024 | 512 / fixed1001 | 64.70 | 46.52 |
| MTP2 / prefill2048 | 2048 / historical per-run seeds | 74.76 | 59.53 |
| **MTP4 / prefill2048** | **2048 / same historical per-run seeds** | **81.54** | **66.04** |

The final two rows replay the six exact seeds and sampling settings recovered from
the prior server request log; both receive identical prompts, effort and output
limits. Generated lengths differ between profiles, so this is a practical serving
comparison, not a kernel microbenchmark. On these representative runs MTP4 improves
median throughput over MTP2 by 9.1% and 10.9%; **MTP4/prefill2048 is selected**.
The 512-token seed1001 ranking differs: a single fixed sampled prefix does not
establish a universally optimal speculative depth.

Cold ~30K TTFT was 31.38–31.40 s at prefill1024 and 29.67–29.75 s at prefill2048,
about 5.5% less. Short-prompt cold TTFT was 1.23 s; cached repeats 0.09 s. The chosen
profile's ~30K cached TTFT was 0.13 s. These include scheduling/frontend time.
Memory at startup leaves approximately 367 MiB free after allocating the full KV;
large image batches are not qualified by the single-image tests.

Compared with the previous engine's historical medians, the selected fork is about
2.8% slower on the short case (81.54 vs83.9) and 4.7% faster at ~30K (66.04 vs63.1).
It retains a roughly 1.39x/1.49x advantage over the historical llama.cpp measurements
(58.5/44.4). Do not claim a universal speedup from the arithmetic fix.

Machine-readable timings/counters: [performance.json](performance.json).

## Production-profile quality

**107/108 on the first attempt**, with complete non-empty outputs and no output-limit
finishes: code51/51, math20/20, instruction15/16, tool calls/follow-up11/11, Vision10/10.
The only failing response is `instr-echo-reverse`: input `топан от` should reverse to
`то напот`, but the model wrote `тон апо`. This is a real sampled error, not a scorer
adjustment. The V100 agent guidance recommends executable checks for exact counts,
string operations and arithmetic, in addition to normal code tests.

The former code-16 case passes in the full suite and all three targeted seed/effort
checks with MTP0 and MTP2. This does not establish that the numerical fix alone cured
that sample: precision, effort, sampling and speculation can change the generated path.
The old clock oracle was wrong; the corrected clock case passes in the production
suite, while two targeted 1024-thinking probes differed between MTP profiles. Fine
visual details still need independent verification.

[quality.json](quality.json) contains per-case seeds, finish reasons, usage, timing
and verdicts. Full raw responses remain in the local campaign receipts. Offline
rescoring with the final stricter checker changed no production-suite verdicts.
The helper's HTTP-only test confirms that Markdown/prose around required bare JSON,
output exhaustion, and a failing program that prints `TESTOK` cannot produce a pass.
