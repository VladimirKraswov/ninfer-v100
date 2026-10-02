# V100 deployment fork

Based on geoffwatts/ninfer-v100 at b37d0dd. The speculative acceptance algorithm is
unchanged. HTTP controls bound thinking; they do not eliminate model mistakes.

Changes:
- V100 27B normalization/control projection uses the existing SIMT FP32 fused kernel.
  The former composed route rounded the internal normalized values to BF16 before
  the control dots, violating the unchanged FP64 oracle tolerance on graph replay
  (T=16, norm-weight sign change). Only the public `h` output is rounded to BF16.
  The original regression and added tile-boundary replay cases retain their tolerances.
- OpenAI Chat/Responses accept a validated positive `thinking_budget` extension,
  overriding the process cap and using the Engine's existing canonical close path.
- `--default-reasoning-effort medium` prevents clients that omit the option from
  accidentally selecting the artifact's XHigh default. Explicit client effort wins.
- The CLI option test links product logging (same underlying build fix as upstream PR #7).
- Production service serializes inference with a bounded queue of eight requests and a
  30-minute queue deadline, accommodating a cold long-context request before queued agent work.
- Production service uses a 32768 output budget and an 8192 thinking cap, leaving room
  for the canonical close tokens and the answer/tools. Small output budgets can still
  finish with `length`; neither effort nor a cap guarantees a correct answer.

Measured hardware, parameter comparisons, qualification and limits: [2026-09-23 report](results/2026-09-23/README.md).

## Before comparing quality

The 2026-09-22 benchmark had material methodology mistakes:
1. Its clock drawing and prompt describe 12:15, but the scorer expected 3.
2. Seed 4294967295 means random in the llama.cpp baseline; it is a fixed numeric seed
   in NInfer. Repeating it does not produce independent trials. The runner also
   allowed retries (6 baseline cases and 3 NInfer cases used a second attempt), so
   its reported aggregate is not pass@1.

3. The uppercase-translation checker rejected a correctly capitalized sentence solely
   because it ended with a period; punctuation was not prohibited by the prompt.
4. The strict-JSON checker accepted surrounding prose/Markdown despite the prompt
   prohibiting it. The new runner requires the whole response to parse as JSON.

Preserve historical raw records. Correct the oracle, retain full outputs and finish
reasons, use recorded independent seeds for repeated trials, and distinguish a
reproducible sampled mistake from a proven arithmetic or speculative-decoding bug.
The code-16 off-by-one is real for the old fixed seed; do not special-case its prompt
or change the checker to make that output pass.

## Upstream review

Checked the V100 fork's open issues and pull requests on 2026-09-23.
[PR #7](https://github.com/geoffwatts/ninfer-v100/pull/7) addresses the test-link error.
[Issue #8](https://github.com/geoffwatts/ninfer-v100/issues/8) and
[issue #9](https://github.com/geoffwatts/ninfer-v100/issues/9) request artifact v3
support; this deployment pins a valid v2 artifact. No open patch there resolved the
GDN oracle failure reproduced in this campaign.

## Installation

`ninfer-v100.service` records this deployment's paths/address/user. Adjust those three
for another machine. Build with CUDA 12.9, sm_70, and benchmarks disabled; the engine
and tests are enabled. Model artifact is v2 from the pinned Hugging Face revision
11dbbbbbc33db198afe2f02c9232c771ff7031be of neroued/Qwen3.8-27B-nvfp4-NInfer,
SHA256 552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c.
Do not replace it with an incompatible v3 artifact or modify the version byte.

Only one inference process runs on V100. Test and verify the new service's health,
model ID/context, tool calling, reasoning budget and Vision before removing the old
runtime and GGUF. The service keeps the public alias `qwen-v100` for existing clients.

## Reproducing quality checks

Run `python3 deploy/v100/test_quality.py` for the HTTP-fixture checks of strict JSON,
output exhaustion, explicit budgets, independent seeds, no retries and resumability.
These checks do not use a GPU.

The frozen small suite is `tests/fixtures/v100/quality.jsonl` (108 code, math,
instruction, tool and image cases). Prompts include test assertions in some coding
cases: this is a regression/smoke suite, not an independent coding leaderboard.
`deploy/v100/quality.py` records full responses, finish reasons and deterministic
per-case seeds, performs no retries by default, and treats output exhaustion as a
failure. Run it only against an idle, explicitly selected engine. Example:

```sh
python3 deploy/v100/quality.py --url http://127.0.0.1:8080/v1/chat/completions \
  --model qwen-v100 --suite tests/fixtures/v100/quality.jsonl \
  --out /tmp/ninfer-quality.jsonl --effort medium --seed 20260923 --retry 0 \
  --max-tokens 32768 --thinking-budget 8192
```

Use a new output path for a different seed/profile; existing case IDs are skipped
for resumability. Historical reports must not be overwritten with rerun results.
The production profile overrides older per-fixture output limits. Record that change
when comparing results; it is not a controlled engine-only quality comparison.

## 2026-10-03: 256K packed prefill and controlled evaluation

This campaign uses the selected **Huihui abliterated Qwen3.8-27B NVFP4** artifact:
`/srv/ninfer-v100-lab/models/huihui-abliterated/qwen3_8_27b_huihui_abliterated_nvfp4.ninfer`,
SHA256 `02c0c80616e2dd353133355d840aa6418d83f4c523369ad93b426e6c5bbc83c8`,
served as `qwen-v100`. The September installation/report above records an earlier
artifact. Baseline and final runs retain the Huihui file, tokenizer, template,
reasoning settings and fixtures. The portable [October report](results/2026-10-03/README.md)
and [metrics.json](results/2026-10-03/metrics.json) contain the measured evidence,
intermediate MTP3/4/7 comparisons, final quality results and remaining limitations.

Three controls are available on SM70:

- `NINFER_VOLTA_PACKED_PREFILL=ON` is now the **CMake default** for qualified
  shapes. The [1Cat v37 adaptation](../../third_party/onecat_v37/UPSTREAM.md) packs
  six query heads per KV group, with FP32 QK/PV accumulation and online softmax
  state, using FP16 Tensor Core operands. It reuses the existing INT8-G64 codec
  and gather workspace. The gate covers Hq24/Hkv4/D256, internal query blocks
  of 64–1024 tokens and 32768–262144 keys with an exact execution envelope.
  Smaller blocks, unsupported shapes and graph capture retain the repaired FP32
  flash-attention fallback. `OFF` is a diagnostic control.
- `NINFER_V100_INT8_SPLIT_KEYS` selects long-context INT8 decode/verify split size
  once per process. Unset preserves **480**; restart between choices. The sweep
  did not justify a different global setting. See [bench/README.md](../../bench/README.md).
- `NINFER_V100_INT8_QPN_TAIL=1` opts into the long-context INT8 T7/T8 QPN tail.
  Default remains `0`; restart between choices. The selected MTP4 profile uses
  `1`. T1/T5 are outside this optimization.

The current tree also repairs FP32 PV accumulation and probability-mass
normalization in the existing SM70 fallback. These changes are independent of
`NINFER_VOLTA_PACKED_PREFILL`: `OFF` **does not restore the unmodified baseline**.
The measured control is a separate build of production commit
`ce67fc9cd7fb58a82b57e9e7d27f4943ede7ce8a`.

HyperQwen supplied useful cache/speculation and measurement ideas; its KVarN,
BF16 speculative kernels and Ampere INT8 Tensor Core paths are not enabled here.
The fused T1024 attention Op measured 1.337× / 1.892× / 2.105× speedups at
32K / 128K / near-256K. QPN T7/T8 measured 26–29%. Those are Op results, not model
speedups. The temporary T1024-only gate compared against the old numerically
incorrect fallback. Against the corrected FP32 fallback, fused T64/T128 is
3.1–5.5× faster; the restored gate is 64–1024.

The independent FP64 oracle remains unrounded; errors use that reference.
**The acceptance envelope changed:** each gross bound is
`profile.absolute + max(profile.relative * max(abs(r)), halfULP_BF16(r[i]))`,
and raw relative L2 is bounded by
`max(profile.L2, norm(halfULP_BF16(r)) / max(norm(r), 1e-30))`.
Original BF16 base constants remain `2.8e-3 / 1e-3 / 2.7e-3` (L2 / absolute /
relative), and INT8-G64 retains `3.15e-3 / 1.1e-3 / 3.0e-3`. The added output
representability floor expands the decision bound; tolerances are not unchanged.
It applies to causal A1 append, A3 cached and batch for all registered storage
profiles, independently of shape or private route. Other attention suites retain
their criteria. See [tests/README.md](../../tests/README.md) for the contract.

The final 64–1024-gate binary has SHA256
`36b0f3638a11791c5fdded71e11b20facf2ba575a2dc99624ec8be8ab70d8e27`.
Focused13 and four tail GPU cases passed, exit0, in 241.94 s and 30.90 s.
Both baseline and final MTP4 passed **6/6 quality cases on the first attempt**,
including 229233 actual input tokens, with `xhigh`, thinking budget16384 and
output budget32768. No truncated response was accepted; there were no retries.
This establishes the small regression gate, not general reasoning parity.

MTP4 was selected from the intermediate MTP3/4/7 measurements: MTP3's long-context
rate is only 1.65% above MTP4, while MTP4 is 20.46% faster at 16K; MTP7 is much
slower on the long fixture. Final matched fixed-1024 performance completed all
three valid pairs. At about 229K, prefill is 593.763 → 417.192 s (−29.7% latency),
decode 32.768 → 38.947 tok/s (+18.9%), and wall 625.142 → 443.619 s (−29.0%).
At 16K, wall time increases 32.799 → 34.886 s (+6.4%); at 1K it falls 3.5%.
This is one prompt/seed per length, not a universal speedup. The profile is
qualified within the report's scope; publication and deployment remain pending.
Baseline 128K early EOS at 702 output tokens remains excluded.
The final binary was compiled with explicit `ON`; changing the CMake default
introduces no GPU arithmetic change. Hardware was requested at 1530/877 MHz,
but an under-load snapshot recorded 1477/877 MHz, so clocks are not claimed
constant at 1530 MHz.

Build an isolated laboratory binary with the VM's CUDA 12.9 toolchain from the
repository root. Use the separately preserved production commit for a historical
baseline; this command builds the candidate and does not replace the service:

```sh
cmake -S . -B build-v100-candidate -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_VOLTA_PACKED_PREFILL=ON
cmake --build build-v100-candidate -j --target ninfer-serve \
  ninfer_softmax_attention_test ninfer_softmax_attention_criterion_test \
  ninfer_causal_softmax_attention_bench
./build-v100-candidate/tests/ninfer_softmax_attention_criterion_test
./build-v100-candidate/tests/ninfer_softmax_attention_test --volta-prefill-only
./build-v100-candidate/tests/ninfer_softmax_attention_test --volta-prefill-fallback-only
./build-v100-candidate/tests/ninfer_softmax_attention_test --volta-prefill-tail-only
NINFER_V100_INT8_QPN_TAIL=0 \
  ./build-v100-candidate/tests/ninfer_softmax_attention_test --volta-long-int8-only
NINFER_V100_INT8_QPN_TAIL=1 \
  ./build-v100-candidate/tests/ninfer_softmax_attention_test --volta-long-int8-only
```

The [attention qualification subsets](../../tests/README.md) use the independent
FP64 oracle and the causal BF16 output criterion described above. The historical
`--volta-prefill-fallback-only` name selects 13 cases; with the broader gate its
exact-envelope INT8 cases can use packed prefill while graph cases retain fallback.
`--volta-prefill-tail-only` adds four actual INT8-G64 cases: T127/T256/T512 at 32768
and T877 at 131072, sampling first/middle/last queries, all heads and all keys,
with biased values and strong changing maxima. Run GPU checks and model serving serially
under the deployment's exclusive GPU lease. For HTTP evaluation, select exactly one
server with the artifact above, `--model-id qwen-v100 --max-context 262144
--kv-capacity 262144 --max-concurrency 1 --kv-dtype int8 --preserve-thinking`.
Record its actual MTP, prefill, clocks and candidate settings with `--profile` below;
use identical settings when attributing a change to the engine alone. The launcher
divides a 2048-token service prefill chunk into two internal 1024-token query blocks
(`kVoltaFlashQBlockTokens`); both can use the candidate when the remaining shape
and execution gates are satisfied. Apply `NINFER_V100_INT8_QPN_TAIL=1` to the
server process as well when recording the QPN-enabled profile.

The Python harness needs no CUDA build. Its CPU-only contract checks are:

```sh
python3 tests/test_long_context_eval.py
python3 deploy/v100/test_quality.py
```

Prepare one immutable corpus against the selected server, then replay it on each
revision. These examples run on the VM against port 8080; the owner's existing Mac
tunnel uses port 18021. Use `--split tuning` and a separate corpus while selecting
settings; reserve the heldout corpus for the fixed acceptance comparison.

```sh
V100_EVAL_SHA=02c0c80616e2dd353133355d840aa6418d83f4c523369ad93b426e6c5bbc83c8
python3 deploy/v100/long_context_eval.py prepare \
  --url http://127.0.0.1:8080 --model qwen-v100 \
  --artifact-sha256 "$V100_EVAL_SHA" --split heldout \
  --corpus /tmp/v100-heldout.json
python3 deploy/v100/long_context_eval.py run \
  --url http://127.0.0.1:8080 --model qwen-v100 \
  --artifact-sha256 "$V100_EVAL_SHA" --corpus /tmp/v100-heldout.json \
  --out /tmp/v100-baseline --engine-revision BASELINE_REVISION \
  --profile 'record actual MTP, prefill, GPU clocks and candidate settings'
```

Repeat `run` with the same arguments to resume; for the candidate use its revision,
actual profile and `/tmp/v100-candidate`, retaining the same corpus. Then compare:

```sh
python3 deploy/v100/long_context_eval.py compare /tmp/v100-baseline /tmp/v100-candidate
```

The active October campaign retains the original remote harness for every
profile, SHA256 `2919b416c8640437589b8d2bf1222a816aed9cabab30aeb6d3a802cc1a174cf9`;
its preserved task-local copy is `outputs/v100-engine-20261002/ops/campaign-harness.py`.
Its metadata predates `harness_sha256`. Do not backfill that metadata with the
current local script hash. The current `compare` reports this legacy limitation;
compare those records by matching trial IDs and checked metadata with the preserved
remote-harness provenance. The command above applies directly to new campaigns
whose metadata contains all required identity fields.

Defaults are six quality fixtures and two performance templates, two independent
seed repeats, and at most eight requests per invocation. Quality uses `xhigh`, a
16384-token thinking cap and 32768 total output tokens; `--thinking-budget 8192`
is an explicit alternative that must match in both runs. Input targets are 1024,
16384, 131072 and 229248±100, preserving output room within 262144 total tokens.
The artifact's Anthropic count endpoint sizes the prompts; actual OpenAI usage must
confirm the requested input band. Quality checks cover Russian structured answers,
code with assertions withheld from the prompt, reasoning and three-depth retrieval.
This small suite is a regression gate, not proof of general reasoning parity.

Performance requests stream to record first-visible-token latency and retain native
prefill/decode timing and actual usage. A row is valid only at exactly 1024 output
tokens with `finish_reason=length`: HTTP cannot force EOS suppression, so early EOS
invalidates that performance row. Quality treats `length` as failure. There are no
automatic retries, including failed or interrupted attempts; use a new output
directory for an intentional rerun. Each invocation is bounded to 3600 seconds and
each request to 1800 seconds. Comparisons reject differing artifact/corpus/budgets
and omit prefill speedups when measured cache residency differs. These harness
checks do not replace a completed 256K run on the actual V100.

The corpus orders all Russian performance lengths before the code lengths. With
two seeds and the default eight-job limit, a performance invocation covers only
the Russian fixtures. The current bounded MTP sweep deliberately uses one seed
and only 1K/16K/about 229K Russian fixtures; code performance, a second repeat and
general reasoning parity remain unmeasured.

## OpenCode profile and service ownership

Keep the stable `local-qwen38/qwen-v100` model ID and SSH-tunneled endpoint. Model
limits are context 262144, input 229376, output 32768. Set the model default to
`reasoningEffort: medium, thinking_budget: 8192`; variant budgets are Low 2048,
Medium 8192 and XHigh 24576. The remaining output budget is available for answer/tool
calls and the canonical close sequence. `qwen-v100-build.md` is the matching agent
profile. Flash Next on RTX 5090 remains the user's global default.
OpenCode 1.18.18 currently sends a 32000-token response ceiling. With its existing
global compaction reserve of 32768, the input229376 setting starts automatic
compaction at 196608 tokens; the physical model context remains 262144. This avoids
silently changing the Flash Next compaction settings when adding V100.

This host also has an archived MIMIR llama.cpp unit. Disabling that unit alone is
insufficient because `mimir.target` explicitly Wants it. Install
`mimir-llm-guard.conf` as a drop-in for that legacy unit on this host: while the
NInfer service file exists, its condition prevents a second model from loading on
boot. Do not restart or remove the separate CPU ASR/TTS/VAD services.

## Operating and recovering this deployment

The owner's SSH alias is `ssh vm-v100`; the VM is Proxmox 5100 on node `pve`.
The service's source checkout is `/srv/ninfer-v100-lab/engine/ninfer`, with this
GitHub fork as `origin` and geoffwatts as `upstream`. Use the checked-out production
revision, rather than pulling an untested upstream update into the running service.
Inspect with `systemctl status ninfer-v100` and `journalctl -u ninfer-v100`.
The API listens on the existing VM LAN address/port in the unit. The Mac connects
through its existing local SSH tunnel on port18021; no new public port is exposed.

Deployment receipts and old unit/source provenance are retained in
`/srv/ninfer-v100-lab/migration-20260923/`. The former active Qwen GGUF/projector and
standalone llama.cpp installation were deleted at the owner's request after service
and OpenCode acceptance. Old benchmark data is retained. Restoring that old engine
now requires downloading/rebuilding its recorded runtime and weights; it is no
longer an instant service toggle. Do not start the old and new GPU engines together.
