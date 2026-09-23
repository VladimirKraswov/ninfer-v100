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
