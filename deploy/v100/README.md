# V100 deployment fork

Based on geoffwatts/ninfer-v100 at b37d0dd. Runtime changes preserve the target's
numerical kernels and speculative acceptance algorithm. We do not claim to have
eliminated model mistakes by modifying the HTTP frontend.

Changes:
- OpenAI Chat/Responses accept a validated positive `thinking_budget` extension,
  overriding the process cap and using the Engine's existing canonical close path.
- `--default-reasoning-effort medium` prevents clients that omit the option from
  accidentally selecting the artifact's XHigh default. Explicit client effort wins.
- The CLI option test links product logging (same underlying build fix as upstream PR #7).
- Production service uses a 32768 output budget and an 8192 thinking cap, leaving room
  for the canonical close tokens and the answer/tools. Small output budgets can still
  finish with `length`; neither effort nor a cap guarantees a correct answer.

## Before comparing quality

The 2026-09-22 benchmark had two material methodology mistakes:
1. Its clock drawing and prompt describe 12:15, but the scorer expected 3.
2. Seed 4294967295 means random in the llama.cpp baseline; it is a fixed numeric seed
   in NInfer. Repeating it does not produce independent trials. The runner also
   allowed retries, so its reported aggregate is not necessarily pass@1.

Preserve historical raw records. Correct the oracle, retain full outputs and finish
reasons, use recorded independent seeds for repeated trials, and distinguish a
reproducible sampled mistake from a proven arithmetic or speculative-decoding bug.
The code-16 off-by-one is real for the old fixed seed; do not special-case its prompt
or change the checker to make that output pass.

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
