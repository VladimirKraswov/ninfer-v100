# Single-resident Qwen / Pi service qualification

Hardware: VM5100, 16 CPU cores, 64 GiB RAM, Tesla V100-SXM2-32GB (sm_70),
CUDA 12.9, driver 580.178.04, persistence enabled, clocks 877/1530 MHz.
The sole physical V100 remains assigned to this VM. Gemma VM5151 is preserved,
stopped with its GPU assignment removed. RTX 5090 / FreeToken is unaffected.

## Artifacts and runtime

| Public ID | Weights | Allowed agents | Artifact SHA256 |
|---|---|---|---|
| qwen-v100 | Existing Huihui abliteration, NVFP4 | OpenCode, Pi | `02c0c80616e2dd353133355d840aa6418d83f4c523369ad93b426e6c5bbc83c8` |
| qwen-v100-pi | bytkim/Qwen3.8-27B-pi, own BF16 converted with existing mixed groupwise recipe | Pi | `09f24d5af4325d300f1488273324a278b7d312db4532074db8eba0a33f99a5e3` |

Pi source revision `34d3f21045f7b41453e66c39e137cfc818a603ce`: all 37
publisher files verified before conversion, including its own MTP weights.
Native artifact size 20,444,499,968 bytes. Its embedded author DFlash2 companion
is retained for artifact framing; serving uses MTP, not DFlash2.

Both entries configure 262144 context/KV capacity, int8 KV, MTP4,
`--lm-head-draft`, vision, preserved thinking, maximum output32768,
default Medium/thinking8192, prefill chunks2048, concurrency1, FIFO8,
queue deadline1800000ms. V100 INT8 split keys480 and QPN tail1 are retained.
Generation remains native C++/CUDA; the Python supervisor performs HTTP routing
and process lifecycle only. Production binary SHA256:
`d1f134d02456cb32f90b22a15c36f9e5d2321c8ee2c23bc0f3845b69c4108b5f`.

The pinned Pi export omits legacy tokenizer-config fields. Compatibility accepts
explicit null-BOS, tokenizer.json added tokens and an independently hash-validated
standalone chat template. Conflicting definitions, prefix changes and unsupported
templates remain rejected. No CUDA arithmetic or speculation algorithm changed.

## Real V100 results

Qwen → Pi → Qwen: **9/9** Russian JSON, logic and tool-call smoke checks passed.
Typical warm-file load was about17s for Qwen and20s for Pi. One resident process
at a time; Pi allocated about29 GiB VRAM, Qwen about31.4 GiB.

Pi replayed six frozen heldout tasks from the prior quality suite, without retries:

| Task | Actual input tokens | Output tokens | Decode tok/s | Prefill seconds | Outcome |
|---|---:|---:|---:|---:|---|
| Russian ledger | 1028 | 238 | 54.56 | 1.05 | Pass |
| Code intervals | 1043 | 927 | 44.91 | 1.18 | Pass |
| Logic schedule | 1013 | 411 | 49.67 | 1.07 | Pass |
| Retrieval16K | 16382 | 315 | 62.97 | 19.44 | Pass |
| Retrieval131K | 131092 | 376 | 45.73 | 198.40 | Pass |
| Retrieval229K | 229233 | 303 | 34.76 | 419.76 | Pass |

An additional image-arrow/color case passed. The 262144 allocation fits, but the
largest tested *input* was229233; do not report a262K-input quality result.

A fixed16K-input performance request produced16394 input /1024 output tokens,
**46.84 native decode tok/s**,835.28 prefill tok/s,19.63s prefill and41.49s
wall time. MTP accepted690/1327 draft tokens (52%). The shorter retrieval response
at62.97tok/s is a different workload, not the fixed-output benchmark.

These artifacts differ in finetuning and quantization. This campaign does not
establish BF16 equivalence, a Pi speedup over NVFP4, or a multi-user throughput
claim. Quality success on these fixtures does not guarantee arbitrary task success.

## Deployment and limits

`ninfer-model-gateway.service` is enabled at192.168.31.93:8080 with the native
child restricted to127.0.0.1:18080. Ordinary Qwen is the startup default. The old
`ninfer-v100.service` is disabled and retained for rollback. Control requires a
private bearer credential; catalog agent compatibility is explicit. Generation
must request the resident model; unknown/wrong-agent/wrong-resident requests fail
instead of silently using other weights. Active streams drain before unload.

Mac providers use existing strict SSH forwarding through Proxmox. OpenCode binds
`local-qwen38/qwen-v100`; Pi provider `local-qwen-v100` exposes both IDs and sends
`X-Ninfer-Agent: pi`. Desktop control uses a separate loopback binding. Model-control
credentials reside in OS credential storage, never in committed registries/preferences.

Seven CPU-process gateway checks, affected conversion checks and native SM70
frontend tests passed. Historical unrelated NVFP4 converter tests have obsolete
fixtures; they were not rewritten or counted as passing. Raw qualification outputs
and private deployment credentials remain outside Git. See
[the lifecycle/API contract](../../MODEL-SWITCHING.md) for installation and failure behavior.
