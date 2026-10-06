# Single-resident model gateway

The Python supervisor owns HTTP routing and the native child lifecycle only. All
model execution remains the C++/CUDA NInfer engine. The native `StartupObserver`
emits structured events to an inherited nonblocking pipe, never to a parsed log.
The supervisor exposes weights-stage bytes and warmup/readiness without fabricating
an overall percentage. Its registry owns public IDs and allowed client agents;
there are no model-name tests in the inference runtime or Desktop.

## Registry and installation

Create a private registry with `backend_port:18080`, `load_timeout:900`,
`drain_timeout:1800`, `default_model:<explicit-id>` and `models`, keyed by public
model ID. Each entry supplies `label`, `agents` (explicit list), `context_window`,
`argv` (array, complete native command), optional `cwd`, and `env`. Native argv must
bind loopback on the shared backend port and include the same `--model-id` as the
registry key. Paths are explicit; no globs, shell commands or artifact guessing.
Protect registry/state directory and environment file with owner-only permissions.
Set a generated `NINFER_MODEL_CONTROL_TOKEN` via systemd EnvironmentFile. The token
is never passed to the native model process or written to an access log.

`ninfer-model-gateway.service` is a deployment template for VM5100, not a portable
installer: review paths, address, GPU ownership and clock settings. Stop/disable
the old `ninfer-v100.service` only when inference is idle and both artifacts are
qualified. The new unit conflicts with the old one. A listener collision fails
before selecting/unloading anything. Preserve the old artifact/binary/config for
rollback; do not run two GPU engines. Private backend port ownership must also be
checked before enabling the unit.

For GPU qualification, record the old unit's enabled state and temporarily disable
its autostart before assigning the card. Run the candidate on an isolated loopback
port with no production listener. Restore the old enabled state on qualification
failure; enable the production gateway only after both artifacts pass.

## API

Authenticated control routes:

- `GET /v1/model-control/catalog`: schema1/model ID/label/agents/context.
- `POST /v1/model-control/select`, body `{model,agent}`: 202 asynchronous operation
  or 200 if already resident. Same in-progress target is idempotent; another target
  gets409. Unknown model404 and incompatible client agent403 never load weights.
- `GET /v1/model-control/status`: operation ID, phase, ready, active/target model,
  elapsed seconds, active HTTP requests, error and optional real loader counters.

Phases: unloaded → draining → unloading → loading → warming → ready, or failed.
The old process is reaped before spawning the next. Readiness requires native
health200 and `/v1/models` confirming the selected ID; late startup events cannot
undo readiness or mutate another operation. A load failure is visible and an
explicit selection can retry; no hidden fallback to another model.

`/v1/models` advertises registered models and `ninfer.agents/context_window` metadata;
`/health`200 means one confirmed live resident engine. Generation must request the
resident ID. Another ID gets409 instead of silently using the wrong weights.
Streaming responses are relayed incrementally. Active HTTP/stream leases drain
before unloading; a timeout preserves the old child and reports failure to switch.
This does not reserve an entire agent tool loop between requests. Coordinate
other clients/chats before changing a shared single-GPU model.

Providers for Pi-only models must send `X-Ninfer-Agent: pi`. Missing identity is
conservatively `opencode`. This is a client compatibility policy, not cryptographic
agent authentication. Keep the inference endpoint on the trusted private network;
use authenticated SSH tunnels for remote Desktop control. Control routes require
bearer auth and allow only trusted app/loopback CORS origins. Never expose this
HTTP endpoint publicly without an appropriate authenticated TLS proxy.
Native Responses state is owned by the child and is lost on unload; persistent
agent conversations remain in their owning OpenCode/Pi histories.

## Pi checkpoint conversion

Pinned source: `bytkim/Qwen3.8-27B-pi`, revision
`34d3f21045f7b41453e66c39e137cfc818a603ce`. `pi_source.json` records publisher hashes.
No compatible native prequantized Pi artifact was found. Convert its **own BF16
weights** with the existing groupwise-int recipe; do not substitute base-model
NVFP4 weights or treat a renamed artifact as a finetune. Its graph shape remains
the registered Qwen3.8-27B. Its MTP weights and frontend come from the same checkpoint;
the author's DFlash2 companion is required by the native complete-artifact framing,
while production uses MTP, not a claim of a new Pi-trained DFlash2 accelerator.

After the one pinned download completes:

```sh
python -m tools.convert.qwen3_8_27b.pi_checkpoint /path/to/pi-source /path/to/pi-view
python -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/pi-view --dflash2-model /path/to/pi-view/dflash2 \
  --out /path/to/qwen3_8_27b_pi.ninfer --device cpu --checkpoint-profile pi
```

The verifier checks complete publisher digests, creates a non-copying symlink view,
derives the missing video resource from the pinned processor and records provenance.
Conversion rejects changed or incomplete verification receipts and mismatched
resources/configs/tensors. The default official profile retains its strict hashes.
Conversion changes quantization; do not claim BF16-equivalent quality or NVFP4
performance without actual model evidence. Qualify memory fit, reasoning, tool
calls, images, context and both unload/load directions on the target V100 first.

## Checks

`python3 deploy/v100/test_model_gateway.py` exercises actual CPU-only fixture child
processes, streams, auth/policy, drain/reap/failure/retry and delayed progress. It
is not a GPU or quality benchmark. `pytest tests/convert/qwen3_8_27b` covers the
closed recipes and pinned-source preparation; real-artifact conversion/qualification
remains necessary. No speedup is claimed by adding a lifecycle supervisor.
