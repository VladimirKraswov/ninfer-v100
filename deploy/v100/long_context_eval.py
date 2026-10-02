#!/usr/bin/env python3
"""Bounded, resumable V100 HTTP evaluation; no server management or GPU concurrency.

Prepare ONCE against the selected artifact, then replay the same corpus against each
engine revision (one resident engine at a time). Example:

  python3 deploy/v100/long_context_eval.py prepare --url http://127.0.0.1:18021 \
    --model qwen-v100 --artifact-sha256 SHA --corpus /tmp/v100-corpus.json
  python3 deploy/v100/long_context_eval.py run --url http://127.0.0.1:18021 \
    --model qwen-v100 --artifact-sha256 SHA --corpus /tmp/v100-corpus.json \
    --out /tmp/baseline --engine-revision REV --profile 'mtp=off,kv=int8'
  python3 deploy/v100/long_context_eval.py compare /tmp/baseline /tmp/candidate

NInfer /v1/messages/count_tokens counts the loaded artifact's ANTHROPIC prompt.
It sizes fixtures only: OpenAI usage.prompt_tokens validates the actual input band.
No character-count/token-count approximation or automatic truncation is used. Corpus
files retain exact prompts, hidden checks, construction counts and SHA256 identities.

Six quality fixtures include Russian instructions, code with undisclosed assertions,
structured reasoning, and three-depth retrieval at 16K/128K/~229K. Two performance
templates run at configurable lengths. Quality uses quality.py unchanged (no retries,
length is failure); performance streams and must reach exactly the output cap. NInfer
HTTP cannot suppress EOS, so early-EOS performance rows are INVALID, never stretched
or retried. A native fixed-output benchmark is required to force generation length.

Default maximum: eight generation jobs per invocation; 1800s/request and 3600s/run.
Resume skips failed and interrupted attempts as well as successes. An interrupted
request is never silently retried. Change the run directory for an intentional rerun.
Use --split tuning during optimization; heldout is a separate, fixed acceptance set.
These small suites do not establish a leaderboard score or general reasoning parity.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
from urllib.parse import urlsplit


SCHEMA = 1
CONTEXT = 262144
# Leave the full ±100-token construction band inside the 32K output reserve.
DEFAULT_LENGTHS = (1024, 16384, 131072, 229248)
QUALITY_SCRIPT = Path(__file__).with_name("quality.py")
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api_url(base: str, route: str) -> str:
    # Accept a server origin, /v1 base, or the explicit Chat endpoint.
    for suffix in ("/v1/chat/completions", "/v1"):
        if base.rstrip("/").endswith(suffix):
            base = base.rstrip("/")[:-len(suffix)]
            break
    return base.rstrip("/") + route


def connect(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("URL must be an HTTP(S) endpoint")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = cls(parsed.hostname, parsed.port, timeout=timeout)
    return connection, parsed.path + ("?" + parsed.query if parsed.query else "")


def post_json(url: str, payload: dict, timeout: float) -> dict:
    connection, path = connect(url, timeout)
    try:
        connection.request("POST", path, json.dumps(payload, ensure_ascii=False).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read(8 * 1024 * 1024)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:1000].decode(errors='replace')}")
        return json.loads(body)
    finally:
        connection.close()


def count_payload(model: str, messages: list[dict], thinking: bool) -> dict:
    systems = [m["content"] for m in messages if m["role"] == "system"]
    if len(systems) > 1:
        raise ValueError("fixture must have at most one system message")
    result = {"model": model, "messages": [m for m in messages if m["role"] != "system"],
              "thinking": {"type": "adaptive" if thinking else "disabled"},
              "preserve_thinking": True}
    if systems:
        result["system"] = systems[0]
    if thinking:
        result["output_config"] = {"effort": "xhigh"}
    return result


def artifact_counter(url: str, model: str, timeout: float) -> Callable:
    def count(messages: list[dict], thinking: bool) -> int:
        result = post_json(api_url(url, "/v1/messages/count_tokens"),
                           count_payload(model, messages, thinking), timeout)
        value = result.get("input_tokens")
        if type(value) is not int or value <= 0:
            raise ValueError("count_tokens did not return a positive input_tokens integer")
        return value
    return count


def filler(index: int, split: str) -> str:
    key = hashlib.sha256(f"{split}:ballast:{index}".encode()).hexdigest()[:12]
    return (f"Запись {index:06d}, архив {key}: проверка датчика завершена; "
            "показания сохранены, повторная обработка не требуется.\n")


def fixture_template(name: str, split: str, blocks: int) -> dict:
    """Tests/answers are local metadata and are never appended to model messages."""
    heldout = split == "heldout"
    tag = "H" if heldout else "T"
    verify: dict = {}
    if name == "ru_ledger":
        entries = ([{"id": "К", "n": 7}, {"id": "А", "n": 2}, {"id": "Б", "n": 7}]
                   if heldout else [{"id": "Д", "n": 8}, {"id": "В", "n": 3}, {"id": "Г", "n": 8}])
        expected = sorted((r for r in entries if r["n"] > 5), key=lambda r: (-r["n"], r["id"]))
        task = ("Из записей " + json.dumps(entries, ensure_ascii=False) +
                " оставь n>5, отсортируй по убыванию n, при равенстве по id. "
                'Верни только JSON {"ids":[...],"sum":...}; sum — сумма оставшихся n.')
        verify = {"mode": "json_strict", "expect": {"ids": [r["id"] for r in expected],
                                                     "sum": sum(r["n"] for r in expected)}}
    elif name == "code_intervals":
        task = ("Напиши функцию merge_intervals(intervals) на Python3: объединить "
                "перекрывающиеся и соприкасающиеся замкнутые интервалы [a,b], a<=b. "
                "Вход может быть неотсортирован, содержать дубликаты и отрицательные числа. "
                "Верни отсортированный список списков; вход не изменять. Пустой вход — []. "
                "Ответ — один блок ```python с определением функции, без тестов и объяснений.")
        data = [[9, 11], [-4, -1], [0, 2], [2, 6], [5, 9], [-4, -1]] if heldout else [[5, 8], [1, 3], [3, 5]]
        expected = [[-4, -1], [0, 11]] if heldout else [[1, 8]]
        verify = {"mode": "python_exec", "test": (
            f"x={data!r}; saved=[p[:] for p in x]; assert merge_intervals(x)=={expected!r}; "
            "assert x==saved; assert merge_intervals([])==[]; "
            "assert merge_intervals([[2,2],[1,1]])==[[1,1],[2,2]]; "
            "assert merge_intervals([[1,10],[3,4],[1,10]])==[[1,10]]")}
    elif name == "logic_schedule":
        durations = {"A": 3, "B": 2, "C": 4, "D": 2} if heldout else {"A": 2, "B": 5, "C": 3, "D": 1}
        starts = {"A": 0, "B": durations["A"], "C": durations["A"],
                  "D": durations["A"] + max(durations["B"], durations["C"])}
        task = (f"Есть два одинаковых работника. Длительности задач: {json.dumps(durations)}. "
                "A доступна в момент 0; B и C после A; D после обеих B и C. "
                "Задачи нельзя прерывать, одна задача занимает одного работника. "
                'Найди минимальное время завершения и наиболее ранние старты. Только JSON '
                '{"start":{"A":...,"B":...,"C":...,"D":...},"finish":...}.')
        verify = {"mode": "json_strict", "expect": {"start": starts,
                                                     "finish": starts["D"] + durations["D"]}}
    elif name == "retrieval":
        task = (f"Из архивного документа извлеки значения ровно трёх ключей: {tag}-ALPHA, "
                f"{tag}-BETA, {tag}-GAMMA. Верни только JSON-объект ключ→строковое значение, "
                "без Markdown и дополнительных ключей. Подобные ключи с -OLD не подходят.")
        verify = {"mode": "json_strict", "expect": {
            f"{tag}-{key}": hashlib.sha256(f"{split}:needle:{key}".encode()).hexdigest()[:16]
            for key in ("ALPHA", "BETA", "GAMMA")}}
    elif name == "perf_ru":
        task = ("Продолжай нумерованный технический журнал на русском: для каждого номера "
                "подробно опиши новую проверку программного обеспечения. Начни с 1, "
                "дойди до 10000. Не сокращай список, не делай выводов.")
    elif name == "perf_code":
        task = ("Write Python test functions test_case_0001 through test_case_10000 for "
                "a stable sort. Each function must contain different literal inputs and "
                "assertions. Emit every function in order, no ellipses or explanations.")
    else:
        raise ValueError(f"unknown fixture {name}")
    lines = [filler(i, split) for i in range(blocks)]
    positions = []
    if name == "retrieval":
        for fraction, (key, value) in zip((0.05, 0.5, 0.95), verify["expect"].items()):
            position = min(len(lines), int(blocks * fraction))
            lines.insert(position, f"КОНТРОЛЬНАЯ ЗАПИСЬ {key}: {value}\n")
            positions.append({"key": key, "document_line": position, "nominal_depth": fraction})
        lines.insert(0, f"Устаревшая запись {tag}-ALPHA-OLD: НЕ_ИСПОЛЬЗОВАТЬ\n")
    document = "".join(lines)
    return {"name": name, "kind": "performance" if name.startswith("perf_") else "quality",
            "messages": [{"role": "system", "content": "Выполни задание после архивного документа."},
                         {"role": "user", "content": f"<archive>\n{document}</archive>\n\n{task}"}],
            "verify": verify, "needle_positions": positions}


def fit_fixture(name: str, split: str, target: int, count: Callable,
                tolerance: int = 100) -> dict:
    """Bracket then binary-search actual tokenizer calls, never infer tokens from bytes."""
    thinking = not name.startswith("perf_")
    observations = []
    best = None
    low, high = 0, 1
    bracketed = False
    for _ in range(40):
        blocks = high if not bracketed else (low + high) // 2
        case = fixture_template(name, split, blocks)
        measured = count(case["messages"], thinking)
        observations.append((blocks, measured))
        if best is None or abs(measured - target) < abs(best[0] - target):
            best = measured, case
        # Aim nearer the center than the permitted band: the OpenAI wrapper can
        # differ from the Anthropic wrapper even for the same text-only messages.
        if abs(measured - target) <= min(8, tolerance):
            break
        if measured < target:
            low = blocks
            if blocks == high:
                high *= 2
        else:
            high = blocks
            bracketed = True
        if bracketed and high - low <= 1:
            break
    assert best is not None
    measured, case = best
    if abs(measured - target) > tolerance:
        raise ValueError(f"cannot fit {name} to {target}±{tolerance}; nearest count={measured}")
    case.update({"case_id": f"{split}/{name}/{target}", "target_input_tokens": target,
                 "counted_input_tokens": measured, "count_calls": len(observations),
                 "count_protocol": "anthropic_messages_count_tokens",
                 "openai_input_count_verified": False, "input_tolerance": tolerance})
    case["fixture_sha256"] = digest(case)
    return case


def prepare(args: argparse.Namespace) -> None:
    destination = Path(args.corpus)
    if destination.exists():
        raise ValueError("corpus already exists; reuse it or choose a new path")
    counter = artifact_counter(args.url, args.model, args.count_timeout)
    specs = [(name, args.lengths[0]) for name in ("ru_ledger", "code_intervals", "logic_schedule")]
    specs.extend(("retrieval", n) for n in args.lengths[1:])
    specs.extend((name, n) for name in ("perf_ru", "perf_code") for n in args.lengths)
    cases = []
    for name, target in specs:
        case = fit_fixture(name, args.split, target, counter, args.tolerance)
        cases.append(case)
        print(f"prepared {case['case_id']}: Anthropic count={case['counted_input_tokens']}", flush=True)
    corpus = {"schema_version": SCHEMA, "split": args.split,
              "artifact_sha256": args.artifact_sha256, "context_capacity": CONTEXT,
              "count_source": api_url(args.url, "/v1/messages/count_tokens"), "cases": cases,
              "limits": "Construction counts use Anthropic rendering; OpenAI usage must confirm input band. "
                        "Needle depths are document-line fractions, not measured token-depth fractions."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")


def native_metrics(response: dict, wall_s: float, ttft_s: float | None = None) -> dict:
    usage, timing = response.get("usage") or {}, response.get("timings") or {}
    result = {"wall_s": wall_s, "stream_first_visible_token_s": ttft_s,
              "prompt_tokens": usage.get("prompt_tokens"),
              "completion_tokens": usage.get("completion_tokens"),
              "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
              "cached_prompt_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
              "native_prefill_s": None, "native_decode_s": None,
              "native_prefill_tok_s": None, "native_decode_tok_s": None}
    for phase, time_key, rate_key in (("prefill", "prompt_ms", "prompt_per_second"),
                                      ("decode", "predicted_ms", "predicted_per_second")):
        value, rate = timing.get(time_key), timing.get(rate_key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            result[f"native_{phase}_s"] = value / 1000
        if isinstance(rate, (int, float)) and math.isfinite(rate) and rate >= 0:
            result[f"native_{phase}_tok_s"] = rate
    result["native_timings"] = timing
    # SSE chunks may contain many tokens: never count chunks or re-tokenize output.
    return result


def stream_request(url: str, payload: dict, timeout: float) -> tuple[dict, float, float | None]:
    connection, path = connect(url, timeout)
    started = time.monotonic()
    deadline = started + timeout
    aggregate: dict = {"choices": [{"message": {"content": "", "reasoning_content": ""},
                                   "finish_reason": None}]}
    first = None
    done = False
    received_bytes = 0
    try:
        connection.request("POST", path, json.dumps(payload, ensure_ascii=False).encode(),
                           {"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(1000).decode(errors='replace')}")
        data = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("stream exceeded request deadline")
            # getresponse may detach a closing socket from connection; the response
            # still owns it. Bound every read by the remaining whole-request deadline.
            sock = connection.sock or response.fp.raw._sock
            sock.settimeout(remaining)
            line = response.readline(1024 * 1024)
            received_bytes += len(line)
            if received_bytes > 32 * 1024 * 1024:
                raise ValueError("stream exceeds 32MiB response bound")
            if not line:
                break
            if line.strip():
                if line.startswith(b"data:"):
                    data.append(line[5:].strip())
                continue
            if not data:
                continue
            event = b"\n".join(data)
            data = []
            if event == b"[DONE]":
                done = True
                break
            item = json.loads(event)
            if item.get("error"):
                raise RuntimeError(f"stream error: {item['error']}")
            for field in ("usage", "timings", "id", "model"):
                if item.get(field) is not None:
                    aggregate[field] = item[field]
            for choice in item.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    if first is None:
                        first = time.monotonic() - started
                message = aggregate["choices"][0]["message"]
                for key in ("content", "reasoning_content"):
                    message[key] += delta.get(key) or ""
                if choice.get("finish_reason") is not None:
                    aggregate["choices"][0]["finish_reason"] = choice["finish_reason"]
        if not done or aggregate["choices"][0]["finish_reason"] is None:
            raise RuntimeError("incomplete SSE response: missing finish reason or [DONE]")
        return aggregate, time.monotonic() - started, first
    finally:
        connection.close()


def trial_case(case: dict, seed: int) -> dict:
    return {**case, "case_id": f"{case['case_id']}/seed-{seed}",
            "extra": {"enable_thinking": True, "preserve_thinking": True}}


def actual_seed(case_id: str, seed: int) -> int:
    # Matches quality.py exactly, so both lanes record the seed actually sent.
    return int.from_bytes(hashlib.sha256(f"{seed}:{case_id}".encode()).digest()[:4], "big")


def quality_request(case: dict, args: argparse.Namespace, seed: int,
                    timeout: float) -> dict:
    """Reuse existing code/JSON scorers and pass@1 behavior without importing its CLI."""
    with tempfile.TemporaryDirectory(prefix="ninfer-quality-") as directory:
        suite, output = Path(directory) / "suite.jsonl", Path(directory) / "response.jsonl"
        suite.write_text(json.dumps(case, ensure_ascii=False) + "\n")
        command = [sys.executable, str(QUALITY_SCRIPT), "--url", api_url(args.url, "/v1/chat/completions"),
                   "--model", args.model, "--suite", str(suite), "--out", str(output),
                   "--seed", str(seed), "--retry", "0", "--effort", "xhigh",
                   "--max-tokens", str(args.quality_output), "--thinking-budget", str(args.thinking_budget),
                   "--timeout", str(max(1, math.floor(timeout)))]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout + 30)
        return json.loads(output.read_text().strip())


def validate_result(case: dict, response: dict, output_budget: int, quality_ok: bool | None) -> tuple[bool, str]:
    usage = response.get("usage") or {}
    prompt, completed = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(prompt) is not int or type(completed) is not int:
        return False, "missing actual OpenAI token usage"
    if abs(prompt - case["target_input_tokens"]) > case["input_tolerance"]:
        return False, "actual OpenAI input outside requested token band"
    if prompt + output_budget > CONTEXT:
        return False, "actual prompt plus reserved output exceeds 262144 context"
    timing = response.get("timings") or {}
    if "prompt_n" in timing and "cache_n" in timing and timing["prompt_n"] + timing["cache_n"] != prompt:
        return False, "native prompt timing counts disagree with actual usage"
    if "predicted_n" in timing and timing["predicted_n"] != completed:
        return False, "native generation timing count disagrees with actual usage"
    finish = response.get("choices", [{}])[0].get("finish_reason")
    if case["kind"] == "performance":
        ok = completed == output_budget and finish == "length"
        return ok, "fixed output cap reached" if ok else "invalid fixed-output trial: early stop or wrong usage"
    if finish != "stop":
        return False, f"quality incomplete: finish_reason={finish!r}"
    return bool(quality_ok), "quality scorer passed" if quality_ok else "quality scorer failed"


def append_record(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"damaged journal at {path}:{number}; preserve it and use a new run directory") from exc
    return result


def run(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus)
    corpus = json.loads(corpus_path.read_text())
    if corpus["schema_version"] != SCHEMA:
        raise ValueError("unsupported corpus schema version")
    if corpus["artifact_sha256"] != args.artifact_sha256 or corpus["context_capacity"] != CONTEXT:
        raise ValueError("corpus artifact/context differs from selected run")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": SCHEMA, "corpus_sha256": file_sha(corpus_path),
                "artifact_sha256": args.artifact_sha256, "artifact_identity_source": "operator-supplied SHA256",
                "engine_revision": args.engine_revision, "profile": args.profile,
                "url": args.url, "model": args.model, "split": corpus["split"], "context_capacity": CONTEXT,
                "quality_output": args.quality_output, "thinking_budget": args.thinking_budget,
                "performance_output": args.performance_output, "effort": "xhigh", "seeds": args.seeds,
                "quality_script_sha256": file_sha(QUALITY_SCRIPT), "sampling": SAMPLING,
                "harness_sha256": file_sha(Path(__file__)),
                "planned_trial_ids": [trial_case(c, seed)["case_id"] for c in corpus["cases"] for seed in args.seeds],
                "limits": "HTTP cannot force EOS suppression. TTFT is first visible content/reasoning, not first committed token. "
                          "Quality runs aggregate through quality.py; only performance records streaming TTFT. "
                          "Cache state is observed, not reset. Artifact SHA is an operator attestation, not an HTTP proof."}
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("run identity changed; use a new output directory")
    else:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    journal = root / "results.jsonl"
    previous = load_records(journal)
    attempted = {r["trial_id"] for r in previous}
    started, submitted = time.monotonic(), 0
    for case in corpus["cases"]:
        if args.only != "all" and args.only != case["kind"]:
            continue
        if args.filter and args.filter not in case["case_id"]:
            continue
        if digest({k: v for k, v in case.items() if k != "fixture_sha256"}) != case["fixture_sha256"]:
            raise ValueError("fixture hash mismatch")
        budget = args.quality_output if case["kind"] == "quality" else args.performance_output
        if case["target_input_tokens"] + case["input_tolerance"] + budget > CONTEXT:
            raise ValueError(f"{case['case_id']}: token band and output budget exceed context")
        for seed in args.seeds:
            trial = trial_case(case, seed)
            cid = trial["case_id"]
            if cid in attempted:
                continue
            remaining = args.max_seconds - (time.monotonic() - started)
            if submitted >= args.max_jobs or remaining <= 35:
                print("Invocation bound reached; rerun the same command to resume.", flush=True)
                return
            timeout = min(args.timeout, remaining - 30)
            base = {"trial_id": cid, "case_id": case["case_id"], "kind": case["kind"],
                    "fixture_sha256": case["fixture_sha256"], "seed": actual_seed(cid, seed),
                    "seed_repeat": seed, "target_input_tokens": case["target_input_tokens"],
                    "construction_input_tokens": case["counted_input_tokens"],
                    "construction_count_protocol": case["count_protocol"],
                    "output_budget": budget, "attempts": 1}
            append_record(journal, {**base, "event": "started", "utc_epoch": time.time()})
            submitted += 1
            try:
                if case["kind"] == "quality":
                    raw = quality_request(trial, args, seed, timeout)
                    if "response" not in raw:
                        raise RuntimeError(raw.get("why", "quality runner returned no response"))
                    response, wall, first = raw["response"], raw["wall_s"], None
                    scorer_ok = raw["ok"]
                else:
                    payload = {"model": args.model, "messages": case["messages"], "max_tokens": budget,
                               "stream": True, "stream_options": {"include_usage": True},
                               "reasoning_effort": "none", "enable_thinking": False,
                               "preserve_thinking": True, "seed": base["seed"],
                               **SAMPLING}
                    response, wall, first = stream_request(api_url(args.url, "/v1/chat/completions"), payload, timeout)
                    raw, scorer_ok = {}, None
                ok, why = validate_result(case, response, budget, scorer_ok)
                record = {**base, "event": "finished", "ok": ok, "why": why,
                          "scorer": raw, "response": response,
                          "metrics": native_metrics(response, wall, first)}
                append_record(journal, record)
                print(f"{cid}: {'PASS' if ok else 'FAIL'} {why}", flush=True)
            except Exception as exc:
                append_record(journal, {**base, "event": "finished", "ok": False,
                                        "why": f"{type(exc).__name__}: {exc}", "transport_failed": True})
                # A timed-out server may still be cancelling GPU work; do not enqueue
                # more jobs and accidentally benchmark against an unknown predecessor.
                raise RuntimeError(f"stopped after failed request {cid}; no retry") from exc


def compare(left: str, right: str) -> dict:
    baseline, candidate = Path(left), Path(right)
    a = json.loads((baseline / "metadata.json").read_text())
    b = json.loads((candidate / "metadata.json").read_text())
    controlled = ("corpus_sha256", "artifact_sha256", "model", "context_capacity", "quality_output",
                  "thinking_budget", "performance_output", "effort", "seeds", "quality_script_sha256", "sampling",
                  "planned_trial_ids", "harness_sha256")
    missing = [f"{label}:{key}" for label, identity in (("baseline", a), ("candidate", b))
               for key in controlled if key not in identity]
    if missing:
        raise ValueError(
            "incomplete comparison metadata: " + ", ".join(missing) + ". "
            "Legacy runs without harness_sha256 require the original remote harness snapshot/SHA "
            "and a manual audit; do not backfill historical metadata with the current script hash.")
    different = [key for key in controlled if a[key] != b[key]]
    if different:
        raise ValueError("uncontrolled comparison: " + ", ".join(different))
    ar = {r["trial_id"]: r for r in load_records(baseline / "results.jsonl") if r["event"] == "finished"}
    br = {r["trial_id"]: r for r in load_records(candidate / "results.jsonl") if r["event"] == "finished"}
    rows = []
    for key in sorted(ar.keys() & br.keys()):
        x, y = ar[key], br[key]
        row = {"trial_id": key, "kind": x["kind"], "baseline_ok": x["ok"], "candidate_ok": y["ok"]}
        xm, ym = x.get("metrics", {}), y.get("metrics", {})
        if x["kind"] == "performance" and x["ok"] and y["ok"]:
            same_input = xm.get("prompt_tokens") == ym.get("prompt_tokens")
            same_cache = xm.get("cached_prompt_tokens") is not None and xm.get("cached_prompt_tokens") == ym.get("cached_prompt_tokens")
            row["prefill_comparable"] = same_input and same_cache
            for phase in ("prefill", "decode"):
                xv, yv = xm.get(f"native_{phase}_tok_s"), ym.get(f"native_{phase}_tok_s")
                if same_input and (phase == "decode" or same_cache) and xv and yv:
                    row[f"{phase}_speedup"] = yv / xv
        rows.append(row)
    quality_rows = [row for row in rows if row["kind"] == "quality"]
    return {"paired": rows, "paired_quality_pass_at_1": {
                "baseline_passed": sum(row["baseline_ok"] for row in quality_rows),
                "candidate_passed": sum(row["candidate_ok"] for row in quality_rows),
                "trials": len(quality_rows)},
            "incomplete_baseline": sorted(set(a["planned_trial_ids"]) - ar.keys()),
            "incomplete_candidate": sorted(set(b["planned_trial_ids"]) - br.keys()),
            "unpaired_baseline": sorted(ar.keys() - br.keys()),
            "unpaired_candidate": sorted(br.keys() - ar.keys()),
            "limits": "Only valid fixed-output pairs support speed comparisons; quality rows compare pass@1. "
                      "Different cache residency excludes prefill speedup. Small heldout suite is not a reasoning benchmark."}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="freeze token-counted fixtures; no generation")
    run_parser = sub.add_parser("run", help="bounded serial generation, same frozen corpus")
    for child in (prepare_parser, run_parser):
        child.add_argument("--url", required=True)
        child.add_argument("--model", default="qwen-v100")
        child.add_argument("--artifact-sha256", required=True)
        child.add_argument("--corpus", required=True)
    prepare_parser.add_argument("--split", choices=("tuning", "heldout"), default="heldout")
    prepare_parser.add_argument("--lengths", type=int, nargs=4, default=list(DEFAULT_LENGTHS), metavar=("SHORT", "16K", "128K", "LONG"))
    prepare_parser.add_argument("--tolerance", type=int, default=100)
    prepare_parser.add_argument("--count-timeout", type=float, default=60)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--engine-revision", required=True)
    run_parser.add_argument("--profile", required=True, help="record actual engine flags/KV/MTP/prefill/GPU settings")
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003])
    run_parser.add_argument("--quality-output", type=int, default=32768)
    run_parser.add_argument("--thinking-budget", type=int, choices=(8192, 16384), default=16384)
    run_parser.add_argument("--performance-output", type=int, default=1024)
    run_parser.add_argument("--timeout", type=float, default=1800)
    run_parser.add_argument("--max-seconds", type=float, default=3600)
    run_parser.add_argument("--max-jobs", type=int, default=8)
    run_parser.add_argument("--only", choices=("all", "quality", "performance"), default="all")
    run_parser.add_argument("--filter", default="")
    compare_parser = sub.add_parser("compare", help="pair compatible run records")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    args = parser.parse_args(argv)
    if args.command != "compare":
        if len(args.artifact_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.artifact_sha256):
            parser.error("artifact SHA256 must be 64 lowercase hex digits")
    if args.command == "prepare":
        if not 1 <= args.tolerance <= 100 or min(args.lengths) <= args.tolerance or args.count_timeout <= 0:
            parser.error("positive lengths/timeouts and tolerance 1..100 required")
        if list(args.lengths) != sorted(set(args.lengths)):
            parser.error("four input lengths must be distinct and ascending")
        if max(args.lengths) + args.tolerance + 32768 > CONTEXT:
            parser.error("input band must leave 32768 output tokens inside 262144 context")
    if args.command == "run":
        if min(args.quality_output, args.performance_output, args.timeout, args.max_seconds, args.max_jobs) <= 0:
            parser.error("budgets, timeouts and invocation bounds must be positive")
        if args.quality_output < args.thinking_budget + 1024:
            parser.error("quality output must leave at least 1024 tokens after thinking budget")
        if len(args.seeds) != len(set(args.seeds)):
            parser.error("repeat seeds must be distinct")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        print(json.dumps(compare(args.baseline, args.candidate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
