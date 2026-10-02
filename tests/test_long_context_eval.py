"""CPU-only HTTP fixtures for bounded V100 evaluation; no model/GPU required."""
from __future__ import annotations

from contextlib import contextmanager
import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "deploy/v100/long_context_eval.py"
SPEC = importlib.util.spec_from_file_location("long_context_eval", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
SHA = "a" * 64


@contextmanager
def endpoint(callback):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            callback(self, payload)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def send_json(handler, value):
    data = json.dumps(value).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def send_stream(handler, value, *, complete=True):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()
    events = [{"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {"reasoning_content": "проверка"}, "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {"content": value["choices"][0]["message"]["content"]},
                            "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": value["choices"][0]["finish_reason"]}]},
              {"choices": [], "usage": value["usage"], "timings": value["timings"]}]
    handler.wfile.write(b": keepalive\n\n")
    for item in events:
        handler.wfile.write(b"data: " + json.dumps(item).encode() + b"\n\n")
        handler.wfile.flush()
        time.sleep(0.002)
    if complete:
        handler.wfile.write(b"data: [DONE]\n\n")


def response(content="ok", finish="stop", prompt=1024, output=12):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": output,
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 4}},
            "timings": {"cache_n": 0, "prompt_n": prompt, "prompt_ms": 2000,
                        "prompt_per_second": prompt / 2, "predicted_n": output,
                        "predicted_ms": 500, "predicted_per_second": max(output - 1, 0) * 2}}


def fixture(name="json", kind="quality"):
    case = {"case_id": f"heldout/{name}/1024", "name": name, "kind": kind,
            "messages": [{"role": "user", "content": name}],
            "verify": {"mode": "json_strict", "expect": {"x": 1}},
            "target_input_tokens": 1024, "counted_input_tokens": 1008,
            "count_protocol": "anthropic_messages_count_tokens", "input_tolerance": 100}
    if name == "code":
        case["verify"] = {"mode": "python_exec", "test": "assert f([3,1,3]) == [3,1]; assert f([])==[]"}
    case["fixture_sha256"] = evaluation.digest(case)
    return case


def write_corpus(path, cases):
    path.write_text(json.dumps({"schema_version": 1, "artifact_sha256": SHA,
                               "context_capacity": 262144, "split": "heldout", "cases": cases}))


def run_args(url, corpus, output, *extra):
    return evaluation.parse_args(["run", "--url", url, "--corpus", str(corpus), "--out", str(output),
                                  "--artifact-sha256", SHA, "--engine-revision", "fixture-revision",
                                  "--profile", "fixture-only", "--timeout", "5", *extra])


class LongContextEvaluationTests(unittest.TestCase):
    def test_sizing_uses_tokenizer_calls_and_keeps_three_needles(self):
        calls = []

        def count(messages, thinking):
            calls.append((messages, thinking))
            # Deliberately different from character length; synthetic tokenizer oracle.
            return 173 + messages[-1]["content"].count("Запись ") * 29

        case = evaluation.fit_fixture("retrieval", "heldout", 16384, count)
        self.assertLessEqual(abs(case["counted_input_tokens"] - 16384), 100)
        self.assertGreater(len(calls), 1)
        self.assertLessEqual(len(calls), 40)
        self.assertTrue(all(thinking for _, thinking in calls))
        self.assertEqual(len(case["verify"]["expect"]), 3)
        for key, value in case["verify"]["expect"].items():
            self.assertIn(f"{key}: {value}", case["messages"][-1]["content"])
        self.assertFalse(case["openai_input_count_verified"])
        self.assertEqual(evaluation.digest({k: v for k, v in case.items() if k != "fixture_sha256"}),
                         case["fixture_sha256"])

    def test_count_endpoint_records_artifact_protocol_not_chat_estimate(self):
        received = []

        def handler(h, body):
            received.append((h.path, body))
            send_json(h, {"input_tokens": 16371})

        with endpoint(handler) as url:
            counter = evaluation.artifact_counter(url + "/v1/chat/completions", "qwen-v100", 1)
            n = counter([{"role": "system", "content": "rules"},
                         {"role": "user", "content": "payload"}], True)
        self.assertEqual(n, 16371)
        self.assertEqual(received[0][0], "/v1/messages/count_tokens")
        self.assertEqual(received[0][1]["system"], "rules")
        self.assertEqual(received[0][1]["output_config"], {"effort": "xhigh"})
        self.assertEqual(received[0][1]["thinking"], {"type": "adaptive"})

    def test_prepare_freezes_six_quality_cases_and_two_performance_templates(self):
        def counter(messages, thinking):
            return 173 + messages[-1]["content"].count("Запись ") * 29

        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.json"
            args = evaluation.parse_args(["prepare", "--url", "http://127.0.0.1:1", "--corpus", str(corpus),
                                          "--artifact-sha256", SHA])
            self.assertLessEqual(max(args.lengths) + args.tolerance + 32768, 262144)
            args.lengths = [1024, 2048, 3072, 4096]
            with patch.object(evaluation, "artifact_counter", return_value=counter):
                evaluation.prepare(args)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    evaluation.prepare(args)
            cases = json.loads(corpus.read_text())["cases"]
            self.assertEqual(sum(c["kind"] == "quality" for c in cases), 6)
            self.assertEqual(sum(c["kind"] == "performance" for c in cases), 8)
            self.assertEqual({c["name"] for c in cases if c["kind"] == "performance"}, {"perf_ru", "perf_code"})

    def test_actual_usage_and_finish_reason_are_required(self):
        quality = fixture()
        self.assertEqual(evaluation.validate_result(quality, response('{"x":1}'), 32768, True)[0], True)
        self.assertFalse(evaluation.validate_result(quality, response('{"x":1}', "length"), 32768, True)[0])
        self.assertFalse(evaluation.validate_result(quality, response('{"x":1}', prompt=1150), 32768, True)[0])
        self.assertFalse(evaluation.validate_result(quality, {"choices": []}, 32768, True)[0])
        performance = fixture("perf", "performance")
        self.assertTrue(evaluation.validate_result(performance, response(finish="length", output=1024), 1024, None)[0])
        self.assertFalse(evaluation.validate_result(performance, response(finish="stop", output=1024), 1024, None)[0])
        self.assertFalse(evaluation.validate_result(performance, response(finish="length", output=1023), 1024, None)[0])
        inconsistent = response(finish="length", output=1024)
        inconsistent["timings"]["predicted_n"] = 2048
        self.assertFalse(evaluation.validate_result(performance, inconsistent, 1024, None)[0])

    def test_stream_uses_final_usage_native_timing_not_chunk_count(self):
        expected = response("final answer", "length", output=1024)
        with endpoint(lambda h, b: send_stream(h, expected)) as url:
            result, wall, first = evaluation.stream_request(url + "/v1/chat/completions", {}, 2)
        metrics = evaluation.native_metrics(result, wall, first)
        self.assertEqual(result["choices"][0]["message"]["reasoning_content"], "проверка")
        self.assertEqual(metrics["completion_tokens"], 1024)
        self.assertEqual(metrics["native_decode_tok_s"], 2046)
        self.assertEqual(metrics["native_prefill_s"], 2)
        self.assertGreater(first, 0)
        self.assertLess(first, wall)
        self.assertNotEqual(metrics["native_decode_s"], wall - first)

    def test_missing_stream_terminal_is_not_success(self):
        with endpoint(lambda h, b: send_stream(h, response(), complete=False)) as url:
            with self.assertRaisesRegex(RuntimeError, "incomplete SSE"):
                evaluation.stream_request(url + "/v1/chat/completions", {}, 2)

    def test_waiting_for_stream_data_obeys_request_timeout(self):
        def handler(h, body):
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.end_headers()
            time.sleep(0.15)

        with endpoint(handler) as url:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                evaluation.stream_request(url + "/v1/chat/completions", {}, 0.04)
            self.assertLess(time.monotonic() - started, 0.5)

    def test_quality_runner_reuse_strict_json_code_no_retry_and_resume(self):
        received = []

        def handler(h, body):
            received.append(body)
            name = body["messages"][-1]["content"]
            content = {"json": '{"x":1}', "length": '{"x":1}',
                       "wrapped": '```json\n{"x":1}\n```',
                       "code": "```python\ndef f(xs):\n    return list(dict.fromkeys(xs))\n```"}[name]
            send_json(h, response(content, "length" if name == "length" else "stop"))

        with endpoint(handler) as url, tempfile.TemporaryDirectory() as directory:
            corpus, output = Path(directory) / "corpus.json", Path(directory) / "run"
            write_corpus(corpus, [fixture(n) for n in ("json", "length", "wrapped", "code")])
            args = run_args(url, corpus, output)
            evaluation.run(args)
            records = [r for r in evaluation.load_records(output / "results.jsonl") if r["event"] == "finished"]
            self.assertEqual([r["ok"] for r in records], [True, True, False, False, False, False, True, True])
            self.assertEqual(len(received), 8)
            self.assertEqual(len({b["seed"] for b in received}), 8)
            self.assertTrue(all(b["reasoning_effort"] == "xhigh" and b["thinking_budget"] == 16384
                                and b["max_tokens"] == 32768 and b["enable_thinking"] for b in received))
            evaluation.run(args)
            self.assertEqual(len(received), 8)
            self.assertTrue(all(r["scorer"]["attempts"] == 1 for r in records))
            self.assertTrue(all(r["metrics"]["prompt_tokens"] == 1024 for r in records))
            self.assertEqual(records[0]["construction_input_tokens"], 1008)

    def test_interrupted_trial_is_not_automatically_reissued(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus, output = Path(directory) / "corpus.json", Path(directory) / "run"
            write_corpus(corpus, [fixture()])
            output.mkdir()
            evaluation.append_record(output / "results.jsonl", {
                "event": "started", "trial_id": "heldout/json/1024/seed-7"})
            # No HTTP server exists: success proves the uncertain attempt wasn't retried.
            evaluation.run(run_args("http://127.0.0.1:1", corpus, output, "--seeds", "7"))
            self.assertEqual(len(evaluation.load_records(output / "results.jsonl")), 1)

    def test_max_jobs_bounds_requests_and_resume_keeps_identity(self):
        received = []

        def handler(h, body):
            received.append(body)
            send_json(h, response('{"x":1}'))

        with endpoint(handler) as url, tempfile.TemporaryDirectory() as directory:
            corpus, output = Path(directory) / "corpus.json", Path(directory) / "run"
            write_corpus(corpus, [fixture()])
            args = run_args(url, corpus, output, "--max-jobs", "1")
            evaluation.run(args)
            self.assertEqual(len(received), 1)
            evaluation.run(args)
            self.assertEqual(len(received), 2)
            args.profile = "different"
            with self.assertRaisesRegex(ValueError, "identity changed"):
                evaluation.run(args)

    def test_compare_requires_same_artifact_and_excludes_cache_mismatch(self):
        received = []

        def handler(h, body):
            received.append(body)
            send_stream(h, response(finish="length", output=1024))

        with endpoint(handler) as url, tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.json"
            write_corpus(corpus, [fixture("perf", "performance")])
            a, b = Path(directory) / "a", Path(directory) / "b"
            evaluation.run(run_args(url, corpus, a, "--seeds", "7"))
            evaluation.run(run_args(url, corpus, b, "--seeds", "7"))
            paired = evaluation.compare(str(a), str(b))["paired"][0]
            self.assertEqual(paired["decode_speedup"], 1)
            self.assertEqual(paired["prefill_speedup"], 1)
            original_metadata = {path: (path / "metadata.json").read_text() for path in (a, b)}
            for legacy_paths in ((a,), (b,), (a, b)):
                with self.subTest(legacy_paths=[path.name for path in legacy_paths]):
                    for path in legacy_paths:
                        legacy = json.loads(original_metadata[path])
                        del legacy["harness_sha256"]
                        (path / "metadata.json").write_text(json.dumps(legacy))
                    before_compare = {path: (path / "metadata.json").read_text() for path in (a, b)}
                    with self.assertRaisesRegex(ValueError, "harness_sha256.*manual audit"):
                        evaluation.compare(str(a), str(b))
                    for path in (a, b):
                        self.assertEqual((path / "metadata.json").read_text(), before_compare[path])
                        (path / "metadata.json").write_text(original_metadata[path])
            lines = evaluation.load_records(b / "results.jsonl")
            lines[-1]["metrics"]["cached_prompt_tokens"] = 512
            (b / "results.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
            paired = evaluation.compare(str(a), str(b))["paired"][0]
            self.assertNotIn("prefill_speedup", paired)
            self.assertEqual(paired["decode_speedup"], 1)
            identity = json.loads((b / "metadata.json").read_text())
            identity["artifact_sha256"] = "b" * 64
            (b / "metadata.json").write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError, "artifact_sha256"):
                evaluation.compare(str(a), str(b))

    def test_hidden_code_checks_and_tuning_split_are_separate(self):
        heldout = evaluation.fixture_template("code_intervals", "heldout", 1)
        tuning = evaluation.fixture_template("code_intervals", "tuning", 1)
        self.assertNotEqual(heldout["verify"], tuning["verify"])
        self.assertNotIn("assert ", heldout["messages"][-1]["content"])
        self.assertNotIn(str([[-4, -1], [0, 11]]), heldout["messages"][-1]["content"])

    def test_context_headroom_and_duplicate_seed_validation(self):
        with self.assertRaises(SystemExit):
            evaluation.parse_args(["prepare", "--url", "http://localhost", "--corpus", "/tmp/no-write",
                                   "--artifact-sha256", SHA, "--lengths", "1024", "16384", "131072", "230000"])
        with self.assertRaises(SystemExit):
            run_args("http://localhost", Path("not-read"), Path("not-created"), "--seeds", "1", "1")


if __name__ == "__main__":
    unittest.main()
