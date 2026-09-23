#!/usr/bin/env python3
"""Run quality suite against an OpenAI-compatible endpoint and score deterministically.

Usage: run_quality.py --url URL --model MODEL --suite quality_suite.jsonl --out RESULT.jsonl
                      [--only kind[,kind]] [--filter substring] [--retry 1]
Each case -> one JSONL record with raw output + verdict. Resumable: existing case_ids in --out are skipped.
"""

import argparse, hashlib, json, re, subprocess, sys, time, urllib.request, os

AP = argparse.ArgumentParser()
AP.add_argument("--url", required=True)  # .../v1/chat/completions
AP.add_argument("--model", required=True)
AP.add_argument("--suite", required=True)
AP.add_argument("--out", required=True)
AP.add_argument("--only", default="")
AP.add_argument("--filter", default="")
AP.add_argument("--retry", type=int, default=0)
AP.add_argument("--seed", type=int, default=20260923)
AP.add_argument("--effort", choices=("low", "medium", "xhigh"), default="medium")
AP.add_argument("--timeout", type=int, default=3600)
A = AP.parse_args()

SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 4294967295,
}


def request(payload):
    req = urllib.request.Request(
        A.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=A.timeout) as r:
        return json.loads(r.read())


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return max(m, key=len)
    # fallback: whole text if it looks like code
    if re.search(r"^\s*(def|class) ", text, re.M):
        return text
    return None


def run_python(code, test):
    script = (
        code
        + "\n\nassert_fn_tests = lambda: ("
        + test
        + ")\n"
        + 'exec("import types\\ntests = []\\n")\n'
    )
    # asserts may contain ';' separated statements and multiple exprs — run as exec block
    runner = (
        code + "\n\n" + "import traceback, sys\n"
        "try:\n    " + test.replace("\n", " ") + "\nexcept Exception as e:\n"
        "    print('TESTFAIL', type(e).__name__, e); sys.exit(1)\nprint('TESTOK')\n"
    )
    p = subprocess.run(
        [sys.executable, "-I", "-c", runner], capture_output=True, text=True, timeout=25
    )
    return ("TESTOK" in p.stdout, p.stdout + p.stderr[-800:])


def parse_json_block(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def normalize(o):
    if isinstance(o, dict):
        return {k: normalize(v) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return [normalize(x) for x in o]
    return o


def score(case, msg):
    v = case["verify"]
    content = (msg.get("content") or "").strip()
    tc = msg.get("tool_calls") or []
    mode = v["mode"]
    if mode == "python_exec":
        code = extract_code(content)
        if not code:
            return False, "no code block"
        ok, log = run_python(code, v["test"])
        return ok, log if not ok else "ok"
    if mode == "regex":
        rx = re.compile(v["pattern"])
        ok = bool(
            rx.fullmatch(content)
            if v.get("flags") == "fullmatch"
            else rx.search(content)
        )
        if ok and v.get("must"):
            ok = all(s in content for s in v["must"])
        return ok, f"content={content[:200]!r}"
    if mode == "must":
        missing = [s for s in v["must"] if s not in content]
        ok = not missing
        if ok and v.get("max_words"):
            ok = len(content.split()) <= v["max_words"]
        return ok, f"missing={missing} words={len(content.split())}"
    if mode == "exact":
        return content == v["expect"], f"got={content[:200]!r}"
    if mode == "exact_lines":
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        ok = all(any(want.strip() in got for got in lines) for want in v["lines"])
        return ok, f"lines={lines[:8]}"
    if mode == "no_char":
        words = content.split()
        ok = (v["char"] not in content.lower()) and len(words) == v["words"]
        return ok, f"words={len(words)} content={content[:120]!r}"
    if mode == "word_count":
        n = len(content.split())
        return n == v["count"], f"n={n} content={content[:120]!r}"
    if mode == "json_strict":
        got = parse_json_block(content)
        ok = got is not None and normalize(got) == normalize(v["expect"])
        return ok, f"got={got}"
    if mode == "paragraphs":
        paras = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
        ok = len(paras) == v["count"] and len(paras[0].split()) <= v["first_max_words"]
        return (
            ok,
            f"paras={len(paras)} first_words={len(paras[0].split()) if paras else 0}",
        )
    if mode == "forbid_regex":
        m = re.search(v["pattern"], content)
        return (m is None), f"hit={m.group(0) if m else None}"
    if mode == "tool_call":
        for c in tc:
            fn = c.get("function", {})
            if fn.get("name") == v["name"] and re.search(
                v["arg_regex"], fn.get("arguments") or ""
            ):
                return True, "ok"
        return (
            False,
            f"tool_calls={json.dumps(tc, ensure_ascii=False)[:300]} content={content[:150]!r}",
        )
    if mode == "no_tool_call_then_regex":
        ok = (not tc) and bool(re.search(v["pattern"], content))
        return ok, f"tc={len(tc)} content={content[:150]!r}"
    if mode == "must_semicolon":
        prompt = case["messages"][0]["content"]
        m = re.search(r"SECREF-K[\d-]+: (\d+); дата-протокол ([\d.]+); код ([A-Z]+-\d+)", prompt)
        want = list(m.groups())
        missing = [w for w in want if w not in content]
        return (not missing), f"missing={missing} content={content[:200]!r}"
    return False, f"unknown mode {mode}"


done = set()
if os.path.exists(A.out):
    with open(A.out) as f:
        for line in f:
            try:
                done.add(json.loads(line)["case_id"])
            except Exception:
                pass

only = set(A.only.split(",")) if A.only else None
with open(A.suite) as f:
    suite = [json.loads(l) for l in f]

with open(A.out, "a") as out:
    for case in suite:
        cid = case["case_id"]
        if cid in done:
            continue
        if only and case["kind"] not in only:
            continue
        if A.filter and A.filter not in cid:
            continue
        payload = {
            "model": A.model,
            "messages": case["messages"],
            "max_tokens": case.get("max_tokens", 32768),
            "stream": False,
        }
        payload.update(SAMPLING)
        payload["seed"] = int.from_bytes(hashlib.sha256(f"{A.seed}:{cid}".encode()).digest()[:4], "big")
        payload["reasoning_effort"] = A.effort
        if case.get("extra"):
            payload.update(case["extra"])
        rec = {"case_id": cid, "kind": case["kind"], "attempts": 0}
        t0 = time.time()
        last = None
        for attempt in range(A.retry + 1):
            rec["attempts"] = attempt + 1
            try:
                resp = request(payload)
                ch = resp["choices"][0]
                msg = ch.get("message", {})
                usage = resp.get("usage", {})
                ok, why = score(case, msg)
                if ch.get("finish_reason") == "length":
                    ok, why = False, "output exhausted: " + why
                last = {
                    "ok": ok,
                    "why": why[:1500],
                    "finish": ch.get("finish_reason"),
                    "usage": usage,
                    "content_head": (msg.get("content") or "")[:600],
                    "response": resp,
                    "seed": payload["seed"],
                    "effort": A.effort,
                    "tool_calls": msg.get("tool_calls"),
                }
                if ok:
                    break
            except Exception as e:
                last = {"ok": False, "why": f"ERROR {type(e).__name__}: {e}"[:1000]}
        rec.update(last or {"ok": False, "why": "no attempt"})
        rec["wall_s"] = round(time.time() - t0, 1)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(
            f"{cid} {'PASS' if rec['ok'] else 'FAIL'} {rec['wall_s']}s {rec.get('why', '')[:120]}",
            flush=True,
        )

# summary
recs = [json.loads(l) for l in open(A.out)]
by = {}
for r in recs:
    k = r["kind"]
    by.setdefault(k, [0, 0])
    by[k][0] += r["ok"]
    by[k][1] += 1
print("SUMMARY", json.dumps({k: f"{p}/{t}" for k, (p, t) in by.items()}))
print("TOTAL", sum(r["ok"] for r in recs), "/", len(recs))
