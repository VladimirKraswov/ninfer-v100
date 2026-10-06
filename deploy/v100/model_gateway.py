#!/usr/bin/env python3
"""Single-resident NInfer supervisor. Python owns HTTP/lifecycle, never inference.

A trusted registry supplies explicit child argv. Selection is asynchronous, drains
requests including streams, reaps the old process before spawning another and
publishes real startup events. No automatic request-driven model substitution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hmac
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

PHASES = ("engine", "cuda", "artifact", "planning", "weights", "staging",
          "target", "frontend", "runtime", "host_state", "host_kv", "graphs", "finalize")
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length"}


class Refusal(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Manager:
    def __init__(self, registry, token, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.models = registry["models"]
        if not self.models or not token:
            raise ValueError("models and control token are required")
        for model_id, model in self.models.items():
            if not model_id or not isinstance(model.get("argv"), list) or not model["argv"]:
                raise ValueError("each model needs a stable id and explicit argv")
            if not all(isinstance(arg, str) for arg in model["argv"]):
                raise ValueError("argv entries must be strings")
            if not isinstance(model.get("agents"), list) or not model["agents"] or not all(isinstance(agent, str) and agent for agent in model["agents"]):
                raise ValueError("each model needs explicit allowed agents")
        self.port = int(registry.get("backend_port", 18080))
        self.load_timeout = float(registry.get("load_timeout", 900))
        self.drain_timeout = float(registry.get("drain_timeout", 1800))
        self.cv = threading.Condition()
        self.active = 0
        self.proc = None
        self.closed = False
        self.worker = None
        self.state = {"schema": 1, "operation_id": None, "phase": "unloaded", "ready": False,
                      "active_model": None, "target_model": None, "started_at": time.time(),
                      "error": None, "loader": None}

    def snapshot(self):
        with self.cv:
            if self.state["phase"] == "ready" and (not self.proc or self.proc.poll() is not None):
                self.state.update(phase="failed", ready=False, active_model=None, error="Resident engine exited")
            return dict(self.state, active_requests=self.active,
                        elapsed_seconds=round(time.time() - self.state["started_at"], 1))

    def update(self, **changes):
        with self.cv:
            self.state.update(changes)
            self.cv.notify_all()

    def select(self, model_id, agent):
        with self.cv:
            self.snapshot()
            if self.closed:
                raise Refusal(503, "Service is stopping")
            model = self.models.get(model_id)
            if not model:
                raise Refusal(404, "Model is not registered")
            if agent not in model["agents"]:
                raise Refusal(403, "Model is not allowed for this agent")
            if self.state["phase"] not in ("ready", "failed", "unloaded"):
                if self.state["target_model"] == model_id:
                    return self.snapshot()
                raise Refusal(409, "A different model switch is already running")
            if self.state["ready"] and self.state["active_model"] == model_id:
                self.state["error"] = None
                return self.snapshot()
            self.state.update(operation_id=str(uuid.uuid4()), target_model=model_id,
                              phase="draining", ready=False, error=None, loader=None,
                              started_at=time.time())
            self.worker = threading.Thread(target=self._switch, args=(model_id,), daemon=True)
            self.worker.start()
            return self.snapshot()

    @contextmanager
    def lease(self, model_id, agent):
        with self.cv:
            if not self.state["ready"] or not self.proc or self.proc.poll() is not None:
                self.state["ready"] = False
                raise Refusal(503, "Model is not ready; inspect /v1/model-control/status")
            if model_id != self.state["active_model"]:
                raise Refusal(409, "Requested model is not resident; select it explicitly")
            agents = self.models[model_id]["agents"]
            if agent not in agents:
                raise Refusal(403, "Model is not allowed for this agent")
            self.active += 1
        try:
            yield
        finally:
            with self.cv:
                self.active -= 1
                self.cv.notify_all()

    def _reap(self):
        proc = self.proc
        if not proc:
            return
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
        self.proc = None

    def _events(self, fd, operation):
        with os.fdopen(fd, "r") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                    phase = PHASES[event["phase"]]
                    loader = {"phase": phase, "current": event["current"], "total": event["total"],
                              "unit": "bytes" if event["unit"] == 1 else None}
                    with self.cv:
                        # A late event from a previous child cannot undo readiness or a new switch.
                        if self.state["operation_id"] != operation or self.state["phase"] not in ("loading", "warming"):
                            continue
                        self.state["loader"] = loader
                        if phase == "finalize" and event["status"] == 2:
                            self.state["phase"] = "warming"
                        self.cv.notify_all()
                except (ValueError, KeyError, IndexError, TypeError):
                    continue

    def _switch(self, model_id):
        try:
            with self.cv:
                drained = self.cv.wait_for(lambda: self.active == 0 or self.closed,
                                          timeout=self.drain_timeout)
                if self.closed:
                    return
                if not drained:
                    # Preserve the original engine, never abort an owner's stream to switch.
                    self.state.update(phase="ready", ready=True, target_model=self.state["active_model"],
                                      error="Drain timeout; existing model retained")
                    return
            self.update(phase="unloading")
            self._reap()
            self.update(active_model=None, phase="loading")
            model = self.models[model_id]
            read_fd, write_fd = os.pipe()
            os.set_blocking(write_fd, False)
            env = dict(os.environ, **model.get("env", {}), NINFER_STARTUP_FD=str(write_fd))
            # The control credential never enters the model process environment.
            env.pop("NINFER_MODEL_CONTROL_TOKEN", None)
            try:
                with (self.root / "engine.log").open("ab") as log:
                    self.proc = subprocess.Popen(model["argv"], cwd=model.get("cwd"), env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                        pass_fds=(write_fd,))
            except BaseException:
                os.close(read_fd)
                raise
            finally:
                os.close(write_fd)
            threading.Thread(target=self._events, args=(read_fd, self.state["operation_id"]), daemon=True).start()
            deadline = time.monotonic() + self.load_timeout
            while time.monotonic() < deadline:
                if self.closed:
                    self._reap()
                    return
                if self.proc.poll() is not None:
                    raise RuntimeError("Engine exited during loading; inspect engine.log")
                conn = HTTPConnection("127.0.0.1", self.port, timeout=2)
                try:
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    response.read()
                    if response.status == 200:
                        conn.request("GET", "/v1/models")
                        catalog = conn.getresponse()
                        entries = json.loads(catalog.read()).get("data", [])
                        if catalog.status != 200 or not any(m.get("id") == model_id for m in entries):
                            raise RuntimeError("Backend did not confirm the selected model")
                        self.update(active_model=model_id, phase="ready", ready=True, loader=None)
                        return
                except (OSError, HTTPException):
                    pass
                finally:
                    conn.close()
                time.sleep(.5)
            raise TimeoutError("Engine loading timed out")
        except Exception as error:
            self._reap()
            self.update(phase="failed", ready=False, active_model=None, error=str(error))

    def close(self):
        with self.cv:
            self.closed = True
            self.state["ready"] = False
            self.cv.notify_all()
        if self.worker:
            self.worker.join(timeout=self.load_timeout + 100)
        self._reap()


class Server(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass  # Request bodies and credentials never enter an access log.

    def end_headers(self):
        origin = self.headers.get("Origin", "")
        if origin in ("tauri://localhost", "http://tauri.localhost", "https://tauri.localhost") or origin.startswith(("http://127.0.0.1:", "http://localhost:")) and origin.rsplit(":", 1)[-1].isdigit():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self):
        if not self.path.startswith("/v1/model-control/"):
            self.json(404, {"error": {"message": "Unknown route"}})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def control(self):
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(token, self.server.manager.token):
            raise Refusal(401, "Control credential required")

    def do_GET(self):
        try:
            manager = self.server.manager
            if self.path == "/v1/model-control/status":
                self.control()
                self.json(200, manager.snapshot())
            elif self.path == "/v1/model-control/catalog":
                self.control()
                self.json(200, {"schema": 1, "models": [dict(id=key, label=m.get("label", key),
                    agents=m["agents"], context_window=m.get("context_window"))
                    for key, m in manager.models.items()]})
            elif self.path == "/v1/models":
                self.json(200, {"object": "list", "data": [dict(id=key, object="model",
                    owned_by="ninfer", ninfer={"agents": m["agents"], "context_window": m.get("context_window")})
                    for key, m in manager.models.items()]})
            elif self.path == "/health":
                ready = manager.snapshot()["ready"] and manager.proc and manager.proc.poll() is None
                self.json(200 if ready else 503, {"status": "ok" if ready else "not_ready"})
            elif self.path.startswith("/v1/"):
                with manager.lease(manager.snapshot()["active_model"], self.headers.get("X-Ninfer-Agent", "opencode")):
                    self.proxy(None)
            else:
                raise Refusal(404, "Unknown route")
        except Refusal as error:
            self.json(error.status, {"error": {"message": error.message}})
        except (OSError, HTTPException):
            self.close_connection = True

    def do_POST(self):
        try:
            if self.headers.get("Transfer-Encoding"):
                raise Refusal(400, "Chunked request bodies are unsupported")
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 64 * 1024 * 1024:
                raise Refusal(413, "Body size is outside the supported range")
            raw = self.rfile.read(size)
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise Refusal(400, "JSON object required")
            manager = self.server.manager
            if self.path == "/v1/model-control/select":
                self.control()
                result = manager.select(body.get("model"), body.get("agent"))
                self.json(200 if result["ready"] else 202, result)
                return
            if not self.path.startswith("/v1/") or self.path.startswith("/v1/model-control/"):
                raise Refusal(404, "Unknown generation route")
            # Agent identity is explicit in the provider configuration, never inferred from model name.
            agent = self.headers.get("X-Ninfer-Agent", "opencode")
            with manager.lease(body.get("model"), agent):
                self.proxy(raw)
        except Refusal as error:
            self.json(error.status, {"error": {"message": error.message}})
        except (ValueError, TypeError):
            self.json(400, {"error": {"message": "Invalid request"}})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (OSError, HTTPException):
            self.close_connection = True

    def do_DELETE(self):
        try:
            if not self.path.startswith("/v1/responses/"):
                raise Refusal(404, "Unknown route")
            manager = self.server.manager
            with manager.lease(manager.snapshot()["active_model"], self.headers.get("X-Ninfer-Agent", "opencode")):
                self.proxy(None)
        except Refusal as error:
            self.json(error.status, {"error": {"message": error.message}})
        except (OSError, HTTPException):
            self.close_connection = True

    def proxy(self, raw):
        conn = HTTPConnection("127.0.0.1", self.server.manager.port, timeout=1800)
        try:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
            conn.request(self.command, self.path, raw, headers)
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while block := response.read1(16384):
                self.wfile.write(f"{len(block):x}\r\n".encode() + block + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    manager = Manager(registry, os.environ.get("NINFER_MODEL_CONTROL_TOKEN", ""), args.state_dir)
    # Bind before touching the resident engine; a port collision must never unload a model.
    server = Server((args.host, args.port), Handler)
    server.manager = manager
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if registry.get("default_model"):
            model = registry["default_model"]
            manager.select(model, manager.models[model]["agents"][0])
        server.serve_forever()
    finally:
        manager.close()
        server.server_close()


if __name__ == "__main__":
    main()
