import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection

spec = importlib.util.spec_from_file_location("model_gateway", Path(__file__).with_name("model_gateway.py"))
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)

FIXTURE = '''
import os,json,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
port,model=int(sys.argv[1]),sys.argv[2]
fd=int(os.environ['NINFER_STARTUP_FD'])
os.write(fd,b'{"phase":4,"status":1,"unit":1,"current":10,"total":20}\\n')
os.write(fd,b'{"phase":12,"status":2,"unit":0,"current":0,"total":0}\\n')
os.close(fd)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  data=json.dumps({'data':[{'id':model}]} if self.path=='/v1/models' else {'status':'ok'}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_POST(self):
  data=self.rfile.read(int(self.headers['Content-Length']))
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
HTTPServer(('127.0.0.1',port),Handler).serve_forever()
'''


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ModelGatewayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / "fixture.py"
        self.script.write_text(FIXTURE)
        self.port = unused_port()
        registry = {"backend_port": self.port, "drain_timeout": .15, "load_timeout": 5,
                    "models": {m: {"argv": [sys.executable, str(self.script), str(self.port), m],
                                    "agents": ["pi"] if m == "tuned" else ["pi", "opencode"]}
                               for m in ["base", "tuned"]}}
        self.manager = gateway.Manager(registry, "test-token", self.tmp.name)

    def tearDown(self):
        self.manager.close()
        self.tmp.cleanup()

    def ready(self, model):
        self.manager.select(model, "pi")
        with self.manager.cv:
            self.assertTrue(self.manager.cv.wait_for(lambda: self.manager.state["phase"] in ("ready", "failed"), timeout=7))
        self.assertEqual(self.manager.snapshot()["active_model"], model)

    def test_registry_policy_and_unknown_models(self):
        for model, agent, status in [("missing", "pi", 404), ("tuned", "opencode", 403)]:
            with self.assertRaises(gateway.Refusal) as result:
                self.manager.select(model, agent)
            self.assertEqual(result.exception.status, status)
        self.assertIsNone(self.manager.proc)

    def test_switch_reaps_old_and_confirms_new_model(self):
        self.ready("base")
        old = self.manager.proc
        self.ready("tuned")
        self.assertIsNotNone(old.poll())
        self.assertNotEqual(old.pid, self.manager.proc.pid)
        with self.assertRaises(gateway.Refusal):
            with self.manager.lease("base", "pi"): pass
        with self.assertRaises(gateway.Refusal):
            with self.manager.lease("tuned", "opencode"): pass

    def test_busy_drain_timeout_retains_original_without_killing_request(self):
        self.ready("base")
        old = self.manager.proc
        with self.manager.lease("base", "opencode"):
            switch = self.manager.select("tuned", "pi")
            self.assertFalse(switch["ready"])
            same = self.manager.select("tuned", "pi")
            self.assertEqual(switch["operation_id"], same["operation_id"])
            with self.assertRaises(gateway.Refusal) as result:
                self.manager.select("base", "pi")
            self.assertEqual(result.exception.status, 409)
            self.manager.worker.join(timeout=2)
            self.assertEqual(self.manager.snapshot()["active_model"], "base")
            self.assertIsNone(old.poll())
        self.assertEqual(self.manager.active, 0)
        self.ready("tuned")

    def test_waits_for_released_lease_then_unloads(self):
        self.manager.drain_timeout = 5
        self.ready("base")
        old = self.manager.proc
        with self.manager.lease("base", "opencode"):
            self.manager.select("tuned", "pi")
            time.sleep(.03)
            self.assertEqual(self.manager.snapshot()["phase"], "draining")
            self.assertIsNone(old.poll())
        self.manager.worker.join(timeout=7)
        self.assertEqual(self.manager.snapshot()["active_model"], "tuned")
        self.assertIsNotNone(old.poll())

    def test_load_failure_is_visible_and_retry_recovers(self):
        self.manager.models["tuned"]["argv"] = [sys.executable, "-c", "raise SystemExit(3)"]
        self.manager.select("tuned", "pi")
        self.manager.worker.join(timeout=7)
        self.assertEqual(self.manager.snapshot()["phase"], "failed")
        self.assertFalse(self.manager.snapshot()["ready"])
        self.assertIsNone(self.manager.proc)
        self.ready("base")
        self.manager.proc.kill()
        self.manager.proc.wait()
        self.assertEqual(self.manager.snapshot()["phase"], "failed")
        self.ready("base")

    def test_late_progress_cannot_change_ready_or_new_operation(self):
        import os
        self.ready("base")
        operation = self.manager.snapshot()["operation_id"]
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b'{"phase":12,"status":2,"unit":0,"current":0,"total":0}\n')
        os.close(write_fd)
        self.manager._events(read_fd, operation)
        self.assertEqual(self.manager.snapshot()["phase"], "ready")
        self.manager.update(operation_id="new", phase="loading", loader=None)
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b'{"phase":4,"status":1,"unit":1,"current":10,"total":20}\n')
        os.close(write_fd)
        self.manager._events(read_fd, operation)
        self.assertIsNone(self.manager.snapshot()["loader"])

    def test_http_contract_control_auth_agent_gate_and_stream_proxy(self):
        self.ready("tuned")
        server = gateway.Server(("127.0.0.1", 0), gateway.Handler)
        server.manager = self.manager
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection(*server.server_address, timeout=5)
            conn.request("GET", "/v1/model-control/status")
            response = conn.getresponse(); self.assertEqual(response.status, 401); response.read()
            conn.request("GET", "/v1/model-control/catalog", headers={"Authorization": "Bearer test-token"})
            response = conn.getresponse(); self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["models"][1]["agents"], ["pi"])
            conn.request("POST", "/v1/chat/completions", json.dumps({"model": "tuned"}))
            response = conn.getresponse(); self.assertEqual(response.status, 403); response.read()
            conn.request("POST", "/v1/chat/completions", json.dumps({"stream": True}), {"X-Ninfer-Agent": "pi"})
            response = conn.getresponse(); self.assertEqual(response.status, 409); response.read()
            body = json.dumps({"model": "tuned", "stream": True})
            conn.request("POST", "/v1/chat/completions", body, {"X-Ninfer-Agent": "pi", "Content-Type": "application/json"})
            response = conn.getresponse(); self.assertEqual(response.status, 200)
            self.assertEqual(response.read().decode(), body)
            with self.manager.cv:
                self.assertTrue(self.manager.cv.wait_for(lambda: self.manager.active == 0, timeout=1))
            conn.close()
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
