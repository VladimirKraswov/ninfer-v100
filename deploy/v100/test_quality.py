"""HTTP-level checks of the quality runner; no inference model/GPU required."""
import http.server
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest


class QualityRunnerTest(unittest.TestCase):
    def test_scoring_and_budgets(self):
        received = []
        answers = ['{"a":1}', '```json\n{"a":1}\n```', 'Here: {"a":1}', '{"a":1}',
                   "```python\nprint('TESTOK')\ndef f(): return 1\n```",
                   '```python\ndef f(): return 2\n```']

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                index = len(received)
                received.append(request)
                body = json.dumps({'choices': [{'message': {'content': answers[index]},
                    'finish_reason': 'length' if index == 3 else 'stop'}], 'usage': {}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                suite, out = pathlib.Path(tmp) / 'suite.jsonl', pathlib.Path(tmp) / 'result.jsonl'
                cases = [{'case_id': str(i), 'kind': 'instruction',
                    'messages': [{'role': 'user', 'content': 'Only JSON'}], 'max_tokens': 4,
                    'verify': {'mode': 'json_strict', 'expect': {'a': 1}}} for i in range(4)]
                cases.extend({'case_id': str(i), 'kind': 'code',
                    'messages': [{'role': 'user', 'content': 'Implement f() returning 2'}],
                    'verify': {'mode': 'python_exec', 'test': 'assert f() == 2'}} for i in (4, 5))
                suite.write_text('\n'.join(map(json.dumps, cases)))
                command = [sys.executable, str(pathlib.Path(__file__).with_name('quality.py')),
                    '--url', f'http://127.0.0.1:{server.server_port}', '--model', 'fixture',
                    '--suite', str(suite), '--out', str(out), '--max-tokens', '32768',
                    '--thinking-budget', '8192']
                subprocess.run(command, check=True, capture_output=True, timeout=10)
                results = [json.loads(x) for x in out.read_text().splitlines()]
                self.assertEqual([x['ok'] for x in results], [True, False, False, False, False, True])
                self.assertEqual(len(received), 6)  # no hidden retry
                self.assertEqual(len({x['seed'] for x in received}), 6)
                self.assertTrue(all(x['max_tokens'] == 32768 and x['thinking_budget'] == 8192
                                    for x in received))
                subprocess.run(command, check=True, capture_output=True, timeout=10)
                self.assertEqual(len(received), 6)  # resume does not reissue completed cases
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    unittest.main()
