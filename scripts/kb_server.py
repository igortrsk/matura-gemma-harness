"""Tiny HTTP search service over the offline Wikipedia KB (kb.py). Runs on (GPU machine) next to the index.

The harness (run_exam.py --retrieval) calls it: GET /search?q=...&k=4 -> {"results": [{title, url, text, score}]}.
Local only (127.0.0.1); from (local machine) reach it through the SSH tunnel (-L 8090:127.0.0.1:8090); on exam day the
harness runs on the same machine. Standard library only, besides kb.py (tantivy).

  setsid nohup /workspace/ft-venv/bin/python /workspace/ft/kb_server.py --dir /workspace/kb >/workspace/kb/server.log 2>&1 &
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import parse_qs, urlparse

from kb import KB

LOCK = Lock()


def make_handler(kb):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self.reply(200, {"status": "ok"})
            if url.path != "/search":
                return self.reply(404, {"error": "use /search?q=...&k=4"})
            qs = parse_qs(url.query)
            q, k = qs.get("q", [""])[0], int(qs.get("k", ["4"])[0])
            with LOCK:  # one sqlite connection / searcher shared by all threads
                res = kb.search(q, k)
            self.reply(200, {"query": q, "results": res})

        def reply(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/kb")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(KB(args.dir))).serve_forever()


if __name__ == "__main__":
    main()
