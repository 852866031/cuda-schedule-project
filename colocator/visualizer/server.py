#!/usr/bin/env python3
"""Replay web server — stdlib only, no dependencies.

Serves the exported replays/ directory (index.html + one page per run) and a
tiny API:

    GET /                -> replays/index.html
    GET /<run>.html      -> that run's replay
    GET /api/runs        -> JSON list of exported replay names

Run:  python3 server.py [port] [--replays DIR]     (default 8000, ../../replays)

Re-export at any time (`python colocator/visualizer/export_all.py`) — pages
are read from disk per request, so a refresh picks up new exports; no restart
needed.
"""

import http.server
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPLAYS = os.path.normpath(os.path.join(HERE, "..", "..", "replays"))


def make_handler(replays_dir):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=replays_dir, **kw)

        def do_GET(self):
            # Friendly page instead of a bare directory listing when the
            # replays haven't been exported (yet) — e.g. mid re-run.
            if self.path in ("/", "/index.html") and \
                    not os.path.exists(os.path.join(replays_dir, "index.html")):
                body = (b"<body style='background:#0d0d0d;color:#c3c2b7;"
                        b"font:14px system-ui;padding:40px'>"
                        b"<h2 style='color:#fff'>No replays exported yet</h2>"
                        b"<p>The replays directory is empty &mdash; a run/export may be "
                        b"in progress. Run:</p>"
                        b"<pre>python colocator/demo/run_pairs.py\n"
                        b"python colocator/visualizer/export_all.py</pre>"
                        b"<p>then refresh this page.</p></body>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/runs":
                runs = sorted(f[:-5] for f in os.listdir(replays_dir)
                              if f.endswith(".html") and f != "index.html")
                body = json.dumps(runs).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, fmt, *args):  # quieter logs
            sys.stderr.write("[server] %s\n" % (fmt % args))

    return Handler


def main():
    args = [a for a in sys.argv[1:]]
    replays = DEFAULT_REPLAYS
    if "--replays" in args:
        i = args.index("--replays")
        replays = os.path.abspath(args[i + 1])
        del args[i:i + 2]
    port = int(args[0]) if args else 8000

    if not os.path.isdir(replays):
        sys.exit(f"replays dir not found: {replays} — run "
                 "colocator/visualizer/export_all.py first")
    print(f"[server] serving {replays} on http://localhost:{port}")
    http.server.ThreadingHTTPServer(("", port), make_handler(replays)).serve_forever()


if __name__ == "__main__":
    main()
