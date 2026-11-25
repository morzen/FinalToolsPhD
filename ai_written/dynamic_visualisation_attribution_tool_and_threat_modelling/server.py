#!/usr/bin/env python3
"""
Dynamic visualisation attribution tool and threat modelling: a local web page over the thesis tools
(FinalToolsPhD_6feat / _3feat), read only.

    bash run_explorer.sh            (six-feature tool; opens http://127.0.0.1:8765)
    bash run_explorer.sh 3f         (three-feature tool)

or directly, with the tool's own Python:
    ../../FinalToolsPhD_6feat/venv/bin/python server.py --tool ../../FinalToolsPhD_6feat

The page sends the evidence you type to this server, which runs the eight approaches of the tool (their own
hybrid_query) and the WRRF hybrid (common/protocol.py), and returns the rankings and what each approach
computed on the way.  The attack generator (attack.py, with tm_data.py) builds a possible attack
of a chosen group from its profile, with the tactics and the items (or CTI report) the user chooses.  Nothing is written into the tool folders; names.json (display names read from
Cyber_-Necronomicon) is written next to this file the first time.
"""
import sys
sys.dont_write_bytecode = True                         # before anything from the tools is imported

import os                                              # noqa: E402
import json                                            # noqa: E402
import argparse                                        # noqa: E402
import threading                                       # noqa: E402
import traceback                                       # noqa: E402
import webbrowser                                      # noqa: E402
from urllib.parse import urlparse, parse_qs            # noqa: E402
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, 'static')
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
         '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.json': 'application/json'}
ENGINE = None
ATTACK = {'status': 'not started', 'gen': None}


def load_attack_generator(tool, dumps):
    """The attack generator reads the dataset records and the dumps (about 20 s): loaded in the background."""
    ATTACK['status'] = 'loading'
    try:
        from attack import AttackGenerator
        ATTACK['gen'] = AttackGenerator(tool, dumps, log=lambda *a: None)
        ATTACK['status'] = 'ready'
        print("[explorer] attack generator ready", flush=True)
    except Exception as exc:                           # noqa: BLE001
        ATTACK['status'] = f"unavailable: {type(exc).__name__}: {exc}"
        print(f"[explorer] attack generator {ATTACK['status']}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):                 # keep the terminal quiet
        pass

    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == '/api/meta':
                meta = ENGINE.meta()
                meta['attack_status'] = ATTACK['status']
                return self._send(200, meta)
            if url.path == '/api/attack/status':
                return self._send(200, {'status': ATTACK['status']})
            if url.path == '/api/attack/groups':
                if ATTACK['gen'] is None:
                    return self._send(503, {'error': ATTACK['status']})
                return self._send(200, {'groups': ATTACK['gen'].groups()})
            if url.path == '/api/attack':
                if ATTACK['gen'] is None:
                    return self._send(503, {'error': ATTACK['status']})
                q = parse_qs(url.query)
                return self._send(200, ATTACK['gen'].generate(q['group'][0], int(q.get('seed', ['0'])[0]),
                                                              q.get('platform', ['any'])[0],
                                                              q.get('objective', ['both'])[0]))
            if url.path == '/api/profile':
                q = parse_qs(url.query)
                ev = ENGINE.profile_evidence(q['group'][0], q.get('mode', ['full'])[0],
                                             q.get('imitated', [None])[0])
                return self._send(200, {'evidence': ev})
            path = 'studio.html' if url.path in ('/', '') else url.path.lstrip('/')
            full = os.path.realpath(os.path.join(STATIC, path))
            if not full.startswith(os.path.realpath(STATIC)) or not os.path.isfile(full):
                return self._send(404, {'error': 'not found'})
            with open(full, 'rb') as fh:
                return self._send(200, fh.read(), TYPES.get(os.path.splitext(full)[1], 'application/octet-stream'))
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {'error': f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            n = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(n) or b'{}')
            if url.path == '/api/attribute':
                return self._send(200, ENGINE.attribute(body.get('evidence', {})))
            if url.path == '/api/attack':
                if ATTACK['gen'] is None:
                    return self._send(503, {'error': ATTACK['status']})
                return self._send(200, ATTACK['gen'].generate(body['group'], int(body.get('seed', 0)),
                                                              body.get('platform', 'any'),
                                                              body.get('objective', 'both'),
                                                              tactics=body.get('tactics'),
                                                              include=body.get('include')))
            return self._send(404, {'error': 'not found'})
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {'error': f"{type(exc).__name__}: {exc}"})


def main():
    global ENGINE
    parent = os.path.dirname(os.path.dirname(HERE))            # claudePhDFolder
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool', default='6f', help="6f, 3f or the path of a FinalToolsPhD_* folder")
    ap.add_argument('--dumps', default=os.path.join(parent, 'Cyber_-Necronomicon'),
                    help='folder of the MongoDB dumps (display names only)')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--methods', default=None, help='comma-separated subset (testing only)')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--no-attack', action='store_true', help='do not load the attack generator')
    args = ap.parse_args()
    tool = {'6f': os.path.join(parent, 'FinalToolsPhD_6feat'),
            '3f': os.path.join(parent, 'FinalToolsPhD_3feat')}.get(args.tool, args.tool)
    sys.path.insert(0, HERE)
    from engine import Engine
    print(f"[explorer] loading the approaches of {tool} ...", flush=True)
    ENGINE = Engine(tool, dump_dir=args.dumps,
                    methods=args.methods.split(',') if args.methods else None)
    if not args.no_attack:
        threading.Thread(target=load_attack_generator, args=(tool, args.dumps), daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/"
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f"[explorer] ready: {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[explorer] stopped")


if __name__ == '__main__':
    main()
