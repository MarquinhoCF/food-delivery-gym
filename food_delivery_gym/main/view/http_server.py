import threading
import http.server
import os
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)
    def log_message(self, *_):
        pass   # silencioso

def start_http_server(port: int = 8080):
    server = http.server.HTTPServer(("localhost", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[HTTP] Visualização em http://localhost:{port}/index.html")
    return server