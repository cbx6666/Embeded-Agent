"""本地 HTTP 桌宠预览（供 Cursor 端口转发 / 内置浏览器打开，无需 VNC）。"""

from __future__ import annotations

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_PREVIEW_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>Embeded-Agent 桌宠预览</title>
  <style>
    body {{ margin: 0; background: #1e1e28; color: #eee; font-family: system-ui, sans-serif; }}
    main {{ max-width: 720px; margin: 0 auto; padding: 16px; text-align: center; }}
    img {{ max-width: 100%; border-radius: 12px; background: #111; }}
    .meta {{ opacity: 0.75; font-size: 14px; margin-top: 12px; }}
  </style>
</head>
<body>
  <main>
    <h1>桌宠预览</h1>
    <img id="pet" src="/pet.png" alt="pet"/>
    <p class="meta" id="meta">刷新中…</p>
  </main>
  <script>
    const img = document.getElementById('pet');
    const meta = document.getElementById('meta');
    function tick() {{
      const t = Date.now();
      img.src = '/pet.png?t=' + t;
      meta.textContent = '最近刷新: ' + new Date().toLocaleTimeString();
    }}
    setInterval(tick, 500);
    tick();
  </script>
</body>
</html>
"""


class PetPreviewServer:
    """在 127.0.0.1 提供 PNG 与自动刷新的预览页。"""

    def __init__(
        self,
        *,
        port: int = 8765,
        png_path: str | Path = "data/runtime/pet_preview.png",
    ) -> None:
        self.port = int(port)
        self.png_path = Path(png_path)
        self._lock = threading.Lock()
        self._png: bytes = b""
        self._state_label: str = "idle"
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def set_frame(self, png_bytes: bytes, *, state_label: str = "") -> None:
        with self._lock:
            self._png = png_bytes
            if state_label:
                self._state_label = state_label
            self.png_path.parent.mkdir(parents=True, exist_ok=True)
            self.png_path.write_bytes(png_bytes)

    def get_png(self) -> bytes:
        with self._lock:
            return self._png

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path in {"/", "/index.html"}:
                    body = _PREVIEW_HTML.format().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/pet.png":
                    data = server_ref.get_png()
                    if not data:
                        self.send_error(HTTPStatus.NOT_FOUND, "no frame yet")
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path == "/state":
                    body = server_ref._state_label.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="PetPreviewServer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
