#!/usr/bin/env python3
"""Grok(xAI) 잎 분류기용 로컬 프록시 서버 (파이썬 표준 라이브러리만 사용).

브라우저에서 xAI API를 직접 부르면 CORS로 막히므로, 같은 컴퓨터에서 도는
이 작은 서버가 요청을 대신 중계한다. 추가 설치(pip) 없이 파이썬만 있으면 된다.

실행:
    python proxy.py
그다음 브라우저에서 http://127.0.0.1:8000 접속.

API 키 두 가지 방법 (하나만):
  1) 환경변수(권장, 키가 브라우저에 안 남음):
        macOS/Linux:  export XAI_API_KEY="xai-..."
        Windows PS :  $env:XAI_API_KEY = "xai-..."
  2) 웹 화면의 '키 저장' 칸에 입력 → 그 키가 이 프록시로 전달됨.

포트를 바꾸려면:  PORT=9000 python proxy.py
"""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
XAI_BASE = "https://api.x.ai/v1"
BASE_DIR = Path(__file__).parent
ENV_KEY = os.environ.get("XAI_API_KEY", "").strip()


def _auth_for(handler: "Handler") -> str:
    """사용할 Authorization 헤더 값. 서버 env 키 우선, 없으면 브라우저가 보낸 값."""
    if ENV_KEY:
        return f"Bearer {ENV_KEY}"
    return handler.headers.get("Authorization", "")


def _relay(handler: "Handler", method: str, path: str, body: bytes | None = None) -> None:
    """요청을 xAI로 그대로 전달하고 응답을 되돌려준다."""
    auth = _auth_for(handler)
    if not auth:
        handler.send_json(
            401,
            {"error": {"message": "API 키가 없습니다. XAI_API_KEY 환경변수를 설정하거나 웹에서 키를 저장하세요."}},
        )
        return

    req = urllib.request.Request(f"{XAI_BASE}{path}", data=body, method=method)
    req.add_header("Authorization", auth)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            handler.send_raw(resp.status, resp.read())
    except urllib.error.HTTPError as e:
        # xAI가 준 상태코드/본문을 그대로 전달
        handler.send_raw(e.code, e.read())
    except Exception as e:  # 네트워크 오류 등
        handler.send_json(502, {"error": {"message": f"xAI 연결 실패: {e}"}})


class Handler(BaseHTTPRequestHandler):
    def send_raw(self, status: int, data: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status: int, obj: dict) -> None:
        self.send_raw(status, json.dumps(obj).encode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = (BASE_DIR / "index.html").read_bytes()
            self.send_raw(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/models":
            _relay(self, "GET", "/models")
        else:
            self.send_raw(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None
            _relay(self, "POST", "/chat/completions", body)
        else:
            self.send_raw(404, b'{"error":"not found"}')

    def log_message(self, *args) -> None:  # 콘솔을 조용하게
        pass


if __name__ == "__main__":
    print(f"🌱 잎 분류 프록시 실행 중 → http://{HOST}:{PORT}")
    if ENV_KEY:
        print("   XAI_API_KEY 환경변수를 사용합니다 (웹에서 키 입력 불필요).")
    else:
        print("   웹 화면에서 xAI 키를 입력하세요. (또는 XAI_API_KEY 환경변수 설정)")
    print("   종료하려면 Ctrl + C")
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n종료했습니다.")
