#!/usr/bin/env python3
"""AI 패션 스타일리스트 — 네이버 쇼핑 검색 API 로컬 프록시.

네이버 오픈API는 CORS 를 지원하지 않고(preflight 405, ACAO 헤더 없음)
Client Secret 이 필요하므로 브라우저에서 직접 호출할 수 없다.
이 스크립트가 키를 들고 대신 호출해준다.

실행:
    export NAVER_CLIENT_ID=...
    export NAVER_CLIENT_SECRET=...
    python3 stylist/proxy.py            # http://127.0.0.1:8765

또는 키를 stylist/.naver_keys.json 에 넣어도 된다(깃 제외됨):
    {"client_id": "...", "client_secret": "..."}

※ 이 docstring 은 설명일 뿐이다. 여기에 키를 적어도 적용되지 않으며(아래
  load_keys 가 환경변수·키파일에서만 읽는다), 깃에 커밋되므로 적지 마라.
  Client ID 와 Client Secret 은 서로 다른 값이다(ID 20자, Secret 10자 내외).

키가 없어도 서버는 뜬다. 검색만 비활성이고 앱은 정상 동작한다.
표준 라이브러리만 쓴다.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYFILE = HERE / ".naver_keys.json"
NAVER_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"
DEFAULT_PORT = 8765


def load_keys():
    """환경변수 우선, 없으면 .naver_keys.json. 둘 다 없으면 (None, None)."""
    cid = os.environ.get("NAVER_CLIENT_ID")
    sec = os.environ.get("NAVER_CLIENT_SECRET")
    if cid and sec:
        return cid, sec
    if KEYFILE.exists():
        try:
            d = json.loads(KEYFILE.read_text(encoding="utf-8"))
            cid = d.get("client_id") or cid
            sec = d.get("client_secret") or sec
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] {KEYFILE.name} 을 읽지 못했습니다: {e}", file=sys.stderr)
    return cid, sec


CLIENT_ID, CLIENT_SECRET = load_keys()


class Handler(SimpleHTTPRequestHandler):
    """stylist/ 의 정적 파일 + /api/* 프록시."""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    # --- 응답 헬퍼 -------------------------------------------------
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # --- 라우팅 ---------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            return self._json(200, {"ok": True, "naver": bool(CLIENT_ID and CLIENT_SECRET)})
        if parsed.path == "/api/shop":
            return self.handle_shop(urllib.parse.parse_qs(parsed.query))
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def handle_shop(self, qs):
        if not (CLIENT_ID and CLIENT_SECRET):
            return self._json(503, {
                "error": "no_keys",
                "message": "네이버 API 키가 설정되지 않았습니다. "
                           "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수 또는 "
                           "stylist/.naver_keys.json 을 설정한 뒤 프록시를 다시 실행하세요.",
            })

        query = (qs.get("query") or [""])[0].strip()
        if not query:
            return self._json(400, {"error": "no_query", "message": "검색어가 비어 있습니다."})

        try:
            display = max(1, min(30, int((qs.get("display") or ["12"])[0])))
        except ValueError:
            display = 12

        params = urllib.parse.urlencode({"query": query, "display": display, "sort": "sim"})
        req = urllib.request.Request(
            f"{NAVER_ENDPOINT}?{params}",
            headers={
                "X-Naver-Client-Id": CLIENT_ID,
                "X-Naver-Client-Secret": CLIENT_SECRET,
                "User-Agent": "stylist-proxy/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("errorMessage", "")
            except Exception:
                pass
            # 네이버의 에러 메시지를 그대로 노출하되 키는 절대 싣지 않는다.
            return self._json(502, {
                "error": "naver_http",
                "status": e.code,
                "message": detail or f"네이버 API 오류 ({e.code}). 키와 API 권한 설정을 확인하세요.",
            })
        except urllib.error.URLError as e:
            return self._json(502, {"error": "network", "message": f"네이버 API 에 연결하지 못했습니다: {e.reason}"})

        # 앱이 쓰는 필드만 추려서 전달
        items = [{
            "title": it.get("title", ""),
            "image": it.get("image", ""),
            "link": it.get("link", ""),
            "lprice": it.get("lprice", ""),
            "brand": it.get("brand", "") or it.get("maker", "") or it.get("mallName", ""),
            "category": [it.get(f"category{i}", "") for i in (1, 2, 3, 4)],
        } for it in data.get("items", [])]
        return self._json(200, {"total": data.get("total", 0), "items": items})

    def log_message(self, fmt, *args):
        # 정적 파일 요청 로그는 시끄러워서 API 만 남긴다.
        if self.path.startswith("/api/"):
            path = self.path.split("?")[0]
            sys.stderr.write(f"[proxy] {path} {fmt % args}\n")


def main():
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"포트 번호가 올바르지 않습니다: {sys.argv[1]}", file=sys.stderr)
            return 1

    if CLIENT_ID and CLIENT_SECRET:
        print("[proxy] 네이버 API 키 확인됨 — 상품 검색 사용 가능")
    else:
        print("[proxy] 네이버 API 키 없음 — 상품 검색은 비활성 (수동 등록은 정상 동작)")
        print("[proxy] 키 설정: NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수 또는 stylist/.naver_keys.json")

    # 로컬 전용 바인딩. 외부에 키를 노출하지 않는다.
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"[proxy] 포트 {port} 를 열지 못했습니다: {e}", file=sys.stderr)
        print(f"[proxy] 다른 포트로: python3 {Path(__file__).name} 8766", file=sys.stderr)
        return 1

    print(f"[proxy] http://127.0.0.1:{port} 에서 앱을 여세요 (Ctrl+C 로 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[proxy] 종료")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
