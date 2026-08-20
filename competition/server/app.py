"""FastAPI 앱 — API + 프론트 정적 서빙.

    pip install fastapi "uvicorn[standard]"
    python -m uvicorn competition.server.app:app --reload --port 8000

    http://localhost:8000/         프론트
    http://localhost:8000/docs     자동 생성 API 문서(OpenAPI)

## 정적 뷰는 그대로 둔다

`dashboard/*.html` 22개는 **서버 없이도 브라우저로 열면 돈다.** 서버를
붙였다고 그걸 없애지 않는다 — 심사위원이 서버를 못 띄워도 볼 수 있어야
한다. 이 앱은 그 위에 얹는 것이지 대체가 아니다.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(HERE)
# 도메인 모듈은 src/ 에 있고 서로를 평평하게 import 한다(`import batch_flow`).
# 경로를 먼저 넣어야 라우터의 import 가 선다.
for p in (os.path.join(COMP, "src"), COMP):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI                                    # noqa: E402
from fastapi.responses import FileResponse                     # noqa: E402
from fastapi.staticfiles import StaticFiles                    # noqa: E402

from .routers import (breeding, capacity, diagnosis, farms,     # noqa: E402
                      season, vision)

WEB = os.path.join(COMP, "web")
DASH = os.path.join(COMP, "dashboard")

app = FastAPI(
    title="양돈 AI — 돈사·번식 운영 API",
    version="0.1.0",
    description=(
        "기존 도메인 모듈 68개를 HTTP 로 노출한다. **서버는 계산을 다시 "
        "구현하지 않는다** — `src/` 가 정본이고 여기는 얇은 어댑터다.\n\n"
        "수치 등급: 실측 / 계산 / 유도 / 합성. 응답에 섞지 않는다."),
)

app.include_router(farms.router)
app.include_router(capacity.router)
app.include_router(breeding.router)
app.include_router(diagnosis.router)
app.include_router(season.router)
app.include_router(vision.router)


@app.get("/api/health", tags=["meta"], summary="살아 있는가 + 무엇을 물고 있는가")
def health() -> dict:
    import batch_flow as bf
    import farm_economics as fe
    import farm_registry as fr

    return {
        "ok": True,
        "stages": list(fr.BARN_STAGES),
        "housings": list(fr.HOUSING),
        # 프론트가 자기 상수를 갖지 않도록 **여기서 내려 준다.**
        # 화면이 계산을 다시 구현하면 언젠가 갈린다.
        "constants": {
            "farrow_rate_p10": bf.FARROW_RATE_P10,
            "sow_turnover": bf.SOW_TURNOVER,
            "weaned_per_crate": bf.WEANED_PER_CRATE,
            "grow_survival": bf.GROW_SURVIVAL,
            "ceiling": bf.CEILING,
            "downstream_days": bf.DOWNSTREAM_DAYS,
            "margin_per_pig": fe.margin_per_pig()["margin"],
        },
        "note": ("정적 뷰(dashboard/*.html)는 서버 없이도 돈다. "
                 "이 API 는 그 위에 얹은 것이다."),
    }


if os.path.isdir(DASH):
    app.mount("/dashboard", StaticFiles(directory=DASH, html=True),
              name="dashboard")
if os.path.isdir(WEB):
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """자체완결 원칙 — 외부 아이콘을 물어 오지 않는다. 인라인 SVG 로 낸다."""
    from fastapi.responses import Response
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
           '<text y="26" font-size="26">\U0001F416</text></svg>')
    return Response(svg, media_type="image/svg+xml")


@app.get("/", include_in_schema=False)
def index():
    p = os.path.join(WEB, "index.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"detail": "web/index.html 이 없다. /docs 로 API 를 볼 수 있다."}
