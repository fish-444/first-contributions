"""내보내기 — 화면에서 본 표를 CSV 로 가져간다.

산술도 서식도 여기서 만들지 않는다. 각 표는 이 서버가 이미 쓰는 함수를
그대로 부르고(`capacity.compute` · `season.compute` …), CSV 로 펴는 일은
`table_export` 가 한다. 그래서 화면에서 본 수와 내려받은 수가 같다.

## 등급과 각주가 파일을 떠나지 않게 한다

CSV 는 서식이 없어서 화면의 배지와 각주가 통째로 사라진다. 그러면 격차
분해가 개입 효과처럼, 유도값이 실측처럼 읽힌다 — 이 프로젝트가 가장 조심해
온 오독이다. `table_export` 가 머리말(`#`)과 등급 열을 강제하므로 여기서
그 둘을 끄는 길을 열어 두지 않는다(`bare` 는 머리말만 뺀다).

## 파일 이름에 농장 이름을 넣지 않는다

농장 이름은 식별자다. 내려받은 파일이 메일·메신저로 굴러다니는 걸 막을 수
없으므로, 이름은 `yangdon_<표>_<날짜>.csv` 로 고정한다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

import farm_gap as fg                                          # noqa: E402
import psy_priority as pp                                      # noqa: E402
import table_export as tx                                      # noqa: E402
import vision_contract as vc                                   # noqa: E402

from ..schemas import ExportIn
from . import capacity as capr
from . import diagnosis as diagr
from . import season as seasonr

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/sheets", summary="내려받을 수 있는 표")
def sheets() -> dict:
    return {
        "sheets": [
            {"key": "capacity", "kr": "돈사별 지지 두수·병목", "needs": ["setup"]},
            {"key": "interval", "kr": "간격 what-if", "needs": ["setup"]},
            {"key": "diagnosis", "kr": "466농장 대비 격차", "needs": ["performance"]},
            {"key": "priority", "kr": "처방 순서", "needs": ["performance"]},
            {"key": "season", "kr": "여름 손실 분포", "needs": []},
            {"key": "targets", "kr": "오늘의 영상 겨냥", "needs": ["herd"]},
        ],
        "note": ("모든 파일에 **등급 열과 각주 머리말**이 붙는다. CSV 는 "
                 "서식이 없어 화면의 배지·각주가 사라지는데, 그러면 격차 "
                 "분해가 개입 효과처럼 읽힌다."),
    }


@router.post("/{sheet}", summary="표 → CSV (등급 열·각주 머리말 포함)",
             response_class=Response)
def export(sheet: str, body: ExportIn,
           bare: bool = Query(False, description="머리말 없이(기계용). "
                                                 "등급 열은 못 뺀다")) -> Response:
    if sheet not in tx.SHEETS:
        raise HTTPException(404, f"알 수 없는 표: {sheet} "
                                 f"(가능: {', '.join(tx.SHEETS)})")
    on = None

    if sheet in ("capacity", "interval"):
        if not (body.setup and body.setup.barns):
            raise HTTPException(422, "돈사가 등록되지 않았다")
        payload = (capr.compute(body.setup) if sheet == "capacity"
                   else capr.interval_whatif(body.setup))
    elif sheet in ("diagnosis", "priority"):
        perf = body.performance or (body.setup.performance if body.setup
                                    else None)
        if perf is None:
            raise HTTPException(422, "성적이 없다")
        # 비운 칸을 중앙값으로 채우지 않는다 — 진단 라우터와 같은 규칙이다
        farm = diagr._farm_metrics(perf)
        if not farm:
            raise HTTPException(422, "성적이 하나도 없다 — 비운 칸을 "
                                     "중앙값으로 채우지 않으므로 낼 표가 없다")
        n = body.sows or (body.setup.n_sows if body.setup else None) or 300
        payload = {"diagnosis": fg.diagnose(dict(farm), n_sows=n),
                   "priority": pp.build(dict(farm), n, None)}
    elif sheet == "season":
        payload = seasonr.compute(body.sows or 300)
    else:                                   # targets
        if not body.herd or not body.herd.records:
            raise HTTPException(422, "개체 이력이 없다 — 겨냥할 대상이 없다")
        payload = vc.targets(body.herd.records, body.herd.as_of,
                             disease_all=body.herd.include_disease)
        on = body.herd.as_of

    csv_text = tx.build(sheet, payload, bare=bare)
    name = tx.filename(sheet, on)
    return Response(
        csv_text, media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'})
