"""성적 진단 · 처방 순서 — 466농장 실측 분포 대비.

`farm_gap`(격차 → PSY 회수량 → 원/년) 과 `psy_priority`(처방 순서) 를 부른다.
여기도 산술은 없다.

## 이 축은 무엇을 주장하지 않는가

**개입 효과가 아니다.** "중앙값으로 되돌리면 PSY 가 얼마 회수되는가" 라는
격차의 분해이지 "고치면 오른다" 의 추정이 아니다. 실농장 개입 실험은
수행하지 않았고, 응답의 `footer` 가 그 문장을 항상 들고 다닌다.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

import farm_gap as fg                                          # noqa: E402
import psy_priority as pp                                      # noqa: E402

from .. import db
from ..schemas import Performance
from .farms import get_con

router = APIRouter(prefix="/api/diagnosis", tags=["diagnosis"])


def _farm_metrics(p: Performance) -> dict:
    """**비운 것은 넣지 않는다.**

    빈 칸을 중앙값으로 채우면 그 지표의 격차가 늘 0 으로 찍힌다 — 중앙값을
    중앙값과 비교하게 되기 때문이다. 실제로 겪은 버그라 등록 화면·CLI·여기
    셋 다 같은 규칙을 지킨다.
    """
    return {k: v for k, v in
            (("npd", p.npd), ("weaned", p.weaned),
             ("farrowing_rate", p.farrowing_rate),
             ("wean_to_estrus", p.wean_to_estrus))
            if v is not None}


def build(perf: Performance, n_sows: int, name: str | None = None) -> dict:
    farm = _farm_metrics(perf)
    if not farm:
        raise HTTPException(422, "성적이 하나도 없다 — 비운 칸을 중앙값으로 "
                                 "채우지 않으므로 진단할 것이 없다")
    return {
        "given": sorted(farm),
        "diagnosis": fg.diagnose(farm, n_sows=n_sows),
        "priority": pp.build(farm, n_sows, name),
        "quantiles": fg.load_stats().get("quantiles", {}),
    }


@router.post("", summary="성적 → 466농장 분포 대비 격차 · 처방 순서")
def diagnose(perf: Performance, sows: int = 300,
             name: str | None = None) -> dict:
    if not 1 <= sows <= 20000:
        raise HTTPException(422, "모돈 두수는 1~20000")
    return build(perf, sows, name)


@router.get("/farm/{farm_id}", summary="등록한 농장의 진단 · 처방")
def of_farm(farm_id: int, con=Depends(get_con)) -> dict:
    from ..schemas import FarmSetup

    f = db.get_farm(con, farm_id)
    if not f:
        raise HTTPException(404, f"농장 {farm_id} 없음")
    setup = FarmSetup(**f["setup"])
    # 모돈 두수는 **등록한 돈사가 정한 규모**를 우선 쓴다. 그게 이 농장의
    # 실제 규모이고, 원/년 환산이 규모에 비례하기 때문이다.
    n_sows = setup.n_sows or 300
    return {"farm": {"id": f["id"], "name": f["name"]},
            **build(setup.performance, n_sows, f["name"])}
