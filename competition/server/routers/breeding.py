"""번식 운영 — 한 날짜 → 전 일정 · 교배 적기 · 임신진단 3단계.

`repro_calendar` · `breeding_timing` · `pregnancy_check` 를 부른다.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

import breeding_timing as bt                                   # noqa: E402
import pregnancy_check as pc                                   # noqa: E402
import repro_calendar as rc                                    # noqa: E402

from ..schemas import DetectionIn, WeaningIn

router = APIRouter(prefix="/api/breeding", tags=["breeding"])


@router.post("/schedule", summary="이유일 하나 → 교배·임신감정·분만·이유 전 일정")
def schedule(body: WeaningIn) -> dict:
    try:
        tasks = rc.schedule_from_weaning(
            body.weaning_date, parity=body.parity, season_hot=body.season_hot)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"날짜를 읽지 못했다: {e}") from e
    return {"tasks": tasks, "summary": rc.cycle_summary(tasks),
            "expected_wei": rc.expected_wei(body.parity, body.season_hot)}


@router.get("/checkpoints", summary="교배일 → 임신진단 3단계 체크포인트")
def checkpoints(service_date: str = Query(..., description="교배일 YYYY-MM-DD")) -> dict:
    try:
        tasks = pc.checkpoint_tasks(service_date, estimated=True)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"날짜를 읽지 못했다: {e}") from e
    return {"tasks": tasks, "cascade": pc.detection_cascade(),
            "npd_from_returns": pc.npd_from_returns()}


@router.post("/timing", summary="발정 점검 주기 → 그 주기의 최적 주입 시점")
def timing(body: DetectionIn) -> dict:
    """**주기마다 최적 시점을 다시 찾는다** — `detection_value` 가 그 함수다.

    원시 `timing_under_detection` 을 고정 오프셋(12/24h)으로 부르면 안 된다.
    시점을 고정한 채 주기만 늘리면 비교가 불공정해지고, 실제로 그렇게 했다가
    **하루 2회가 연속 관찰보다 높게** 나왔다. 하루 1회 점검하는 농장은 발견이
    늦다는 걸 알고 더 이르게 주입한다 — 그걸 반영해야 남는 차이가
    **불확실성 자체의 비용**이 된다.
    """
    return bt.detection_value(body.check_interval_h, parity=body.parity)


# `best_offsets_for_interval` 은 19×19 격자를 24스텝씩 훑어 3.5초 걸린다.
# 상수만으로 정해지는 결정론적 값이라 한 번 재면 끝이다 — 캐시가 답을
# 바꾸지 않는다(입력이 같으면 늘 같은 값).
@lru_cache(maxsize=8)
def _detection(parity: str) -> tuple:
    rows = tuple({"label": lb, **bt.detection_value(h, parity=parity)}
                 for lb, h in (("연속 관찰 (CCTV)", 1.0), ("하루 2회", 12.0),
                               ("하루 1회", 24.0)))
    base = rows[0]["conception"]
    for r in rows:
        r["vs_continuous_pp"] = round((r["conception"] - base) * 100, 1)
    return rows


@router.get("/detection", summary="점검 주기별 비교 (연속 · 하루2회 · 하루1회)")
def detection(parity: str = "sow") -> dict:
    """세 주기를 **각각 그 주기의 최적 프로토콜**로 비교한다."""
    if parity not in ("sow", "primiparous"):
        raise HTTPException(422, "parity 는 sow 또는 primiparous")
    return {"rows": list(_detection(parity)), "parity": parity}


@router.get("/today", summary="오늘 할 일 — 이유일 목록에서 뽑는다")
def today(weaning: list[str] = Query(..., description="이유일 여러 개"),
          on: str | None = None, horizon: int = 0) -> dict:
    """등록된 배치들의 이유일을 주면 오늘(또는 지정일) 작업 큐를 만든다."""
    try:
        ids = [f"B{i + 1}" for i in range(len(weaning))]
        scheds = {pid: rc.schedule_from_weaning(w)
                  for pid, w in zip(ids, weaning)}
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"날짜를 읽지 못했다: {e}") from e
    day = on or date.today().isoformat()
    return {"on": day,
            "due": rc.due_today(scheds, today=day, horizon=horizon),
            "overdue": rc.overdue(scheds, today=day),
            "horizon_days": horizon}


@router.get("/batches", summary="배치 간격에서 다음 이유일들을 생성 (데모용)")
def batches(first_weaning: str, interval_days: float = 21, n: int = 7) -> dict:
    """데모 화면이 배치 여러 개를 한 번에 보여줄 수 있게 하는 보조 엔드포인트.

    **실제 이력이 아니다** — 간격을 그대로 더한 값이라 유도값이다.
    """
    try:
        d0 = date.fromisoformat(first_weaning)
    except ValueError as e:
        raise HTTPException(422, f"날짜를 읽지 못했다: {e}") from e
    days = [(d0 + timedelta(days=round(interval_days * i))).isoformat()
            for i in range(max(1, min(n, 24)))]
    return {"weaning_dates": days, "grade": "유도 — 실제 이력이 아니다"}
