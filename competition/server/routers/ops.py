"""번식 배정·환경 알람·행동 기준선 — **얇은 어댑터.**

세 모듈을 HTTP 로 낸다. 다른 라우터와 같은 규칙이다: **여기 산술이 없다.**
`mating_plan` · `barn_env_control` · `behavior_baseline` 이 정본이고, 이
파일은 요청을 그 함수의 인자로 옮기고 응답을 그대로 돌려준다.

## 왜 이 셋을 한 라우터에 두나

셋 다 **자기 기준선·자기 제약 안에서 순위를 매기는** 일이라서다. 교배는
근친 한도 아래에서, 환경은 자기 이력 편차로, 행동은 방 자신의 기준선으로.
셋 다 판정이 아니라 **"먼저 보라"** 를 낸다.

## 응답이 자백하는 것

- 교배: 근친율은 **입력 혈통에서의 하한**이다(혈통이 얕으면 낮게 나온다)
- 환경: **알람만** 낸다. 제어 지시는 장비 구성에 달려 있어 로그만 보는
  쪽이 정하면 틀린다
- 행동: 컷은 **자기 이력 경보율 역산**이고, 이력이 모자라면 기준선을
  만들지 않는다 — 조용히 기본값으로 채우지 않는다
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import barn_env_control as ec                                  # noqa: E402
import behavior_baseline as bb                                 # noqa: E402
import mating_plan as mp                                       # noqa: E402

from ..schemas import BaselineIn, EnvIn, MatingIn

router = APIRouter(prefix="/api/ops", tags=["ops"])


@router.post("/mating", summary="교배 배정 — 근친 한도 아래 예상인덱스 최대화")
def mating(body: MatingIn) -> dict:
    """농장장 표 그대로. 배정은 모돈별 최고가 아니라 **농장 전체 최적**이다."""
    for name, xs in (("모돈", body.sows), ("웅돈", body.boars)):
        ids = [x.id for x in xs]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise HTTPException(
                400, f"{name} 번호가 중복이다: {', '.join(dup)} — 조용히 "
                     "접으면 한 마리가 계획 밖에서 교배된다")
    sows = {s.id: {"index": s.index, "sire": s.sire, "dam": s.dam,
                   "max_services": None} for s in body.sows}
    boars = {b.id: {"index": b.index, "sire": b.sire, "dam": b.dam,
                    "max_services": b.max_services} for b in body.boars}
    if not sows or not boars:
        raise HTTPException(400, "모돈과 웅돈이 각각 하나 이상 필요하다")
    try:
        return mp.plan(sows, boars, max_f=body.max_f, services=body.services)
    except ValueError as e:                                    # 혈통 순환
        raise HTTPException(400, str(e)) from e


@router.post("/env", summary="돈사 환경 위험 알람 — 지침 위반과 자기 편차")
def env(body: EnvIn) -> dict:
    """마지막 값이 현재, 그 앞이 이력이다. **알람만** 낸다."""
    names = [b.barn for b in body.barns]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise HTTPException(400, f"돈사 이름이 중복이다: {', '.join(dup)} — "
                                 "한쪽 이력이 소리 없이 사라진다")
    log = {b.barn: {k: v for k, v in
                    (("temp_c", b.temp_c), ("nh3_ppm", b.nh3_ppm),
                     ("rh_pct", b.rh_pct), ("h2s_ppm", b.h2s_ppm))
                    if v} for b in body.barns}
    # 센서 없이 day/spot 온도만 준 돈사는 단열 점검 전용으로 받는다 —
    # 둘 다 없을 때만 거절한다
    empty = [b.barn for b in body.barns
             if not log[b.barn] and not b.day_temps and not b.spot_temps]
    if empty:
        raise HTTPException(400, f"센서 값이 없는 돈사: {', '.join(empty)}")
    sensored = {k: v for k, v in log.items() if v}
    guide = None
    if body.guide:
        guide = {"temp": ec.TEMP_GUIDE, "rh": ec.RH_GUIDE,
                 "nh3": ec.NH3_LIMIT, "h2s": ec.H2S_LIMIT}
        guide.update({k: body.guide[k] for k in list(guide)
                      if k in body.guide})
    out = ec.assess(sensored, {b.barn: b.stage for b in body.barns},
                    guide=guide,
                    implantation={b.barn for b in body.barns
                                  if b.implantation})
    out["insulation"] = {b.barn: ec.insulation_alarms(
        day_temps=b.day_temps or None, spot_temps=b.spot_temps or None)
        for b in body.barns}
    return out


@router.post("/baseline", summary="행동 기준선 — 이탈 점수와 경보")
def baseline(body: BaselineIn) -> dict:
    """`history` 는 창별 구성비, `now` 는 현재 창. 겨냥 헤드는 달력이 준다."""
    if not body.history:
        raise HTTPException(400, "기준선을 만들 이력이 없다")
    b = bb.fit(body.history, body.key, classes=tuple(body.classes)
               if body.classes else None)
    out = bb.assess(b, body.now, recent=body.recent or None,
                    heads=tuple(body.heads) if body.heads else None)
    out["baseline"] = {"center": b.center, "spread": b.spread,
                       "cuts": b.cuts, "n_windows": b.n_windows,
                       "min_windows": bb.MIN_WINDOWS,
                       "rate_band": list(bb.RATE_BAND)}
    return out


@router.get("/guide", summary="환경 지침 상수 — 출처와 함께")
def guide() -> dict:
    """농장이 다른 기준을 쓰면 요청에서 바꾼다. 여기 값은 지침이다."""
    return {
        "grade": "지침",
        "temp_c": {k: list(v) for k, v in ec.TEMP_GUIDE.items()},
        "rh_pct": {k: list(v) for k, v in ec.RH_GUIDE.items()},
        "nh3_ppm_limit": ec.NH3_LIMIT, "h2s_ppm_limit": ec.H2S_LIMIT,
        "insulation": {"spot_diff_c": ec.SPOT_DIFF_LIMIT,
                       "daily_range_c": ec.DAILY_RANGE_LIMIT},
        "mating_max_inbreeding": mp.MAX_F_GUIDE,
        "source": ("국립축산과학원 「환절기 돼지 사양관리」·「쾌적한 사육환경을 "
                   "위한 사양관리」 + 제공 지침표. 출처끼리 갈리는 값은 "
                   "갈린 채로 둔다 — 습도·일교차."),
    }
