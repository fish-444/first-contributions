"""돈사 → 받을 수 있는 두수 · 연간 출하 상한 · 상한까지 가는 길.

전부 `batch_flow` 를 부른다. 이 파일에 산술이 없어야 한다 — 있으면 CLI 와
API 가 같은 농장에 대해 다른 답을 하게 된다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import batch_flow as bf                                        # noqa: E402
import build_farm_setup as bfs                                 # noqa: E402
import farm_economics as fe                                    # noqa: E402

from ..schemas import FarmSetup

router = APIRouter(prefix="/api/capacity", tags=["capacity"])


def _extra_rooms(setup: FarmSetup) -> dict:
    """시뮬레이터는 하위 단계마다 방을 따로 쓴다.

    자돈사를 일령 한 구간(46일)으로 세면 3방인데 전·후기로 나뉘어 4방이
    필요하다. 이걸 안 얹으면 등록은 통과하고 돌리면 적체가 난다 —
    등록 화면이 쓰는 것과 같은 보정이다.
    """
    iv, wash = float(setup.interval_days), float(setup.washout_days)
    out: dict = {}
    for s in bfs.sim_stages():
        need = max(1, -(-(int(s["days"]) + int(wash)) // int(iv)))
        out[s["stage"]] = out.get(s["stage"], 0) + need
    return out


def _weaned_per_crate(setup: FarmSetup) -> float:
    """**방을 지을 때 쓴 값과 같아야 한다.**

    `batch_flow` 기본값 12.0 은 설계 *목표*고 모델값은 11.0 이다. 목표로
    되읽으면 396자리 자돈사가 33분만틀로 보이는데 지을 때는 36으로 잡았다 —
    같은 돈사가 방향에 따라 다른 크기로 나온다.
    """
    w = setup.performance.weaned
    return float(w) if w is not None else float(bfs.defaults()["weaned"])


def compute(setup: FarmSetup) -> dict:
    """등록 정보 → 용량 + 상한 + 처방. 라우터와 테스트가 같이 쓴다."""
    barns = [b.model_dump() for b in setup.barns]
    cap = bf.capacity_from_rooms(
        barns, float(setup.interval_days),
        lactation=int(setup.lactation_days),
        pre_farrow=int(setup.pre_farrow_days),
        washdown=int(setup.washout_days),
        extra_rooms=_extra_rooms(setup),
        weaned_per_crate=_weaned_per_crate(setup))

    p = setup.performance
    tp = bf.throughput(
        cap,
        farrow_rate=None if p.farrowing_rate is None else p.farrowing_rate / 100.0,
        weaned_per_litter=p.weaned,
        grow_survival=None if p.survival is None else p.survival / 100.0)

    margin = fe.margin_per_pig()
    return {
        "capacity": cap,
        "throughput": tp,
        # 원/년은 **한계 이익**이다. 총원가로 재면 개선의 값이 40% 작게 나온다 —
        # 돈사·모돈·인력이 이미 있는 상태에서 빈 틀을 채우는 것이라 노무비·
        # 감가상각은 그 한 두 때문에 늘지 않는다.
        "margin_per_pig": margin,
        "given": any(v is not None for v in
                     (p.farrowing_rate, p.weaned, p.survival)),
    }


@router.get("/preset", summary="모돈 두수 → 기본 돈사 구성")
def preset(sows: int = 300, interval_days: float = 21,
           lactation: float = 24, pre_farrow: float = 7,
           washout: float = 7, weaned: float = 11.0) -> dict:
    """**프론트가 이걸 직접 계산하면 안 된다.**

    실제로 그렇게 했다가 자돈사를 3방으로 잡았는데 서버는 4방을 요구해서,
    기본 구성을 넣자마자 "막힘" 이 떴다 — 화면이 `extra_rooms` 보정을 모르기
    때문이었다. 설계와 검사가 **같은 보정**을 쓰도록 여기서 준다.
    """
    if not 1 <= sows <= 20000:
        raise HTTPException(422, "모돈 두수는 1~20000")
    stub = FarmSetup(interval_days=interval_days, lactation_days=lactation,
                     pre_farrow_days=pre_farrow, washout_days=washout)
    return {"n_sows": sows,
            "barns": bf.design_barns(
                sows, float(interval_days), lactation=int(lactation),
                pre_farrow=int(pre_farrow), washdown=int(washout),
                weaned_per_litter=float(weaned),
                extra_rooms=_extra_rooms(stub))}


@router.post("", summary="돈사 등록 → 용량 · 상한 · 처방")
def capacity_of(setup: FarmSetup) -> dict:
    if not setup.barns:
        raise HTTPException(422, "돈사가 하나도 등록되지 않았다 — "
                                 "방을 만들어 내지 않는다")
    return compute(setup)


@router.post("/watch", summary="배치 전이 400일 감시 (AIAO·역류·적체·무처소)")
def watch(setup: FarmSetup, days: int = 400) -> dict:
    """`barn_watch` 를 등록 정보로 돌린다.

    정적 대조는 "자리가 맞는가" 까지고, 배치가 실제로 밟아 나가는지는 돌려
    봐야 안다 — 무처소·적체·역류는 거기서만 잡힌다.
    """
    import barn_watch as bw
    import run_farm as rf

    if not setup.barns:
        raise HTTPException(422, "돈사가 등록되지 않았다")
    raw = setup.model_dump()
    cfg = bw.default_config()
    sid, why = bw.batch_system_from_setup(raw, cfg)
    if sid:
        cfg.batch_system_id = sid
    n_sows = setup.n_sows or 300
    want = rf.crates_for_sows(n_sows, cfg)
    have = bw.crates_from_setup(raw)
    cfg.crate_count = have or want
    spec, notes = bw.rooms_from_setup(raw, cfg.merged())
    if not spec:
        raise HTTPException(422, "돈군흐름(자돈~비육) 돈사가 등록되지 않았다")
    cfg.rooms = spec
    r = bw.watch(cfg, days=days,
                 rooms=bw.build_rooms(cfg.merged(), from_config=True))
    # 스냅샷 전체는 크다. 판정에 필요한 것만 낸다.
    r.pop("snapshots", None)
    r["batch_system"] = cfg.batch_system_id
    r["crate_count"] = cfg.crate_count
    r["notes"] = [{"barn": a, "stage": b, "why": c} for a, b, c in notes]
    if why:
        r["notes"].append({"barn": None, "stage": None, "why": why})
    return r


@router.get("/sweep", summary="house 마다 방을 빼며 견디는 한계")
def sweep(sows: int = 300, days: int = 400) -> dict:
    import barn_watch as bw
    import run_farm as rf

    cfg = bw.default_config()
    cfg.crate_count = rf.crates_for_sows(sows, cfg)
    return {"n_sows": sows, "rows": bw.sweep(cfg, days=days)}
