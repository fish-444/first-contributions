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


def _cycle(setup: FarmSetup) -> dict:
    """**주기와 회전율을 이 농장 값으로.** 안 그러면 모든 농장이 같게 나온다.

    회전율은 특히 그렇다. 관행 2.3 을 쓰면 NPD 34일 농장과 58일 농장이 같은
    규모로 계산되는데 실제로는 ±7% 갈린다. `herd_cycle` 이 검증된 PSY 항등식
    으로 되풀고, 비운 칸은 채우지 않고 실측 중앙으로 떨어뜨린 뒤 그 사실을
    `source` 에 남긴다.
    """
    return bf.herd_cycle(setup.performance.model_dump()
                         if setup.performance else None)


def compute(setup: FarmSetup) -> dict:
    """등록 정보 → 용량 + 상한 + 처방. 라우터와 테스트가 같이 쓴다."""
    barns = [b.model_dump() for b in setup.barns]
    cap = bf.capacity_from_rooms(
        barns, float(setup.interval_days),
        lactation=int(setup.lactation_days),
        pre_farrow=int(setup.pre_farrow_days),
        washdown=int(setup.washout_days),
        extra_rooms=_extra_rooms(setup),
        weaned_per_crate=_weaned_per_crate(setup),
        cycle=_cycle(setup))

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


@router.post("/relief", summary="병목을 풀면 그 다음은 누구인가 — 투자 순서")
def relief(setup: FarmSetup, steps: int = 5) -> dict:
    """`capacity` 는 "지금 무엇이 붙잡고 있나" 까지고, 이건 그 다음 질문이다.

    각 행의 `gain` 은 **그 돈사가 제약이 아니게 되면 얻는 몫**이다. 방 몇
    개를 더 지으라는 뜻이 아니고 **비용도 계산하지 않는다** — 어디를 건드려야
    움직이는지의 순서일 뿐이다.

    `gain` 이 0 인 구간은 **여러 돈사가 같은 수준에서 나란히 걸린 것**이라,
    하나만 넓혀도 두수가 안 는다. 그 사실이 이 표의 값어치다.
    """
    if not setup.barns:
        raise HTTPException(422, "돈사가 등록되지 않았다")
    rows = bf.relief_chain(
        [b.model_dump() for b in setup.barns], float(setup.interval_days),
        steps=steps,
        lactation=int(setup.lactation_days),
        pre_farrow=int(setup.pre_farrow_days),
        washdown=int(setup.washout_days),
        extra_rooms=_extra_rooms(setup),
        weaned_per_crate=_weaned_per_crate(setup), cycle=_cycle(setup))
    # 같은 수준에 나란히 걸린 묶음을 표시한다 — 하나만 풀면 안 움직인다
    tied: list = []
    for r in rows:
        if tied and tied[-1]["n_sows"] == r["n_sows"]:
            tied[-1]["stages"].append(r["binding"])
        else:
            tied.append({"n_sows": r["n_sows"], "stages": [r["binding"]],
                         "gain": r.get("gain")})
    for i, g in enumerate(tied[:-1]):
        g["gain"] = tied[i + 1]["n_sows"] - g["n_sows"]
    if tied:
        tied[-1]["gain"] = None
    return {"rows": rows, "groups": tied,
            "note": ("각 몫은 그 돈사가 제약이 아니게 되면 얻는 것이다. "
                     "방 몇 개를 지으라는 뜻이 아니고 비용도 계산하지 않는다.")}


@router.post("/interval", summary="간격 what-if — 같은 돈사, 배치 간격만 바꾸면")
def interval_whatif(setup: FarmSetup) -> dict:
    """**돈사는 그대로 두고 간격만 바꾼다.** 짓기 전이 아니라 지은 뒤의 질문이다.

    `batch_flow.compare()` 는 모돈 두수에서 출발해 "이 간격이면 방이 몇 개
    필요하다" 를 낸다. 그건 설계 방향이고, 이미 지은 농장이 묻는 건 반대다 —
    **내 방으로 간격을 바꾸면 규모와 출하가 어떻게 되나.**

    맞바꿈이 세 갈래로 갈린다:

    · 좁히면  같은 분만틀이 더 자주 도니 **규모가 는다**. 대신 회전이 빨라져
              방 수 요구가 커지고 어느 단계가 **막힌다**.
    · 넓히면  방은 넉넉해지는데 배치가 커져 **한 날에 몰린다**(집중도).
              그리고 같은 틀로 도는 횟수가 줄어 **규모가 준다**.

    **`extra_rooms` 를 간격마다 다시 낸다.** 필요 방 수가 간격의 함수라
    한 번 계산해 돌려 쓰면 21일 기준 보정을 35일 행에 적용하게 된다 —
    막히지 않을 곳이 막힌 것으로 나온다.
    """
    if not setup.barns:
        raise HTTPException(422, "돈사가 등록되지 않았다")
    barns = [b.model_dump() for b in setup.barns]
    wpc = _weaned_per_crate(setup)
    cyc = _cycle(setup)
    p = setup.performance
    now = float(setup.interval_days)

    rows = []
    for name, iv in bf.BATCH_INTERVALS.items():
        iv = float(iv)
        # 간격마다 다시 — 방 소요가 간격의 함수다
        stub = setup.model_copy(update={"interval_days": iv})
        cap = bf.capacity_from_rooms(
            barns, iv,
            lactation=int(setup.lactation_days),
            pre_farrow=int(setup.pre_farrow_days),
            washdown=int(setup.washout_days),
            extra_rooms=_extra_rooms(stub), weaned_per_crate=wpc,
            cycle=cyc)
        tp = bf.throughput(
            cap,
            farrow_rate=None if p.farrowing_rate is None
            else p.farrowing_rate / 100.0,
            weaned_per_litter=p.weaned,
            grow_survival=None if p.survival is None else p.survival / 100.0)
        # 막힌 간격은 지지 두수가 0 이다. 그 0 으로 배치 크기를 내면 "배치당
        # 0.0두" 라는 있지도 않은 수가 표에 찍힌다 — 비워 둔다.
        pl = bf.plan(cap["n_sows"], iv) if cap["n_sows"] > 0 else None
        rows.append({
            "name": name, "interval_days": iv,
            "current": abs(iv - now) < 1e-9,
            "n_sows": cap["n_sows"], "binding": cap["binding"],
            "blocked": [r["stage"] for r in cap["rows"] if r.get("why")],
            # 상한은 방만으로 정해지고, 지금값은 성적이 있어야 뜻이 있다
            "ceiling_year": tp["ceiling_year"], "now_year": tp["now_year"],
            "weaned_ceiling": cap.get("weaned_ceiling"),
            "services_per_batch": pl and pl["services_per_event"],
            # 집중도는 간격만의 함수(iv/7)라 막혀도 뜻이 있다
            "peak_ratio": round(iv / 7.0, 1),
            "farrow_rooms_need": next(
                (r["need_rooms"] for r in cap["rows"]
                 if r["stage"] == "분만사"), None),
        })

    ok = [r for r in rows if r["n_sows"] > 0]
    best = max(ok, key=lambda r: r["n_sows"]) if ok else None
    return {
        "current_interval": now, "rows": rows,
        "best": None if best is None else best["interval_days"],
        "given": any(v is not None
                     for v in (p.farrowing_rate, p.weaned, p.survival)),
        # **간격은 성적이 아니라 배선이다.** 규모가 큰 쪽이 늘 옳은 게
        # 아니라, 좁힐수록 사람이 매주 같은 일을 더 자주 해야 한다.
        "note": ("규모가 가장 큰 간격이 늘 정답은 아니다 — 좁힐수록 방 수 "
                 "요구가 커지고 작업이 잦아진다. 이 표는 맞바꿈을 보여줄 뿐 "
                 "인력·공사비를 계산하지 않는다."),
    }


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
