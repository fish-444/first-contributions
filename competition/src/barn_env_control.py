"""돈사 환경 위험 알람 — 지침 대역(절대)과 자기 기준선 편차(상대)를 **겹으로** 본다.

**알람만 낸다** — 제어 지시(냉방·환기 몇 단)는 내지 않는다. 조치는 장비와
현장이 정하고, 이 모듈은 "지금 위험한가 · 평소와 다른가" 두 질문에만 답한다.

온도·암모니아 센서는 돈사마다 설치 위치·보정이 달라 **절대값을 돈사끼리
비교하면 센서 차이를 사육환경 차이로 착각한다.** 그래서 층을 가른다:

- **위험 (지침 층·절대)** — 성장단계별 적온 대역과 암모니아 상한을
  벗어나면 `위험`. 이력이 없어도 첫날부터 동작한다.
- **주의 (편차 층·상대)** — 돈사·센서별 자기 기준선(중앙값·IQR) 대비
  z 가 자기 이력 컷을 넘으면 `주의` — "지침 안이지만 평소와 다르다,
  센서·급이·밀도·분뇨를 점검하라"까지다. 돈사 간 비교는 이 층으로만 한다.

편차 계산은 행동 기준선 층(`behavior_baseline`)의 `_robust`(IQR 산포)와
`_calibrate_cut`(자기 이력 경보율 0.5~5% 역산)을 **그대로 가져다 쓴다** —
같은 문제(절대값에 공통 답이 없다)에 같은 답이고, 문턱을 여기서 또
발명하지 않는다.

## 지침 대역 — `지침` 등급, 발명한 수가 아니다

적온은 표준 사양관리 권장 범위, 암모니아 상한 25ppm 은 축사 환경 권고
수준이다. 농장이 다른 기준을 쓰면 `guide=` 로 통째로 바꾼다 — 상수를
바꾸려고 코드를 열게 하지 않는다.

## 왜 제어 지시를 내지 않는가

겨울의 저온+고암모니아가 이유다: 환기를 늘리면 암모니아는 내려가지만
온도가 무너진다 — 옳은 조치가 장비 구성(최소 환기율·열원)에 달려 있어서,
로그만 보는 쪽이 지시를 내면 틀린다. 그래서 위험 둘을 **동시에** 울리는
것까지가 이 모듈의 일이고, 조치의 우선순위는 현장 몫이다.

    python competition/src/barn_env_control.py                # 합성 시연
    python competition/src/barn_env_control.py --log env.csv  # barn,stage,time,temp_c,nh3_ppm
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from behavior_baseline import MIN_WINDOWS, _calibrate_cut, _robust  # noqa: E402

SENSORS = ("temp_c", "nh3_ppm")

# 성장단계별 적온 대역(℃) — 표준 사양관리 권장 범위. `지침` 등급.
TEMP_GUIDE = {
    "교배·임신": (15.0, 21.0),
    "분만(모돈)": (18.0, 22.0),
    "자돈(이유 초기)": (26.0, 30.0),
    "육성·비육": (18.0, 24.0),
}
NH3_LIMIT = 25.0   # ppm — 축사 환경 권고 상한. `지침` 등급.


def baseline(history: list) -> dict:
    """한 돈사·한 센서의 이력 → {center, spread, cut, n, formed}.

    이력이 `MIN_WINDOWS` 미만이면 기준선을 만들지 않는다 — 행동 기준선
    층과 같은 이유로, 서너 개로 만든 중앙값 위의 경보는 근거가 없다.
    """
    x = np.asarray(history, float)
    out = {"n": len(x), "formed": len(x) >= MIN_WINDOWS,
           "center": None, "spread": None, "cut": None}
    if not out["formed"]:
        return out
    out["center"], out["spread"] = _robust(x)
    z = (x - out["center"]) / out["spread"]
    out["cut"] = _calibrate_cut(np.abs(z))     # 양쪽 이탈 모두 이상이다
    return out


def _guide_state(stage: str, sensor: str, now: float,
                 guide: dict) -> tuple:
    """지침 층 판정 → (상태, 대역 설명)."""
    if sensor == "temp_c":
        lo, hi = guide["temp"].get(stage, guide["temp"]["교배·임신"])
        if now < lo:
            return "저온 위반", f"적온 {lo:g}~{hi:g}℃"
        if now > hi:
            return "고온 위반", f"적온 {lo:g}~{hi:g}℃"
        return "적정", f"적온 {lo:g}~{hi:g}℃"
    if now > guide["nh3"]:
        return "상한 초과", f"상한 {guide['nh3']:g}ppm"
    return "적정", f"상한 {guide['nh3']:g}ppm"


def _alarms(sensors: dict) -> list:
    """알람만 — 지침 위반은 `위험`, 지침 안의 큰 편차는 `주의`(점검)."""
    out = []
    for s, v in sensors.items():
        if v["guide_state"] != "적정":
            out.append({"수준": "위험",
                        "내용": f"{s} {v['guide_state']} — 지금 {v['now']:g}, "
                               f"{v['guide']}"})
        elif v["alert"]:
            out.append({"수준": "주의",
                        "내용": f"{s} 지침 안이지만 평소와 다르다"
                               f"(z {v['z']:+.1f}) — 센서·급이·밀도·분뇨 점검"})
    return out


def assess(log: dict, stages: dict, guide: dict | None = None) -> dict:
    """{돈사: {센서: [이력..., 현재]}} → 돈사별 두 층 판정과 권고.

    마지막 값이 현재, 그 앞이 이력이다. 반환의 `ranking` 은 돈사 간
    비교인데 **원값이 아니라 |z| 로 줄 세운다** — 센서가 돈사마다 달라
    원값 비교는 센서 차이를 사육환경 차이로 읽는 짓이다.
    """
    guide = guide or {"temp": TEMP_GUIDE, "nh3": NH3_LIMIT}
    barns = {}
    for barn, series in log.items():
        stage = stages.get(barn, "교배·임신")
        sensors = {}
        for s in SENSORS:
            hist, now = series[s][:-1], float(series[s][-1])
            b = baseline(hist)
            z = (round((now - b["center"]) / b["spread"], 2)
                 if b["formed"] else None)
            alert = bool(b["formed"] and b["cut"] is not None
                         and z is not None and abs(z) >= b["cut"])
            st = _guide_state(stage, s, now, guide)
            sensors[s] = {"now": now, "guide": st[1], "guide_state": st[0],
                          "baseline_n": b["n"], "formed": b["formed"],
                          "z": z, "cut": b["cut"], "alert": alert}
        barns[barn] = {"stage": stage, "sensors": sensors,
                       "alarms": _alarms(sensors)}
    ranking = sorted(
        ((barn, s, v["z"]) for barn, d in barns.items()
         for s, v in d["sensors"].items() if v["z"] is not None),
        key=lambda t: -abs(t[2]))
    return {"barns": barns,
            "ranking": [{"돈사": b, "센서": s, "z": z} for b, s, z in ranking],
            "grade": "계산",
            "notes": [
                "알람만 낸다 — 제어 지시는 장비 구성에 달려 있어 로그만 "
                "보는 쪽이 정하면 틀린다(겨울 저온+고NH3 가 대표).",
                "위험=지침 위반(절대), 주의=평소와 다름(자기 기준선 편차) — "
                "두 층은 서로를 덮지 않는다.",
                "돈사 간 비교는 |z| 로만 한다 — 센서가 돈사마다 달라 원값 "
                "비교는 센서 차이를 사육환경 차이로 읽는다.",
                f"적온 대역·암모니아 상한 {NH3_LIMIT:g}ppm 은 지침값이다. "
                "농장 기준이 다르면 guide= 로 통째로 바꾼다.",
                "편차 컷은 자기 이력 경보율 0.5~5% 역산이다 — 행동 기준선 "
                "층과 같은 방법, 같은 코드다.",
            ]}


def load_log(path: str) -> tuple:
    """CSV(barn,stage,time,temp_c,nh3_ppm) → (log, stages). time 순 정렬."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: (r["barn"], r["time"]))
    log, stages = {}, {}
    for r in rows:
        d = log.setdefault(r["barn"], {s: [] for s in SENSORS})
        for s in SENSORS:
            d[s].append(float(r[s]))
        stages[r["barn"]] = r.get("stage") or "교배·임신"
    return log, stages


def _demo() -> tuple:
    """합성 시연 — 센서 바이어스와 겨울 상충이 핵심이다."""
    rng = np.random.default_rng(9)
    n = 60
    log, stages = {}, {}
    base = rng.normal(18.5, 0.8, n)
    #  1동: 평범.  2동: 같은 환경인데 센서가 +3℃ 치우침 — 원값 비교의 함정.
    log["1동"] = {"temp_c": list(base + rng.normal(0, .2, n)),
                  "nh3_ppm": list(rng.normal(12, 2, n))}
    log["2동"] = {"temp_c": list(base + 3.0 + rng.normal(0, .2, n)),
                  "nh3_ppm": list(rng.normal(11, 2, n))}
    #  3동: 오늘 암모니아가 평소의 갑절 + 저온 — 겨울 상충 케이스
    t3 = list(rng.normal(16.5, 0.6, n)); t3[-1] = 13.8
    a3 = list(rng.normal(14, 2, n)); a3[-1] = 31.0
    log["3동"] = {"temp_c": t3, "nh3_ppm": a3}
    stages = {"1동": "교배·임신", "2동": "교배·임신", "3동": "분만(모돈)"}
    return log, stages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="돈사 환경 — 지침·편차 두 층")
    ap.add_argument("--log", help="CSV: barn,stage,time,temp_c,nh3_ppm")
    args = ap.parse_args(argv)

    if args.log:
        log, stages = load_log(args.log)
        r = assess(log, stages)
    else:
        log, stages = _demo()
        r = assess(log, stages)
        print("=" * 72)
        print("  돈사 환경 위험 알람 — 합성 시연 (**등급 합성** — 실농장 아님)")
        print("=" * 72)
    for barn, d in r["barns"].items():
        line = []
        for s, v in d["sensors"].items():
            z = f"z {v['z']:+.1f}" if v["z"] is not None else "기준선 미형성"
            line.append(f"{s} {v['now']:.1f} [{v['guide_state']}·{z}]")
        print(f"  {barn}({d['stage']:<7}) " + " · ".join(line))
        for a in d["alarms"]:
            print(f"      🔔 [{a['수준']}] {a['내용']}")
    top = r["ranking"][:3]
    print("  편차 순위(|z|): " + " · ".join(
        f"{t['돈사']}/{t['센서']} {t['z']:+.1f}" for t in top))
    for nline in r["notes"]:
        print(f"  ⚠ {nline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
