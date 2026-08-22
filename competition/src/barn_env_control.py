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

# 아는 센서 — 로그에 있는 열만 본다. temp_c 와 nh3_ppm 이 기본이고
# rh_pct(상대습도)·h2s_ppm(황화수소)은 있으면 같이 본다.
SENSORS = ("temp_c", "nh3_ppm", "rh_pct", "h2s_ppm")

# ── 지침 상수 — 전부 `지침` 등급, 출처가 붙어 있다 ──────────────────────
# 성장단계별 적온 대역(℃) — 사용자 제공 표준 사양관리 표(2026-08-22).
TEMP_GUIDE = {
    "임신돈·웅돈": (16.0, 21.0),
    "포유모돈": (18.0, 21.0),
    "포유자돈": (30.0, 35.0),
    "이유자돈": (22.0, 29.0),
    "육성초기": (20.0, 27.0),
    "육성후기": (18.0, 22.0),
    "비육돈": (16.0, 21.0),
}
# 성장단계별 습도 대역(%) — 「환절기 돼지 사양관리」 표1 을 단계에 맞춘 것.
# 출처끼리 갈린다: 같은 글 본문은 60~70, 사용자 자료는 50~60 을 권한다.
# 표1(단계별)이 가장 구체적이라 그걸 쓰되 이 갈림을 여기 적어 둔다.
RH_GUIDE = {
    "포유자돈": (60.0, 70.0),
    "이유자돈": (60.0, 80.0),
    "육성초기": (50.0, 80.0),
    "육성후기": (40.0, 60.0),
    "비육돈": (40.0, 60.0),
    "임신돈·웅돈": (40.0, 60.0),
    "포유모돈": (40.0, 60.0),
}
NH3_LIMIT = 15.0   # ppm — 「환절기 돼지 사양관리」(국립축산과학원): 15ppm 초과 금지
H2S_LIMIT = 5.0    # ppm — 같은 자료: 황화수소 5ppm 초과 금지

# 최적온도 아래로 떨어질 때의 추가 사료요구량(g/일/두) — 사용자 제공 표.
# 행 = 체중(kg), 열 = 최적온도 대비 부족분(℃).
FEED_EXTRA = {
    20: {-2: 25, -4: 50, -6: 75, -8: 100, -10: 125},
    40: {-2: 31, -4: 62, -6: 92, -8: 123, -10: 154},
    60: {-2: 36, -4: 72, -6: 108, -8: 145, -10: 181},
    80: {-2: 42, -4: 83, -6: 125, -8: 166, -10: 208},
    100: {-2: 47, -4: 94, -6: 141, -8: 188, -10: 235},
    120: {-2: 53, -4: 105, -6: 158, -8: 210, -10: 263},
}

# 환경온도×풍속별 "이 주령 이하는 불쾌" 문턱(주령) — 사용자 제공 표.
# 열 = 풍속 ≤0.15 / 0.15~0.25 / 0.25~0.36 m/s. 0 = 전 주령 쾌적, 99 = 전부 불쾌.
COMFORT_UNCOMFY_BELOW_WEEKS = {
    21: (0, 0, 0), 18: (1, 5, 12), 15: (10 / 7, 3, 12), 13: (8, 12, 14),
    10: (15, 14, 16), 7: (20, 16, 20), 4: (20, 20, 20), 2: (99, 99, 99),
}

# 단열 점검 기준 — 사용자 제공 자료: 자리 간 온도차 2.8℃ 이상, 일교차
# 8.3℃ 이상이면 단열을 점검한다. 출처끼리 또 갈린다: 축산원 자료는
# 육성·비육 일일 온도차 5℃ 이내, 환절기 외기 일교차 10℃ 를 위험으로 본다.
SPOT_DIFF_LIMIT = 2.8
DAILY_RANGE_LIMIT = 8.3


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
        lo, hi = guide["temp"].get(stage, guide["temp"]["임신돈·웅돈"])
        if now < lo:
            return "저온 위반", f"적온 {lo:g}~{hi:g}℃"
        if now > hi:
            return "고온 위반", f"적온 {lo:g}~{hi:g}℃"
        return "적정", f"적온 {lo:g}~{hi:g}℃"
    if sensor == "rh_pct":
        lo, hi = guide["rh"].get(stage, guide["rh"]["임신돈·웅돈"])
        if now < lo:
            return "저습 위반", f"권장 {lo:g}~{hi:g}%"
        if now > hi:
            return "과습 위반", f"권장 {lo:g}~{hi:g}%"
        return "적정", f"권장 {lo:g}~{hi:g}%"
    limit = guide["h2s"] if sensor == "h2s_ppm" else guide["nh3"]
    if now > limit:
        return "상한 초과", f"상한 {limit:g}ppm"
    return "적정", f"상한 {limit:g}ppm"


def _alarms(sensors: dict) -> list:
    """알람만 — 지침 위반은 `위험`, 지침 안의 큰 편차는 `주의`(점검).

    지침 위반인데 **편차로는 평소와 같은 수준**(편차 경보 없음)이면 그
    사실을 알람에 붙인다 — 상시 위반이거나 센서가 치우친 것이고, 어느
    쪽이든 급한 조치보다 센서 교정 확인이 먼저다. 편차 층이 절대값
    알람의 센서 바이어스를 걸러 주는 자리이고, 새 문턱은 없다(기존
    편차 경보 정의를 그대로 쓴다).
    """
    out = []
    for s, v in sensors.items():
        if v["guide_state"] != "적정":
            msg = (f"{s} {v['guide_state']} — 지금 {v['now']:.1f}, "
                   f"{v['guide']}")
            if v["formed"] and not v["alert"]:
                msg += (f" · 평소와 같은 수준(z {v['z']:+.1f}) — 상시 "
                        "위반이거나 센서 치우침, 교정 확인")
            out.append({"수준": "위험", "내용": msg})
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
    guide = guide or {"temp": TEMP_GUIDE, "rh": RH_GUIDE,
                      "nh3": NH3_LIMIT, "h2s": H2S_LIMIT}
    barns = {}
    for barn, series in log.items():
        stage = stages.get(barn, "임신돈·웅돈")
        sensors = {}
        for s in (k for k in SENSORS if k in series):
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
                f"적온·습도 대역, NH3 {NH3_LIMIT:g}ppm·H2S {H2S_LIMIT:g}ppm "
                "상한은 지침값이다(국립축산과학원 자료·사용자 제공 표). "
                "농장 기준이 다르면 guide= 로 통째로 바꾼다.",
                "출처끼리 갈리는 값은 갈린 채로 적었다 — 습도(50~60 vs "
                "60~70 vs 표1 단계별)·일교차(5 vs 8.3 vs 외기 10℃).",
                "편차 컷은 자기 이력 경보율 0.5~5% 역산이다 — 행동 기준선 "
                "층과 같은 방법, 같은 코드다.",
            ]}


def cold_feed_penalty(weight_kg: float, deficit_c: float) -> int:
    """최적온도 아래 deficit_c(음수, ℃)일 때 추가 사료요구량(g/일/두).

    사용자 제공 지침표의 조회다 — 보간하지 않고 **가장 가까운 칸**을
    읽는다(표 밖 외삽 금지: 체중은 20~120, 부족분은 -2~-10 에 물린다).
    """
    if deficit_c > -2:
        return 0
    w = min(FEED_EXTRA, key=lambda k: abs(k - weight_kg))
    d = max(-10, min(-2, round(deficit_c / 2) * 2))
    return FEED_EXTRA[w][d]


def comfort(temp_c: float, wind_ms: float, age_weeks: float) -> str:
    """환경온도×풍속×주령 → '쾌적'/'불쾌'. 사용자 제공 표의 조회다."""
    rows = sorted(COMFORT_UNCOMFY_BELOW_WEEKS, reverse=True)
    row = next((t for t in rows if temp_c >= t), rows[-1])
    col = 0 if wind_ms <= 0.15 else (1 if wind_ms <= 0.25 else 2)
    return "불쾌" if age_weeks <= COMFORT_UNCOMFY_BELOW_WEEKS[row][col] else "쾌적"


def insulation_alarms(day_temps: list | None = None,
                      spot_temps: list | None = None) -> list:
    """단열 점검 알람 — 일교차·자리 간 온도차. 데이터가 있을 때만 잰다.

    `day_temps` 는 같은 날 한 돈사의 온도들, `spot_temps` 는 같은 시각
    여러 지점의 온도들이다. 기준(8.3℃·2.8℃)은 사용자 제공 자료이고,
    축산원 자료는 육성·비육 일일 온도차 5℃ 이내를 권한다 — 더 엄한
    기준을 쓰려면 인자로 바꾼다.
    """
    out = []
    if day_temps and len(day_temps) >= 2:
        rng = max(day_temps) - min(day_temps)
        if rng >= DAILY_RANGE_LIMIT:
            out.append({"수준": "주의",
                        "내용": f"일교차 {rng:.1f}℃ ≥ {DAILY_RANGE_LIMIT}℃ — "
                               "돈사 단열 점검"})
    if spot_temps and len(spot_temps) >= 2:
        diff = max(spot_temps) - min(spot_temps)
        if diff >= SPOT_DIFF_LIMIT:
            out.append({"수준": "주의",
                        "내용": f"자리 간 온도차 {diff:.1f}℃ ≥ {SPOT_DIFF_LIMIT}℃ — "
                               "돈사 단열 점검"})
    return out


def load_log(path: str) -> tuple:
    """CSV(barn,stage,time,temp_c,nh3_ppm) → (log, stages). time 순 정렬."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: (r["barn"], r["time"]))
    log, stages = {}, {}
    for r in rows:
        present = [s for s in SENSORS if (r.get(s) or "").strip() != ""]
        d = log.setdefault(r["barn"], {s: [] for s in present})
        for s in present:
            d[s].append(float(r[s]))
        stages[r["barn"]] = r.get("stage") or "임신돈·웅돈"
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
    log["2동"] = {"temp_c": list(base + 3.5 + rng.normal(0, .2, n)),
                  "nh3_ppm": list(rng.normal(11, 2, n))}
    #  3동: 오늘 암모니아가 평소의 갑절 + 저온 + 저습 — 겨울 케이스
    t3 = list(rng.normal(19.0, 0.6, n)); t3[-1] = 13.8
    a3 = list(rng.normal(9, 1.5, n)); a3[-1] = 22.0
    h3 = list(rng.normal(52, 3, n)); h3[-1] = 34.0
    log["3동"] = {"temp_c": t3, "nh3_ppm": a3, "rh_pct": h3}
    stages = {"1동": "임신돈·웅돈", "2동": "임신돈·웅돈", "3동": "포유모돈"}
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
