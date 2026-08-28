"""번식 성적 공식 — **공식 정의를 정본으로, 입력 변수를 이름으로 받는다.**

지금까지 PSY·MSY 는 모듈마다 필요한 자리에서 계산됐다. 이 모듈은 공식
정의(용어해설·산식)를 한 곳에 두고, **어느 입력이 어느 결과로 갔는지**를
결과에 붙여 돌려준다. 비운 입력은 채우지 않고 "못 낸 것"으로 남긴다.

## 용어 (제공 정의 그대로)

    MSY  연간 모돈 두당 출하두수      PSY  연간 모돈 두당 이유두수
    NPD  일 년 중 모돈이 생산적인 일(임신·포유)에 종사하지 못한 일수
    발정재귀일  이유에서 교배까지의 일수

## 산식

    이유두수(복당)  = 실산자수 × 이유 전 육성율
    출하두수(복당)  = 실산자수 × 이유 전 육성율 × 이유 후 육성율
    모돈회전율      = (365 − NPD) / (임신기간 + 포유기간)
    PSY             = 복당 이유두수 × 모돈회전율
    MSY             = 복당 출하두수 × 모돈회전율  ( = PSY × 이유 후 육성율)
    분만율(%)       = 분만복수 / 교배복수 × 100
    이유 전 육성율  = 이유두수 / 실산자수 × 100
    이유 후 육성율  = 출하두수 / 이유두수 × 100
    7일이내재귀율   = 7일이내 재귀복수 / 총 재귀복수 × 100

## 제공 표기와 어긋나는 곳 둘 — **고르지 않고 적어 둔다**

제공된 표에는 회전율이 `365 / (모돈사육일수 + 평균 임신기간)`, 모돈사육
일수가 `임신기간 + 포유기간 + 비생산일수` 로 적혀 있다. 그대로 계산하면
임신기간이 두 번 들어가고, NPD 를 **연간** 값으로 정의해 놓고 **한 사이클**
길이에 더하게 된다. 임신 114·포유 24·NPD 60 이면:

    제공 표기 그대로   365 / (114+24+60+114) = 1.17   ← 현장값 2.2~2.4 밖
    임신 중복만 제거   365 / (114+24+60)     = 1.84   ← 여전히 낮다
    NPD 를 연간으로    (365−60) / (114+24)   = 2.21   ← 이 모듈이 쓰는 값

셋째가 NPD 의 **공식 정의(일 년 중 …)** 와 정합하고, 466행에서 PSY 항등식
`PSY = 이유두수 × (365−NPD)/(임신+포유)` 로 **86.2% 가 오차 0.05 이내**임을
이미 확인했다. 그래서 셋째를 쓰되, 제공 표기 그대로의 값도 `variants` 로
같이 돌려준다 — 어느 쪽을 봤는지 읽는 사람이 알아야 한다.

둘째: `PSY = 이유두수 × 모돈회전율 / 모돈수` 의 `이유두수` 는 아래 세부
정의(실산자수×이유전육성율)로는 **복당** 값이라, 거기에 다시 모돈수를
나누면 단위가 깨진다(두/모돈/년 ÷ 두). 농장 **총** 이유두수를 쓸 때만
`/모돈수` 가 맞다. 두 읽기를 다 낼 수 있게 `psy_from_total()` 을 따로 뒀다.

## 모돈수의 정의 — 우리가 아직 안 쓰는 것

제공 정의는 `모돈수 : 6개월 전 임신한 모돈수` 다. 출하는 오늘 나가지만
그 돼지를 밴 모돈은 과거에 있었으니 분모를 뒤로 미루는 것이고, 규모가
변하는 농장에서 MSY 가 달라진다. 우리 코드는 **상시모돈**을 쓴다 —
정의가 다르므로 `head_basis` 로 어느 쪽인지 밝혀 돌려준다.

    python competition/src/perf_formula.py        # 합성 시연 (등급 합성)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 입력 변수 명세 — **이름·단위·범위를 한 곳에.** 등록 화면과 API 스키마가
# 이 표를 보고 칸을 만든다. 여기 없는 변수는 공식이 안 쓰는 변수다.
INPUTS = {
    "live_born": ("실산자수", "두/복", 4.0, 20.0),
    "pre_wean_survival": ("이유 전 육성율", "%", 50.0, 100.0),
    "post_wean_survival": ("이유 후 육성율", "%", 50.0, 100.0),
    "gestation": ("임신기간", "일", 108.0, 120.0),
    "lactation": ("포유기간", "일", 14.0, 42.0),
    "npd": ("비생산일수(연간)", "일", 0.0, 200.0),
    "n_sows": ("모돈수", "두", 1.0, 20000.0),
    "services": ("교배복수", "복", 0.0, 100000.0),
    "farrowings": ("분만복수", "복", 0.0, 100000.0),
    "returns_total": ("총 재귀복수", "복", 0.0, 100000.0),
    "returns_7d": ("7일이내 재귀복수", "복", 0.0, 100000.0),
    "weaned_total": ("연간 총 이유두수", "두", 0.0, 1000000.0),
    "shipped_total": ("연간 총 출하두수", "두", 0.0, 1000000.0),
}
# 어느 결과가 어느 입력을 쓰는가 — 결과에 그대로 붙여 돌려준다
USES = {
    "weaned_per_litter": ("live_born", "pre_wean_survival"),
    "shipped_per_litter": ("live_born", "pre_wean_survival", "post_wean_survival"),
    "turnover": ("npd", "gestation", "lactation"),
    "psy": ("live_born", "pre_wean_survival", "npd", "gestation", "lactation"),
    "msy": ("live_born", "pre_wean_survival", "post_wean_survival",
            "npd", "gestation", "lactation"),
    "farrowing_rate": ("farrowings", "services"),
    "return_7d_rate": ("returns_7d", "returns_total"),
}


def _f(v):
    return None if v is None or v == "" else float(v)


def turnover(npd, gestation, lactation) -> dict:
    """모돈회전율 — 정합 판과 제공 표기 그대로를 **둘 다** 돌려준다."""
    npd, g, la = _f(npd), _f(gestation), _f(lactation)
    if None in (npd, g, la) or g + la <= 0:
        return {"value": None, "why": "NPD·임신기간·포유기간이 모두 필요하다"}
    prod = g + la
    v = round((365.0 - npd) / prod, 3)
    return {
        "value": v,
        "formula": "(365 − NPD) / (임신기간 + 포유기간)",
        "basis": ("NPD 의 공식 정의가 **연간** 비생산일수이므로 연간 생산일"
                  "(365−NPD)을 한 사이클의 생산일(임신+포유)로 나눈다. "
                  "466행에서 PSY 항등식으로 86.2% 오차 0.05 이내 확인."),
        "variants": {
            "제공 표기 그대로 365/(임신+포유+NPD+임신)":
                round(365.0 / (prod + npd + g), 3),
            "임신 중복만 제거 365/(임신+포유+NPD)":
                round(365.0 / (prod + npd), 3),
        },
    }


def psy_from_total(weaned_total, n_sows) -> dict:
    """농장 **총** 이유두수 ÷ 모돈수 — 제공 표의 `/모돈수` 읽기."""
    w, n = _f(weaned_total), _f(n_sows)
    if None in (w, n) or n <= 0:
        return {"value": None, "why": "총 이유두수와 모돈수가 필요하다"}
    return {"value": round(w / n, 2), "formula": "총 이유두수 / 모돈수",
            "basis": "복당 값이 아니라 농장 총량을 쓰는 읽기다"}


def msy_from_total(shipped_total, n_sows) -> dict:
    s, n = _f(shipped_total), _f(n_sows)
    if None in (s, n) or n <= 0:
        return {"value": None, "why": "총 출하두수와 모돈수가 필요하다"}
    return {"value": round(s / n, 2), "formula": "총 출하두수 / 모돈수",
            "basis": "복당 값이 아니라 농장 총량을 쓰는 읽기다"}


def _rate(num, den, label) -> dict:
    a, b = _f(num), _f(den)
    if None in (a, b):
        return {"value": None, "why": f"{label}의 두 값이 모두 필요하다"}
    if b <= 0:
        return {"value": None, "why": f"{label}의 분모가 0 이다"}
    return {"value": round(a / b * 100, 2)}


def compute(inp: dict, head_basis: str = "상시모돈") -> dict:
    """입력 변수 → 공식 결과. **비운 것은 비운 채로 남긴다.**

    `head_basis` 는 모돈수의 정의다. 제공 정의는 '6개월 전 임신한 모돈수'
    이고 우리 기본은 '상시모돈' 이라, 어느 쪽인지 결과에 밝혀 둔다.
    """
    lb = _f(inp.get("live_born"))
    pre = _f(inp.get("pre_wean_survival"))
    post = _f(inp.get("post_wean_survival"))
    tv = turnover(inp.get("npd"), inp.get("gestation"), inp.get("lactation"))

    weaned = round(lb * pre / 100, 2) if None not in (lb, pre) else None
    shipped = (round(weaned * post / 100, 2)
               if None not in (weaned, post) else None)
    t = tv["value"]
    psy = round(weaned * t, 2) if None not in (weaned, t) else None
    msy = round(shipped * t, 2) if None not in (shipped, t) else None

    out = {
        "weaned_per_litter": {"value": weaned, "unit": "두/복"},
        "shipped_per_litter": {"value": shipped, "unit": "두/복"},
        "turnover": {**tv, "unit": "복/모돈/년"},
        "psy": {"value": psy, "unit": "두/모돈/년",
                "formula": "복당 이유두수 × 모돈회전율"},
        "msy": {"value": msy, "unit": "두/모돈/년",
                "formula": "복당 출하두수 × 모돈회전율 (= PSY × 이유 후 육성율)"},
        "farrowing_rate": {**_rate(inp.get("farrowings"), inp.get("services"),
                                   "분만율"), "unit": "%",
                           "formula": "분만복수 / 교배복수 × 100"},
        "return_7d_rate": {**_rate(inp.get("returns_7d"),
                                   inp.get("returns_total"), "7일이내재귀율"),
                           "unit": "%",
                           "formula": "7일이내 재귀복수 / 총 재귀복수 × 100"},
        "psy_from_total": psy_from_total(inp.get("weaned_total"),
                                         inp.get("n_sows")),
        "msy_from_total": msy_from_total(inp.get("shipped_total"),
                                         inp.get("n_sows")),
    }
    # 어느 입력이 들어갔는지 — 빈 칸이면 무엇 때문에 못 냈는지
    for key, names in USES.items():
        if key not in out:
            continue
        missing = [INPUTS[n][0] for n in names if _f(inp.get(n)) is None]
        out[key]["uses"] = [INPUTS[n][0] for n in names]
        if missing and out[key].get("value") is None:
            out[key]["missing"] = missing

    given = sorted(k for k in INPUTS if _f(inp.get(k)) is not None)
    return {
        "results": out, "grade": "계산", "head_basis": head_basis,
        "given": [INPUTS[k][0] for k in given],
        "missing": [INPUTS[k][0] for k in INPUTS if k not in given],
        "notes": [
            "**빈 칸을 채우지 않는다.** 못 낸 결과는 무엇이 없어서 못 냈는지 "
            "이름으로 말한다 — 중앙값으로 메우면 격차가 늘 0 이 된다.",
            "회전율은 NPD 의 공식 정의(연간)와 정합하는 (365−NPD)/(임신+포유) "
            "를 쓴다. 제공 표기 그대로의 값도 variants 에 같이 낸다.",
            f"모돈수 기준은 '{head_basis}' 다. 제공 정의('6개월 전 임신한 "
            "모돈수')와 다르면 MSY 가 달라진다 — 규모가 변하는 농장에서 특히.",
            "복당 값(PSY·MSY)과 농장 총량 기준(psy_from_total)은 **다른 "
            "읽기**라 따로 낸다. 한 칸에 섞으면 단위가 깨진다.",
        ],
    }


def _demo() -> dict:
    """합성 시연 — 일부 칸을 일부러 비워 '못 낸 이유'가 나오게 한다."""
    return {"live_born": 11.5, "pre_wean_survival": 90.0,
            "post_wean_survival": 94.0, "gestation": 114.0,
            "lactation": 24.0, "npd": 60.0, "n_sows": 300,
            "services": 1250, "farrowings": 1060,
            "returns_total": 190}          # returns_7d 를 비웠다


def main(argv=None) -> int:
    import json
    inp = _demo() if not argv else json.load(open(argv[0], encoding="utf-8"))
    r = compute(inp)
    print("=" * 72)
    print("  번식 성적 공식 — 합성 시연 (**등급 합성** — 실농장 아님)"
          if not argv else "  번식 성적 공식")
    print("=" * 72)
    print(f"  입력 {len(r['given'])}개: {' · '.join(r['given'])}")
    if r["missing"]:
        print(f"  비움 {len(r['missing'])}개: {' · '.join(r['missing'])}")
    print()
    for k, v in r["results"].items():
        val = v.get("value")
        unit = v.get("unit", "")
        if val is None:
            why = v.get("why") or ("필요: " + " · ".join(v.get("missing", [])))
            print(f"  {k:<20} —        ({why})")
        else:
            print(f"  {k:<20} {val:>8} {unit}")
    tv = r["results"]["turnover"]
    if tv.get("variants"):
        print("\n  회전율 표기 대조:")
        print(f"    쓰는 값  {tv['value']}  ← {tv['formula']}")
        for name, v in tv["variants"].items():
            print(f"    {name}  {v}")
    print()
    for n in r["notes"]:
        print(f"  ⚠ {n.replace('**', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
