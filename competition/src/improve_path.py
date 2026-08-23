"""현재 성적 ↔ **이 농장이 올릴 수 있는 상한**, 그리고 그 사이의 경로.

`farm_gap` 은 지표를 **중앙값**으로 되돌렸을 때를 본다. 여기는 다른 질문에
답한다 — **최대로 올리면 어디까지인가.** 그리고 그 최대가 어디서 오는지를
갈라 놓는다:

    분포 상한   466농장 상위 10% — "국내에 실제로 있는 농장이 내는 수"  `실측`
    돈사 상한   지어 놓은 방·분만틀이 허용하는 천장                     `계산`
    ─────────  둘 중 **작은 쪽**이 이 농장의 달성 가능 상한이다

## 왜 돈사 상한을 겹치나

상위 10% 가 복당 12두를 낸다고 우리 농장 목표를 12두로 잡으면, 자돈사가
못 받는 생산량을 "낼 수 있다" 고 말하게 된다. 이 프로젝트가 실제로 그렇게
허수 594두·1.4억원을 낸 적이 있고, 그래서 `capacity_from_rooms` 가
`weaned_ceiling`(방 자리 ÷ (분만틀 × 배치수))을 낸다. **분포 상한을 돈사
상한으로 깎는 것**이 이 모듈이 하는 일이다.

## 지키는 것 넷

- **개입 효과를 주장하지 않는다.** "이 값을 올리면 PSY 가 오른다" 가 아니라
  "이 값이 그 수였다면 **공식상** PSY 가 이렇게 나온다" 다. 실농장 개입
  실험은 하지 않았다.
- **합산하지 않는다.** 지표를 하나씩만 바꿔 본다. 전부 올린 값도 내지만
  그건 개별 몫의 합이 아니라 **다시 계산한 하나의 시나리오**다 — PSY 는
  곱셈 항등식이라 합이 성립하지 않는다.
- **못 바꾸는 것은 지렛대로 세지 않는다.** 임신기간이 그렇다.
- **PSY 는 `farm_gap.psy_from` 을 그대로 부른다.** 여기서 재구현하지 않는다.

    python competition/src/improve_path.py        # 합성 시연 (등급 합성)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import farm_gap as fg                                          # noqa: E402

# 농장이 실제로 움직일 수 있는 지표와, 상한을 어느 분위수에서 읽는지.
# 낮을수록 좋은 지표는 p10 이 상한이다.
LEVERS = {
    "weaned": ("p90", True, "복당 이유두수"),
    "npd": ("p10", False, "비생산일수(연간)"),
    "lactation": ("p10", False, "포유기간"),
}
# 임신기간은 뺀다 — 줄이라는 권고가 성립하지 않는다(farm_gap 과 같은 판단).


def _psy(d: dict) -> float:
    return fg.psy_from(d["weaned"], d["npd"], d["lactation"], d["gestation"])


def ceilings(farm: dict, weaned_ceiling: float | None = None,
             stats: dict | None = None) -> dict:
    """지표별 상한과 그 출처. 돈사 상한이 있으면 분포 상한을 깎는다."""
    q = (stats or fg.load_stats())["quantiles"]
    out = {}
    for k, (qk, higher, label) in LEVERS.items():
        dist = float(q[k][qk])
        cap, src = dist, f"466농장 상위10%({qk})"
        # 복당 이유두수만 돈사가 천장을 정한다 — 방 자리가 받아야 나온다
        if k == "weaned" and weaned_ceiling is not None:
            if weaned_ceiling < dist:
                cap, src = float(weaned_ceiling), "돈사 상한(자돈사 자리)"
        cur = farm.get(k)
        out[k] = {
            "label": label, "current": cur, "ceiling": round(cap, 2),
            "source": src, "higher_better": higher,
            "distribution_ceiling": round(dist, 2),
            "barn_ceiling": (round(float(weaned_ceiling), 2)
                             if k == "weaned" and weaned_ceiling is not None
                             else None),
            # 이미 상한 위에 있으면 지렛대가 아니다 — 그대로 말한다
            "at_ceiling": (cur is not None
                           and ((cur >= cap) if higher else (cur <= cap))),
        }
    return out


def contrast(farm: dict, weaned_ceiling: float | None = None,
             stats: dict | None = None) -> dict:
    """현재 PSY ↔ 지표를 상한으로 바꿨을 때의 PSY. **하나씩만** 바꾼다."""
    q = (stats or fg.load_stats())["quantiles"]
    med = {k: q[k]["p50"] for k in q}
    base = {k: float(farm.get(k, med.get(k, 0)))
            for k in ("weaned", "npd", "lactation", "gestation")}
    base_psy = round(_psy(base), 2)

    caps = ceilings(farm, weaned_ceiling, stats={"quantiles": q})
    rows = []
    for k, c in caps.items():
        if c["at_ceiling"]:
            rows.append({"지표": c["label"], "현재": c["current"],
                         "상한": c["ceiling"], "출처": c["source"],
                         "PSY": base_psy, "여지": 0.0,
                         "왜": "이미 상한 이상 — 여기서는 더 못 얻는다"})
            continue
        trial = dict(base, **{k: c["ceiling"]})
        psy = round(_psy(trial), 2)
        rows.append({"지표": c["label"], "현재": c["current"],
                     "상한": c["ceiling"], "출처": c["source"],
                     "PSY": psy, "여지": round(psy - base_psy, 2), "왜": None})
    rows.sort(key=lambda r: -r["여지"])

    # 전부 상한 — **합이 아니라 다시 계산한 하나의 시나리오**
    allcap = dict(base, **{k: c["ceiling"] for k, c in caps.items()})
    all_psy = round(_psy(allcap), 2)
    sum_each = round(sum(r["여지"] for r in rows), 2)

    return {
        "psy_now": base_psy, "psy_all_ceiling": all_psy,
        "headroom_total": round(all_psy - base_psy, 2),
        "sum_of_each": sum_each,
        "rows": rows, "ceilings": caps, "grade": "계산",
        "notes": [
            "**개입 효과가 아니다.** '이 값이 그 수였다면 공식상 PSY 가 "
            "이렇게 나온다' 까지다 — 올리면 오른다는 주장이 아니고, 실농장 "
            "개입 실험은 하지 않았다.",
            f"**합이 아니다.** 개별 여지의 합 {sum_each}두 vs 전부 상한 "
            f"{round(all_psy - base_psy, 2)}두 — PSY 는 곱셈 항등식이라 "
            "따로 계산해야 한다. 개별 몫을 더해서 말하지 않는다.",
            "복당 이유두수의 상한은 **분포와 돈사 중 작은 쪽**이다. 방이 못 "
            "받는 생산량을 목표로 잡으면 허수가 된다.",
            "임신기간은 지렛대에서 뺐다 — 줄이라는 권고가 성립하지 않는다.",
        ],
    }


def plan(farm: dict, weaned_ceiling: float | None = None,
         stats: dict | None = None) -> dict:
    """개선 경로 — 여지가 큰 순서로, **무엇을 해야 그 값이 되는지**까지.

    처방 문구는 이 프로젝트가 이미 실측으로 세운 인과 경로에서 온다:
    하락은 사양이 아니라 발정·교배 관리에서 오고(NPD +11.6일 선행,
    이유두수 +0.00), 임신사고의 66.9% 가 재발이며, 여름 손실 경로는
    착상기다. 새 처방을 여기서 지어내지 않는다.
    """
    c = contrast(farm, weaned_ceiling, stats)
    how = {
        "비생산일수(연간)": (
            "재귀발정일과 재발을 줄이는 일이다. 이유 후 3~21일 발정 창과 "
            "교배 후 18~24일 재발 창을 놓치지 않는 것 — 임신사고의 66.9% 가 "
            "재발이고, 성적이 무너질 때 먼저 움직이는 것도 NPD(+11.6일)다."),
        "복당 이유두수": (
            "포유 중 폐사를 줄이는 일이고, **자돈사 자리가 천장을 정한다.** "
            "돈사 상한에 걸려 있으면 사양이 아니라 방을 늘려야 오른다."),
        "포유기간": (
            "짧게 하면 회전이 빨라지지만 이유체중이 떨어져 이유 후 육성률로 "
            "되돌아온다. **PSY 만 보고 줄이면 MSY 에서 잃는다** — 이 표는 "
            "PSY 만 보므로 여기 여지는 MSY 로 다시 확인할 것."),
    }
    steps = []
    for i, r in enumerate(c["rows"], 1):
        if r["여지"] <= 0:
            continue
        steps.append({"순서": i, "지표": r["지표"],
                      "현재→상한": f'{r["현재"]} → {r["상한"]}',
                      "PSY 여지": r["여지"], "상한 출처": r["출처"],
                      "무엇을": how.get(r["지표"], "")})
    return {**c, "steps": steps,
            "note_order": ("여지가 큰 순서다 — 이것은 **효과 순서가 아니라 "
                           "공식상 민감도 순서**다. 실행 난이도는 농장이 안다.")}


def _demo() -> dict:
    """합성 시연 — 돈사 상한이 분포 상한을 깎는 구성."""
    return {"weaned": 10.2, "npd": 62.0, "lactation": 26.0, "gestation": 115.0}


def main(argv=None) -> int:
    import json
    farm = _demo() if not argv else json.load(open(argv[0], encoding="utf-8"))
    # 예시 돈사 상한 11.0두 — 자돈사 396자리 ÷ 분만틀 36
    r = plan(farm, weaned_ceiling=11.0)
    print("=" * 72)
    print("  현재 ↔ 달성 가능 상한 (**등급 합성** — 실농장 아님)"
          if not argv else "  현재 ↔ 달성 가능 상한")
    print("=" * 72)
    print(f"  지금 PSY {r['psy_now']}두  →  전부 상한이면 "
          f"{r['psy_all_ceiling']}두  (여지 {r['headroom_total']:+.2f}두)")
    print(f"  {'지표':<14}{'현재':>7}{'상한':>8}{'PSY':>8}{'여지':>8}  출처")
    for row in r["rows"]:
        print(f"  {row['지표']:<14}{row['현재']:>7}{row['상한']:>8}"
              f"{row['PSY']:>8}{row['여지']:>+8.2f}  {row['출처']}"
              + (f"  ({row['왜']})" if row["왜"] else ""))
    print(f"\n  개선 경로 — {r['note_order'].replace('**', '')}")
    for s in r["steps"]:
        print(f"   {s['순서']}. {s['지표']}  {s['현재→상한']}  "
              f"(PSY {s['PSY 여지']:+.2f}두, 상한 {s['상한 출처']})")
        print(f"      {s['무엇을'].replace('**', '')}")
    print()
    for n in r["notes"]:
        print(f"  ⚠ {n.replace('**', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
