"""교배 배정 계획 — 농장장의 표를 그대로, 손 대신 산식으로 채운다.

농장장이 손으로 만들던 표의 열이 곧 출력 스키마다:

    모돈번호 · 모돈인덱스 · 웅돈번호 · 웅돈인덱스 · 후손의 예상인덱스 ·
    근친율(%) · 교배횟수

## 산식 셋 — 전부 표준이고, 여기서 발명한 것이 없다

- **후손의 예상인덱스 = (모돈인덱스 + 웅돈인덱스) / 2** — 중간부모 기대값.
  인덱스(유전평가) 자체의 정확도는 우리가 만들지 않았다 — 입력의 질을
  그대로 따라간다.
- **근친율 F(자손) = 혈연계수 kinship(모돈, 웅돈)** — 재귀 표법(tabular).
  **아는 혈통에서의 하한이다**: 혈통표에 없는 조상은 무관으로 치므로,
  기록이 얕을수록 실제보다 낮게 나온다. 이 한 줄이 각주로 파일을 따라간다.
- **배정 = 근친 한도·웅돈 사용 상한 아래에서 예상인덱스 합 최대화** —
  헝가리안(scipy `linear_sum_assignment`). 모돈마다 최고 웅돈을 주는 탐욕이
  아니라 **농장 전체의 최적**이다 — 최고 웅돈 하나를 두 모돈이 다투면
  탐욕은 한쪽을 크게 손해 보게 한다.

## 지침 상수

- 근친 한도 기본 **6.25%** (사촌 교배 수준의 관례 한도) — `지침` 값이고
  `--max-f` 로 바꾼다. 여기서 지어낸 문턱이 아니다.
- 웅돈 사용 상한: 웅돈 CSV 에 `max_services` 열이 있으면 그 값, 없으면
  `--services` (기본 3).

## 입력 CSV

    모돈  id,index[,sire,dam]
    웅돈  id,index[,sire,dam][,max_services]

sire/dam 이 목록에 없는 번호면 그 번호를 시조(부모 미상)로 취급한다 —
같은 부 번호를 적은 두 개체는 그만큼의 혈연으로 이어진다.

    python competition/src/mating_plan.py                      # 합성 시연
    python competition/src/mating_plan.py --sows sows.csv --boars boars.csv \
        [--max-f 0.0625] [--services 3] [--out plan.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAX_F_GUIDE = 0.0625     # 지침 — 사촌 교배 수준. 발명한 문턱이 아니다
_BIG = 1e9


class Pedigree:
    """id → (sire, dam) 로 혈연계수를 재귀 표법으로 계산한다."""

    def __init__(self, parents: dict):
        self.p = {k: (s or None, d or None) for k, (s, d) in parents.items()}
        self._depth: dict = {}
        self._kin: dict = {}

    def _parents(self, x):
        return self.p.get(x, (None, None))

    def depth(self, x, _stack=()) -> int:
        if x in _stack:
            raise ValueError(f"혈통표에 순환이 있다: {x}")
        if x not in self._depth:
            s, d = self._parents(x)
            self._depth[x] = 0 if s is None and d is None else 1 + max(
                self.depth(q, _stack + (x,)) for q in (s, d) if q is not None)
        return self._depth[x]

    def inbreeding(self, x) -> float:
        """F(x) — 부모가 둘 다 알려졌을 때만 0 이 아닐 수 있다."""
        s, d = self._parents(x)
        if s is None or d is None:
            return 0.0
        return self.kinship(s, d)

    def kinship(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        key = (a, b) if a <= b else (b, a)
        if key in self._kin:
            return self._kin[key]
        if a == b:
            v = 0.5 * (1.0 + self.inbreeding(a))
        else:
            # 세대가 깊은 쪽을 부모로 푼다 — 종료가 보장되는 표준 순서
            if self.depth(a) < self.depth(b):
                a, b = b, a
            s, d = self._parents(a)
            v = 0.5 * (self.kinship(s, b) + self.kinship(d, b))
        self._kin[key] = v
        return v


def load_animals(path: str) -> dict:
    """CSV → {id: {...}}. id·index 는 필수, 나머지는 있으면 쓴다."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            i = r["id"].strip()
            out[i] = {
                "index": float(r["index"]),
                "sire": (r.get("sire") or "").strip() or None,
                "dam": (r.get("dam") or "").strip() or None,
                "max_services": int(r["max_services"])
                if (r.get("max_services") or "").strip() else None,
            }
    return out


def plan(sows: dict, boars: dict, max_f: float = MAX_F_GUIDE,
         services: int = 3) -> dict:
    """배정표. 열은 농장장의 표 그대로 + 배정 불가 사유."""
    from scipy.optimize import linear_sum_assignment

    ped = Pedigree({k: (v["sire"], v["dam"])
                    for k, v in {**sows, **boars}.items()})
    s_ids = sorted(sows)
    slots = []                              # (웅돈, 슬롯번호)
    for b in sorted(boars):
        for k in range(boars[b]["max_services"] or services):
            slots.append((b, k))

    F = {(s, b): ped.kinship(s, b) for s in s_ids for b in sorted(boars)}
    val = np.full((len(s_ids), len(slots)), -_BIG)
    for i, s in enumerate(s_ids):
        for j, (b, _) in enumerate(slots):
            if F[(s, b)] <= max_f + 1e-12:
                val[i, j] = (sows[s]["index"] + boars[b]["index"]) / 2.0

    rows, unassigned, used = [], [], {b: 0 for b in boars}
    if slots:
        ri, ci = linear_sum_assignment(val, maximize=True)
        chosen = {int(i): int(j) for i, j in zip(ri, ci) if val[i, j] > -_BIG / 2}
    else:
        chosen = {}
    for i, s in enumerate(s_ids):
        if i in chosen:
            b = slots[chosen[i]][0]
            used[b] += 1
            rows.append({
                "모돈번호": s, "모돈인덱스": sows[s]["index"],
                "웅돈번호": b, "웅돈인덱스": boars[b]["index"],
                "후손의 예상인덱스":
                    round((sows[s]["index"] + boars[b]["index"]) / 2, 2),
                "근친율(%)": round(F[(s, b)] * 100, 2),
                "교배횟수": None,           # 아래서 웅돈별 합으로 채운다
            })
        else:
            ok_any = any(F[(s, b)] <= max_f + 1e-12 for b in boars)
            unassigned.append({
                "모돈번호": s, "모돈인덱스": sows[s]["index"],
                "사유": ("웅돈 사용 상한에 밀림 — 상한을 늘리거나 웅돈 추가"
                        if ok_any else
                        f"근친 한도 초과 — 전 웅돈이 F > {max_f * 100:g}%"),
            })
    for r in rows:
        r["교배횟수"] = used[r["웅돈번호"]]   # 이 계획에서 그 웅돈이 맡는 총 횟수

    mean_idx = (round(float(np.mean([r["후손의 예상인덱스"] for r in rows])), 2)
                if rows else None)
    return {
        "rows": rows, "unassigned": unassigned,
        "boar_use": {b: {"배정": used[b],
                         "상한": boars[b]["max_services"] or services}
                     for b in sorted(boars)},
        "mean_expected_index": mean_idx,
        "max_f": max_f, "grade": "계산",
        "notes": [
            "근친율은 입력한 혈통에서의 **하한**이다 — 혈통표에 없는 조상은 "
            "무관으로 치므로, 기록이 얕을수록 실제보다 낮게 나온다.",
            "후손의 예상인덱스는 중간부모 기대값이다 — 인덱스(유전평가) 자체의 "
            "정확도는 입력의 질을 따라간다.",
            f"근친 한도 {max_f * 100:g}% 는 지침값(사촌 교배 수준)이다.",
            "배정은 모돈별 최고가 아니라 농장 전체 최적이다(헝가리안).",
        ],
    }


def to_csv(result: dict, bare: bool = False) -> str:
    """농장장 표 그대로의 CSV — 등급 열과 각주 머리말이 붙는다."""
    import table_export as te
    rows = [{"등급": "계산", **r} for r in result["rows"]]
    rows += [{"등급": "계산", "모돈번호": u["모돈번호"],
              "모돈인덱스": u["모돈인덱스"], "웅돈번호": f"(미배정) {u['사유']}",
              "웅돈인덱스": None, "후손의 예상인덱스": None,
              "근친율(%)": None, "교배횟수": None}
             for u in result["unassigned"]]
    return te.to_csv(rows, {
        "title": "교배 배정 계획 — 근친 한도·웅돈 상한 아래 예상인덱스 최대화",
        "grade": "계산", "module": "mating_plan",
        "source": f"근친 한도 {result['max_f'] * 100:g}% (지침) · "
                  f"평균 예상인덱스 {result['mean_expected_index']}",
        "caveats": result["notes"],
    }, bare=bare)


def _demo() -> dict:
    """합성 시연 — 최고 웅돈이 근친으로 막히는 상황이 핵심이다."""
    sows = {"S001": {"index": 110.0, "sire": "F1", "dam": None, "max_services": None},
            "S002": {"index": 105.0, "sire": "F2", "dam": None, "max_services": None},
            "S003": {"index": 98.0, "sire": "F1", "dam": None, "max_services": None}}
    boars = {"B-X": {"index": 120.0, "sire": "F1", "dam": None, "max_services": 2},
             "B-Y": {"index": 112.0, "sire": "F3", "dam": None, "max_services": 2},
             "B-Z": {"index": 104.0, "sire": "F4", "dam": None, "max_services": 2}}
    return plan(sows, boars)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="교배 배정 계획")
    ap.add_argument("--sows", help="모돈 CSV: id,index[,sire,dam]")
    ap.add_argument("--boars", help="웅돈 CSV: id,index[,sire,dam][,max_services]")
    ap.add_argument("--max-f", type=float, default=MAX_F_GUIDE,
                    help="근친 한도 (기본 0.0625 = 지침)")
    ap.add_argument("--services", type=int, default=3,
                    help="웅돈 사용 상한 기본값 (열이 없을 때)")
    ap.add_argument("--out", help="CSV 저장 경로")
    ap.add_argument("--bare", action="store_true", help="머리말 없이")
    args = ap.parse_args(argv)

    if args.sows and args.boars:
        r = plan(load_animals(args.sows), load_animals(args.boars),
                 max_f=args.max_f, services=args.services)
    else:
        r = _demo()
        print("=" * 72)
        print("  교배 배정 계획 — 합성 시연 (**등급 합성** — 실농장 아님)")
        print("=" * 72)
    for row in r["rows"]:
        print("  " + " · ".join(f"{k} {v}" for k, v in row.items()))
    for u in r["unassigned"]:
        print(f"  미배정 {u['모돈번호']} — {u['사유']}")
    print(f"  평균 예상인덱스 {r['mean_expected_index']} · "
          f"웅돈 사용 {r['boar_use']}")
    for n in r["notes"]:
        print(f"  ⚠ {n}")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            f.write(to_csv(r, bare=args.bare))
        print(f"  CSV → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
