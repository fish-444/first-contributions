"""행동 어휘 축소 — **응용이 쓰는 넷으로 접어서 다시 잰다.**

재학습이 아니다. 가중치도 분할도 피처도 그대로 두고, `y_true` 와 `y_pred`
를 같은 표로 접기만 한다. 그런데도 성능이 달라진다 — 응용이 안 쓰는 구분의
혼동이 점수에서 빠지기 때문이다.

자세에서 이미 본 성질이다: 5클래스 0.557 → 응용에 필요한 3클래스 0.732.
혼동이 사라진 게 아니라 **응용이 그 구분을 안 써서 무해해진 것**이다.

## 어휘의 정본은 여기가 아니다

`behavior_baseline.HEAD_SIGNS` 가 구성비로 쓰는 넷이 응용 어휘이고,
제공 모델 계약(`pig_behavior.RELIABLE_CLASSES`)도 같은 넷이다. 이 모듈은
그 둘에서 어휘를 **읽어 온다** — 여기 따로 적으면 셋이 갈린다.

## 병합표는 규약이지 손잡이가 아니다

`docs/PREREGISTRATION.md` 등록 4 에 측정 전 커밋해 둔 표를 그대로 쓴다.
숫자를 보고 표를 고치면 튜닝이지 설계가 아니라서, 테스트가 이 표를
등록 문서와 대조한다.

    python competition/src/behavior_vocab.py        # 병합 전/후 재측정
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import model_edinburgh_behavior as beh  # noqa: E402
import temporal_features as tfeat  # noqa: E402
from pig_behavior.predictor import RELIABLE_CLASSES  # noqa: E402

DATA = os.path.join(ROOT, "competition", "data", "edinburgh_frames.csv")
OUT = os.path.join(ROOT, "competition", "data", "behavior_vocab.json")

GEOM = ["bbox_w", "bbox_h", "aspect_ratio", "area", "speed", "accel",
        "darea", "centroid_x", "centroid_y"]
OTHER = "기타"

# 등록 4 의 병합표. **경계 둘은 근거가 있어서 그렇게 간 것이다** —
# standing 은 계약에서 Resting 과 별개 클래스라 안정에 섞지 않고,
# drink 는 계약의 Drinking 이 따로 있어 Eating 에 섞지 않는다.
MERGE = {
    "lying": "Resting", "sleep": "Resting", "sitting": "Resting",
    "walk": "Walking", "run": "Walking",
    "eat": "Eating",
    "investigating": "Searching",
}


def app_vocab(label: str) -> str:
    """원 라벨 → 응용 어휘. 응용이 안 쓰는 것은 전부 한 칸으로 접는다."""
    return MERGE.get(label, OTHER)


def _scores(y, pred) -> dict:
    return {"acc": round(float(accuracy_score(y, pred)), 3),
            "mf1": round(float(f1_score(y, pred, average="macro",
                                        zero_division=0)), 3),
            "n_classes": int(len(set(y)))}


def run(df: pd.DataFrame | None = None) -> dict:
    """12종으로 한 번 예측하고, 그 예측을 접어서 다시 잰다.

    **모델을 두 번 돌리지 않는다.** 같은 `pred` 를 두 어휘로 채점하는
    것이라, 차이는 전부 어휘에서 온 것이지 학습 차이가 아니다.
    """
    if df is None:
        df = pd.read_csv(DATA)
    df = tfeat.add_temporal(beh.add_motion(df))
    # 라벨 없는 행을 먼저 뺀다. 안 빼면 아래 `where(...)` 가 NaN 을 실재
    # 클래스 'other' 로 만들어, 정본 파이프라인(model_edinburgh_behavior)과
    # 갈린다 — 지금 CSV 엔 NaN 이 0 건이라 증상이 없지만 재파싱하면 갈린다.
    df = df[df["behavior"].notna()]
    vc = df["behavior"].value_counts()
    keep = set(vc[vc >= beh.MIN_COUNT].index)
    df["behavior"] = df["behavior"].where(df["behavior"].isin(keep), "other")
    feats = GEOM + tfeat.TEMPORAL_COLS
    df = df.dropna(subset=feats)
    y = df["behavior"].to_numpy()
    groups = df["individual_id"].to_numpy()

    clf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=-1,
                                 random_state=42)
    pred = cross_val_predict(clf, df[feats], y, cv=GroupKFold(5),
                             groups=groups, n_jobs=-1)

    ya = np.array([app_vocab(v) for v in y])
    pa = np.array([app_vocab(v) for v in pred])

    labels = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    # 어느 쌍이 서로 먹는가 — 대각선을 빼고 큰 순으로
    pairs = sorted(((float(cm[i, j]), labels[i], labels[j])
                    for i in range(len(labels)) for j in range(len(labels))
                    if i != j), reverse=True)[:8]
    # 병합이 지워 준 혼동 / 남은 혼동을 가른다
    survived = [(r, a, b) for r, a, b in pairs if app_vocab(a) != app_vocab(b)]
    absorbed = [(r, a, b) for r, a, b in pairs if app_vocab(a) == app_vocab(b)]

    # 다수 클래스 기준선 — **어휘를 줄이면 기준선도 같이 오른다.** 이걸 안
    # 내면 0.576 이 공짜로 얻은 몫인지 아닌지 알 수 없다.
    maj = pd.Series(ya).value_counts().idxmax()
    base = _scores(ya, np.array([maj] * len(ya)))
    base_raw = _scores(y, np.array([pd.Series(y).value_counts().idxmax()]
                                   * len(y)))

    app_labels = sorted(set(ya))
    per_class = {}
    for c in app_labels:
        m = ya == c
        per_class[c] = {"n": int(m.sum()),
                        "recall": round(float((pa[m] == c).mean()), 3)}

    return {
        "basis": ("Edinburgh 12,646행·96개체 · 개체 GroupKFold(5) · "
                  "기하+모션+롤링 20피처 RF · **같은 예측을 두 어휘로 채점**"),
        "raw": _scores(y, pred),
        "app": _scores(ya, pa),
        "raw_baseline": base_raw, "app_baseline": base,
        "lift_raw": round(_scores(y, pred)["acc"] - base_raw["acc"], 3),
        "lift_app": round(_scores(ya, pa)["acc"] - base["acc"], 3),
        "merge": dict(MERGE),
        "app_classes": sorted(RELIABLE_CLASSES),
        "other_label": OTHER,
        "per_class": per_class,
        "confusion_top": [{"rate": round(r, 3), "true": a, "pred": b,
                           "absorbed": app_vocab(a) == app_vocab(b)}
                          for r, a, b in pairs],
        "survived_pairs": len(survived), "absorbed_pairs": len(absorbed),
        "grade": "실측",
        "notes": [
            "재학습이 아니다 — 가중치·분할·피처가 같고 라벨 매핑만 다르다.",
            "여기서 재는 것은 **프레임 라벨 정확도**이지 의심 순위의 "
            "적중률이 아니다. 적중률은 정답 라벨이 없어 여전히 미측정이다.",
            "0.516 은 외형 40차원을 포함한 값이라 영상 캐시가 있어야 "
            "재현된다. 비교는 재현 가능한 20피처 위에서 했다.",
            "Edinburgh 공개셋이고 국내 농장이 아니다.",
        ],
    }


def main() -> int:
    import json
    r = run()
    print("=" * 72)
    print("  행동 어휘 축소 — 같은 예측을 두 어휘로 채점")
    print("=" * 72)
    print(f"  {r['basis']}\n")
    for tag, k, b in (("원 어휘", "raw", "raw_baseline"),
                      ("응용 어휘", "app", "app_baseline")):
        s, bs = r[k], r[b]
        print(f"  {tag:<10} {s['n_classes']:>2}종   정확도 {s['acc']:.3f} "
              f"· Macro-F1 {s['mf1']:.3f}   (다수 기준선 {bs['acc']:.3f} "
              f"→ 여유 {s['acc'] - bs['acc']:+.3f})")
    d_acc = r["app"]["acc"] - r["raw"]["acc"]
    d_mf1 = r["app"]["mf1"] - r["raw"]["mf1"]
    print(f"  {'차이':<10}        {d_acc:+.3f}        {d_mf1:+.3f}")
    print("\n  ⚠ 병합은 정확도를 **구조적으로 낮출 수 없다** — 병합군 안의"
          " 혼동이 정답이 되기 때문이다.\n"
          "    그러니 +차이 자체는 개선의 증거가 아니다. 읽을 것은 두 가지다:"
          " 남은 오차가 어디 있는가,\n"
          "    그리고 **같은 어휘의 다수 기준선 대비 여유**가 얼마나 남는가.")
    print("\n  응용 어휘별 재현율:")
    for c, v in sorted(r["per_class"].items(), key=lambda x: -x[1]["n"]):
        print(f"    {c:<10} n={v['n']:>6,}  recall {v['recall']:.3f}")
    print("\n  큰 혼동쌍 — 병합이 흡수했는가:")
    for p in r["confusion_top"]:
        mark = "흡수됨" if p["absorbed"] else "**남음**"
        print(f"    {p['true']:<16}→ {p['pred']:<16} {p['rate']:.3f}  {mark}")
    json.dump(r, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  저장: {OUT}")
    for n in r["notes"]:
        print(f"  ⚠ {n.replace('**', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
