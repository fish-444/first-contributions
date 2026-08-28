"""축사 환경(온·습·환기) 이상치 탐지 — 사분위 표시 + 날짜별 추이 + 알림 목록.

## 왜 절대 문턱이 아니라 편차인가

"온도 28℃ 는 이상" 같은 절대 문턱은 두 가지로 틀린다. 계절이 다르고 방이
다르다 — 1월의 28℃ 와 8월의 28℃ 는 다른 사건이고, 같은 날이라도 분만사와
비육사는 설정 온도가 다르다. 그래서 이 모듈은 **기준 집단 안의 편차**만
본다. `bio_baseline_71763` 이 생체지표에서 쓴 방식과 같은 원리이고, 로버스트
산포와 문턱 탐색은 **그 모듈을 그대로 가져다 쓴다**(같은 식을 두 벌 두면
갈린다).

## 기준 집단을 고정하지 않는다

무엇으로 중심화할지는 자료가 정한다. 후보는 `chamber`(방)와 `month`(월)이고,
**집단 간 분산을 가장 많이 설명하는 쪽**을 고른 뒤 그 사실을 결과에 남긴다
(`center` 필드). 챔버 시험 자료는 방이 지배하고 농장 센서 이력은 월이
지배하는데, 어느 쪽을 골랐는지 모르면 z 를 해석할 수 없다.

## 필드마다 문턱이 다르다

전 필드에 3σ 를 일괄로 물리면 문턱이 죽는다 — 실측으로 생체지표에서
back_temp 0건 / latent_heat 10.2% 가 나왔다. 환기율은 특히 분포가 치우쳐
있어(하한 근처에 몰림) 같은 문제가 난다. `fit_threshold` 로 필드마다 목표
알림률 대역에 들어오는 z 를 찾고, 못 찾으면 `usable` 에 그 사실을 적는다.

## 사분위(Q1~Q4)

값 자체의 사분위를 함께 낸다. 편차 z 는 "얼마나 튀었나"만 말하고 "어느
수준인가"를 말하지 않는데, 현장에서 조치는 수준에 달려 있다 — 같은 +3z 라도
환기 Q1(최저 사분위)에서 난 것과 Q4 에서 난 것은 다른 조치다. 사분위별
이상치 비율(`by_quartile`)이 그 관계를 드러낸다.

## `env_scale` · `barn_env_control` 과 무엇이 다른가

셋 다 돈사 환경을 보지만 **묻는 것이 다르다.** 겹쳐 두면 같은 문제에 답이
둘이 되므로 여기 못박는다.

| 모듈 | 묻는 것 | 기준 |
|---|---|---|
| `barn_env_control` | **지금 위험한가** | 지침 절대 대역(적온·습도·NH₃) |
| `env_scale` | **평소와 얼마나 다른가**(−1~1 눈금) | 돈사별 자기 기준선 · ±1 = 경보 경계 |
| `env_anomaly`(여기) | **어느 관측이 튀었는가 + 어느 수준에서** | 자동 선택 기준집단 · 사분위 |

`env_scale` 은 **화면에 띄울 눈금**을 만들고(행마다 −1~1), 이 모듈은
**감사용 목록**을 만든다(어느 날 어느 방이 얼마나, 사분위 어디에서).
산포 함수도 갈린다 — 이쪽은 `bio.robust_sd`(MAD), 저쪽은
`behavior_baseline._robust`(IQR)다. 같은 자료에 두 값이 나오는 게 정상이고,
**하나를 다른 하나의 근거로 인용하지 말 것.**

## 알림

**이 모듈은 아무것도 발송하지 않는다.** 알림 대상 목록을 JSON·CSV 로 떨어뜨릴
뿐이고 화면 표시는 `build_env_anomaly.py` 가 한다. 발송은 농장 시스템의 몫이다.

실행:
    python competition/src/env_anomaly.py <71763_라벨디렉터리>
    python competition/src/env_anomaly.py --clips clips.csv
    python competition/src/env_anomaly.py --synthetic     # 배관 검증용(합성)
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bio_baseline_71763 as bio  # noqa: E402
import parse_aihub  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(ROOT, "data", "env_anomaly.json")
OUT_CSV = os.path.join(ROOT, "outputs", "env_alerts.csv")

#: 감시 대상. 온·습·환기가 본체고 CO2·NH3 는 원인 해석용으로 같이 싣는다.
ENV_FIELDS = ["temp_c", "humidity_pct", "ventilation"]
EXTRA_FIELDS = ["co2_ppm", "nh3_ppm"]

LABELS = {"temp_c": "온도(℃)", "humidity_pct": "습도(%)",
          "ventilation": "환기율", "co2_ppm": "CO2(ppm)", "nh3_ppm": "NH3(ppm)"}

#: 중심화 후보. 앞에 둔 것이 우선이 아니라 **설명력으로 고른다.**
CENTER_CANDIDATES = ["chamber", "month"]
#: 기준 집단 하나가 이보다 작으면 그 집단 평균을 믿을 수 없다.
MIN_GROUP = 20
#: 필드 하나를 판정하는 데 필요한 최소 관측 수.
MIN_N = 100


def add_month(clips: pd.DataFrame) -> pd.DataFrame:
    """`date`(YYMMDD) → `month`(YYMM). 계절 중심화의 기준이 된다."""
    if "date" not in clips.columns:
        return clips
    d = clips["date"].astype(str).str.strip()
    clips = clips.copy()
    clips["month"] = d.str.slice(0, 4).where(d.str.len() >= 4)
    return clips


def _between_pct(g: pd.DataFrame, field: str, key: str) -> float:
    """집단 간 제곱합 비율(%) — 이 기준으로 중심화가 뜻이 있는지의 척도.

    제곱합으로 가르는 것은 `bio.variance_split` 과 같은 이유다. 집단내와
    집단간의 합이 100 이 되므로 표 안에서 서로 모순될 수 없다.
    """
    ss_tot = float(((g[field] - g[field].mean()) ** 2).sum())
    if not ss_tot:
        return 0.0
    within = g.groupby(key)[field].transform(lambda s: s - s.mean())
    return (ss_tot - float((within ** 2).sum())) / ss_tot * 100


def pick_center(clips: pd.DataFrame, field: str) -> tuple:
    """중심화 기준을 자료가 고르게 한다. 반환 (기준이름|None, 설명력%).

    `None` 은 전체 평균으로 중심화한다는 뜻이고, 후보가 없거나 어느 후보도
    집단 크기를 못 채울 때 그렇게 된다 — **조용히 넘어가지 않고** 결과의
    `center` 에 "전체" 로 적힌다.
    """
    g = clips.dropna(subset=[field])
    best = (None, 0.0)
    for key in CENTER_CANDIDATES:
        if key not in g.columns:
            continue
        h = g.dropna(subset=[key])
        vc = h[key].value_counts()
        if len(vc) < 2 or vc.min() < MIN_GROUP:
            continue
        pct = _between_pct(h, field, key)
        if pct > best[1]:
            best = (key, pct)
    return best


def quartiles(s: pd.Series) -> tuple:
    """값의 사분위 경계와 Q1~Q4 배정.

    같은 값이 4분의 1 을 넘게 차지하면 경계가 겹쳐 4칸이 안 나온다. 그때는
    나오는 만큼만 쓰고 `n_bins` 로 알린다 — 억지로 4칸을 만들면 없는 구분을
    있는 것처럼 보여 준다.
    """
    q = [float(s.quantile(p)) for p in (0.25, 0.50, 0.75)]
    edges = sorted(set([float(s.min())] + q + [float(s.max())]))
    if len(edges) < 3:
        return {"cuts": [round(v, 2) for v in q], "n_bins": 1}, \
            pd.Series("Q1", index=s.index)
    lab = ["Q%d" % (i + 1) for i in range(len(edges) - 1)]
    binned = pd.cut(s, bins=edges, labels=lab, include_lowest=True)
    return {"cuts": [round(v, 2) for v in q], "n_bins": len(lab)}, binned


def analyze(clips: pd.DataFrame, fields: list[str] | None = None) -> dict:
    """클립 표 → 필드별 기준·사분위·이상치 + 날짜별 추이 + 알림 목록."""
    fields = fields or [f for f in ENV_FIELDS + EXTRA_FIELDS
                        if f in clips.columns]
    clips = add_month(clips)
    per_field, alerts = {}, []

    for f in fields:
        if f not in clips.columns:
            continue
        g = clips.dropna(subset=[f])
        if len(g) < MIN_N:
            per_field[f] = {"label": LABELS.get(f, f), "n": int(len(g)),
                            "usable": "표본부족"}
            continue

        key, expl = pick_center(g, f)
        dev = (g[f] - g[f].mean()) if key is None else \
            g.groupby(key)[f].transform(lambda s: s - s.mean())
        rsd = bio.robust_sd(dev)
        if not rsd or rsd != rsd:
            per_field[f] = {"label": LABELS.get(f, f), "n": int(len(g)),
                            "usable": "산포0"}
            continue

        qinfo, qbin = quartiles(g[f])
        z, pct = bio.fit_threshold(dev, rsd)
        flag = dev.abs() > z * rsd
        lo, hi = bio.FLAG_BAND

        # 사분위별 이상치 비율 — 같은 z 도 수준에 따라 다른 조치다
        by_q = []
        for qn, idx in qbin.groupby(qbin, observed=True).groups.items():
            sub = flag.loc[idx]
            by_q.append({"q": str(qn), "n": int(len(sub)),
                         "value_mean": round(float(g.loc[idx, f].mean()), 2),
                         "n_flagged": int(sub.sum()),
                         "flagged_pct": round(float(sub.mean() * 100), 1)})

        per_field[f] = {
            "label": LABELS.get(f, f), "n": int(len(g)),
            "center": "전체" if key is None else key,
            "center_explains_pct": round(expl, 1),
            "mean": round(float(g[f].mean()), 2),
            "robust_sd": round(float(rsd), 3), "z": z,
            "quartile_cuts": qinfo["cuts"], "n_bins": qinfo["n_bins"],
            "n_flagged": int(flag.sum()), "flagged_pct": round(pct, 1),
            "usable": "쓸만함" if lo <= pct <= hi else (
                "문턱무의미" if pct < lo else "알림과다"),
            "by_quartile": sorted(by_q, key=lambda r: r["q"]),
        }

        # 알림 목록 — 발송하지 않는다. 어디서 무엇이 얼마나 튀었나까지.
        for i in g.index[flag]:
            alerts.append({
                "field": f, "label": LABELS.get(f, f),
                "date": None if "date" not in g.columns
                else str(g.at[i, "date"]),
                "chamber": None if "chamber" not in g.columns
                else str(g.at[i, "chamber"]),
                "value": round(float(g.at[i, f]), 2),
                "dev": round(float(dev.loc[i]), 2),
                "z": round(float(dev.loc[i] / rsd), 2),
                "quartile": str(qbin.loc[i]),
                "direction": "높음" if dev.loc[i] > 0 else "낮음",
            })

    alerts.sort(key=lambda a: -abs(a["z"]))
    return {
        "fields": per_field,
        "daily": daily(clips, [f for f in fields if f in clips.columns]),
        "alerts": alerts,
        "n_alerts": len(alerts),
        "n_clips": int(len(clips)),
        "note": ("발송하지 않는다 — 알림 대상 목록만 낸다. 문턱은 필드마다 "
                 "다르고(z 열), 편차는 center 집단 안에서 잰 것이다."),
    }


def daily(clips: pd.DataFrame, fields: list[str]) -> list:
    """날짜별 평균과 관측 수 — 화면의 추이 그래프가 이걸 그린다.

    날짜가 없으면 **빈 목록**을 돌려준다. 순서를 지어내지 않는다 — 클립
    순서를 시간축으로 쓰면 없는 추세가 그려진다.
    """
    if "date" not in clips.columns:
        return []
    g = clips.dropna(subset=["date"])
    if g.empty:
        return []
    rows = []
    for d, h in g.groupby(g["date"].astype(str)):
        row = {"date": d, "n": int(len(h))}
        for f in fields:
            v = h[f].dropna() if f in h.columns else pd.Series(dtype=float)
            row[f] = None if v.empty else round(float(v.mean()), 2)
        rows.append(row)
    return sorted(rows, key=lambda r: r["date"])


def main() -> int:
    args = sys.argv[1:]
    clips_csv = label_dir = None
    synthetic = "--synthetic" in args
    if "--clips" in args:
        clips_csv = args[args.index("--clips") + 1]
    elif args and not args[0].startswith("--"):
        label_dir = args[0]

    if synthetic:
        import tempfile
        # 프레임 수다. 한 클립이 프레임 여러 장이라 접으면 20분의 1 로 줄고,
        # MIN_N(100 클립)에 못 미치면 전 필드가 '표본부족' 으로 나온다.
        n = int(args[args.index("--n") + 1]) if "--n" in args else 6000
        tmp = tempfile.mkdtemp(prefix="env_synth_")
        parse_aihub.generate_synthetic_71763(tmp, n=n)
        label_dir = tmp

    if clips_csv:
        clips = pd.read_csv(clips_csv)
    elif label_dir:
        frames = parse_aihub.parse_71763(label_dir)
        clips = parse_aihub.aggregate_71763_clips(frames)
    else:
        print(__doc__.split("실행:")[1])
        return 2

    r = analyze(clips)
    r["source"] = "합성(배관 검증용 — 실자료 아님)" if synthetic else \
        (clips_csv or label_dir)

    print("클립 %d · 알림 %d건" % (r["n_clips"], r["n_alerts"]))
    print("%-12s %6s %8s %8s %6s %7s %s"
          % ("필드", "n", "기준", "설명력%", "z", "알림%", "판정"))
    for f, d in r["fields"].items():
        if "z" not in d:
            print("%-12s %6d %s" % (d["label"], d.get("n", 0), d["usable"]))
            continue
        print("%-12s %6d %8s %8.1f %6.1f %6.1f%% %s"
              % (d["label"], d["n"], d["center"], d["center_explains_pct"],
                 d["z"], d["flagged_pct"], d["usable"]))
        print("             사분위 경계 %s · %s"
              % (d["quartile_cuts"],
                 " ".join("%s %s건(%.1f%%)" % (q["q"], q["n_flagged"],
                                               q["flagged_pct"])
                          for q in d["by_quartile"])))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    json.dump(r, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # 알림이 0건이어도 머리글은 남긴다 — 빈 파일은 "한 건도 없었다" 와
    # "아직 안 돌렸다" 가 구별되지 않는다.
    cols = ["field", "label", "date", "chamber", "value", "dev", "z",
            "quartile", "direction"]
    pd.DataFrame(r["alerts"], columns=cols).to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n%s\n%s" % (OUT_JSON, OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
