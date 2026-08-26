"""71763 생체지표 **평균 0 기준표** — 사양관리용 이상치 문턱.

절대값을 쓰지 않는 이유가 있다. 라벨의 `rectal-temperature` 는 중앙 37.8℃ 로
돼지 정상 직장온도(38.5~39.5℃)와 맞지 않고, 등·목·머리 온도와의 상관도
0.27 / 0.04 / -0.17 이라 일관된 체표 측정으로도 설명되지 않는다. 절대 보정을
믿을 수 없다는 뜻이다.

그러나 센서가 **내부적으로 일관되기만 하면 편차는 유효하다.** 그래서 이 모듈은
성장단계 평균을 0 으로 두고, 로버스트 산포(1.4826×MAD)로 나눈 z 만 다룬다.

문턱은 필드마다 따로 잡는다. 전 필드에 3σ 를 일괄로 물리면 back_temp 는 0건,
latent_heat 는 10.2% 로 문턱이 죽는다 — 편차가 정규분포가 아니기 때문이다.
FLAG_BAND(0.5~5%) 안에 들어오는 z 를 필드마다 찾는다.

왜 성장단계 평균인가 — 개체 평균으로 중심화하면 개체내 산포가 너무 작아
(직장온도 0.36℃) 3σ 문턱에 20~30% 가 걸린다. 성장단계 기준이면 1~4% 로
현장에서 감당할 빈도가 된다.

**이것은 탐지 '모델' 이 아니라 '기준표' 다.** 편차를 환경으로 예측하려는 시도는
전부 실패했다(R² 전 항목 음수). 개체 안에서 환경이 변하지 않기 때문이다 —
피처마다 개체당 고유값의 중앙이 1 이다. 자세한 근거는 docs/AIHUB_71763.md 2절.

실행:
    python competition/src/bio_baseline_71763.py --clips clips.csv
    python competition/src/bio_baseline_71763.py <라벨디렉터리>
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_aihub  # noqa: E402

#: 기준표를 만들 생체지표. 절대값이 아니라 편차로만 쓴다.
BIO_FIELDS = ["breath_rate", "rectal_temp", "back_temp", "neck_temp",
              "head_temp", "sensible_heat", "latent_heat"]
#: 중심화 기준. 개체 평균은 산포가 너무 작아 문턱이 잡히지 않는다.
CENTER_BY = "pig_class"
#: 로버스트 산포 환산 계수 (정규분포에서 MAD → sd)
MAD_TO_SD = 1.4826
#: 기본 문턱(로버스트 sd 배수). 필드에 따라 아래 대역으로 조정된다.
Z_THRESHOLD = 3.0
#: 목표 알림률 대역(%). 현장에서 감당할 빈도이자, 문턱이 살아 있다는 증거다.
#  전 필드에 3σ 를 일괄로 물리면 분포 모양에 따라 발화가 0건이 되거나
#  (back_temp) 10% 를 넘는다(latent_heat). 그래서 필드마다 이 대역에
#  들어오는 z 를 찾는다. 못 찾으면 그 필드는 문턱을 쓰지 않는다.
FLAG_BAND = (0.5, 5.0)
#: z 탐색 격자. 2.0 미만은 정상 개체를 너무 많이 부르고, 6.0 초과는 사실상 무발화다.
Z_GRID = [round(2.0 + 0.1 * i, 1) for i in range(41)]
#: 분위 문턱에서 한쪽 꼬리로 떼어낼 비율(%). 양쪽이므로 알림률은 2배가 된다.
#  z 문턱이 전량 데이터에서 전 필드 0건이 된 뒤의 대안이다 — 근거는 모듈 상단.
TAIL_PCT = 2.5
#  **왜 0건이었는지는 자료 사고가 아니라 분포의 성질이다.** 균등분포에서는
#  robust_sd = 1.4826×MAD ≈ 0.37×범위 이고 최대 편차가 0.5×범위 라서
#  **최대 |z| 가 1.35 를 못 넘는다**(정규분포는 4 를 넘는다). Z_GRID 가
#  2.0 에서 시작하므로 균등에 가까운 분포는 어떤 z 로도 한 건도 안 걸린다.
#  71763 의 환경값은 실험 설계 격자라 거의 균등이고, 생체지표도 챔버·성장
#  단계로 통제돼 꼬리가 얇다. **꼬리가 없으면 잡을 이상도 없다** — 그래서
#  분위 문턱이 부르는 5% 는 '이상'이 아니라 '이 집단의 끝'이다.


def robust_sd(s: pd.Series) -> float:
    """1.4826 × MAD. 이상치에 끌려가지 않는 산포."""
    s = s.dropna()
    if len(s) < 2:
        return float("nan")
    return float(MAD_TO_SD * (s - s.median()).abs().median())


def build_baseline(clips: pd.DataFrame,
                   fields: list[str] | None = None) -> pd.DataFrame:
    """성장단계별 평균 0 기준표를 만든다.

    반환 열: field, center, n, robust_sd, p1/p5/p50/p95/p99(편차),
             n_flagged, flagged_pct, usable
    `usable` 은 문턱이 실제로 쓸모 있는지다. 알림률이 FLAG_BAND 밖이면
    '문턱무의미'(너무 적음) 또는 '알림과다'(너무 많음)로 표시된다 —
    필드마다 z 를 조정하고도 대역에 못 들어오는 필드가 그렇게 드러난다.
    """
    fields = fields or BIO_FIELDS
    rows = []
    for c in fields:
        if c not in clips.columns:
            continue
        g = clips.dropna(subset=[c])
        if len(g) < 100 or CENTER_BY not in g.columns:
            continue
        dev = g.groupby(CENTER_BY)[c].transform(lambda s: s - s.mean())
        rsd = robust_sd(dev)
        if not rsd or rsd != rsd:
            continue
        z, pct = fit_threshold(dev, rsd)
        flag = dev.abs() > z * rsd
        lo, hi = FLAG_BAND
        rows.append({
            "field": c, "center": CENTER_BY, "n": len(g),
            "robust_sd": round(rsd, 3),
            "z": z,
            "p1": round(dev.quantile(.01), 2),
            "p5": round(dev.quantile(.05), 2),
            "p50": round(dev.median(), 2),
            "p95": round(dev.quantile(.95), 2),
            "p99": round(dev.quantile(.99), 2),
            "n_flagged": int(flag.sum()), "flagged_pct": round(pct, 1),
            "usable": "쓸만함" if lo <= pct <= hi else (
                "문턱무의미" if pct < lo else "알림과다"),
        })
    return pd.DataFrame(rows)


def fit_threshold(dev: pd.Series, rsd: float) -> tuple:
    """필드마다 목표 알림률 대역에 들어오는 z 를 찾는다.

    전 필드에 3σ 를 일괄로 물리면 분포 모양에 따라 문턱이 죽는다 — 실측으로
    back_temp 는 0.0%(한 건도 안 걸림), latent_heat 는 10.2%(알림 과다)였다.
    편차가 정규분포가 아니기 때문이고, 필드를 더한다고 나아지지 않는다.

    대역 한가운데에 가장 가까운 z 를 고른다. 어떤 z 로도 대역에 못 들어오면
    가장 가까운 z 를 돌려주되 `usable` 이 그 사실을 드러낸다 — 조용히
    넘어가지 않는다.
    """
    lo, hi = FLAG_BAND
    mid = (lo + hi) / 2
    best = None
    for z in Z_GRID:
        pct = float((dev.abs() > z * rsd).mean() * 100)
        score = (0 if lo <= pct <= hi else 1, abs(pct - mid))
        if best is None or score < best[0]:
            best = (score, z, pct)
    return best[1], best[2]


def build_baseline_quantile(clips: pd.DataFrame,
                           fields: list[str] | None = None,
                           tail_pct: float = TAIL_PCT) -> pd.DataFrame:
    """분위 문턱 기준표 — 편차 상·하위 `tail_pct`% 를 잘라낸다.

    z 문턱과 결정적으로 다른 점: **문턱이 반드시 잡힌다.** 분포 모양과
    무관하게 알림률이 2×tail_pct 로 고정되기 때문이다. 전량 데이터에서
    z 방식이 7개 필드 전부 0건이 된 뒤의 대안이 이것이다.

    대가가 있다. z 문턱은 "산포 대비 몇 배나 벗어났는가"를 물어 정말로
    이상한 게 없으면 0건을 낸다. 분위 문턱은 정상만 있어도 늘 5% 를 부른다 —
    **'이상'이 아니라 '이 집단에서 가장 끝'이라는 뜻으로만 읽어야 한다.**
    그래서 cut_lo/cut_hi 옆에 sd_ratio(문턱이 robust_sd 의 몇 배인가)를 같이
    낸다. 이 값이 2 보다 한참 작으면 잘라낸 끝값이 산포 안에 있다는 뜻이고,
    그 필드의 알림은 우선순위를 낮춰야 한다.

    반환 열: field, center, n, robust_sd, cut_lo, cut_hi, sd_ratio,
             p50, n_flagged, flagged_pct
    """
    fields = fields or BIO_FIELDS
    rows = []
    for c in fields:
        if c not in clips.columns:
            continue
        g = clips.dropna(subset=[c])
        if len(g) < 100 or CENTER_BY not in g.columns:
            continue
        dev = g.groupby(CENTER_BY)[c].transform(lambda s: s - s.mean())
        rsd = robust_sd(dev)
        lo = float(dev.quantile(tail_pct / 100))
        hi = float(dev.quantile(1 - tail_pct / 100))
        flag = (dev < lo) | (dev > hi)
        # 문턱이 산포 대비 어디쯤인가. 작을수록 '끝'일 뿐 '이상'이 아니다.
        ratio = (min(abs(lo), abs(hi)) / rsd) if rsd else float("nan")
        rows.append({
            "field": c, "center": CENTER_BY, "n": len(g),
            "robust_sd": round(rsd, 3),
            "cut_lo": round(lo, 2), "cut_hi": round(hi, 2),
            "sd_ratio": round(ratio, 2),
            "p50": round(dev.median(), 2),
            "n_flagged": int(flag.sum()),
            "flagged_pct": round(float(flag.mean() * 100), 1),
        })
    return pd.DataFrame(rows)


def variance_split(clips: pd.DataFrame,
                   fields: list[str] | None = None) -> pd.DataFrame:
    """개체 간 / 개체 내 분산 분해 — 편차에 신호가 남는지 먼저 확인한다.

    개체 간 차이가 전부라면 편차는 잡음뿐이고 기준표를 만들 이유가 없다.

    제곱합으로 가른다(SS_total = SS_within + SS_between). 개체내%와 개체간%가
    반드시 100 이 되므로 표 안에서 서로 모순될 수 없다. ICC 를 따로 쓰지
    않는 이유가 여기 있다 — 개체 평균의 표본분산으로 ICC 를 내면 가중이
    달라져 개체내%와 합이 1 이 안 되고, 그 불일치가 표를 못 믿게 만든다.
    """
    fields = fields or BIO_FIELDS
    if "chamber" not in clips.columns or "pig_number" not in clips.columns:
        return pd.DataFrame()
    ind = (clips["chamber"].astype(str) + "|"
           + clips["pig_number"].astype(str))
    rows = []
    for c in fields:
        if c not in clips.columns:
            continue
        g = clips.assign(_ind=ind).dropna(subset=[c])
        if len(g) < 100:
            continue
        ss_tot = float(((g[c] - g[c].mean()) ** 2).sum())
        if not ss_tot:
            continue
        within = g.groupby("_ind")[c].transform(lambda s: s - s.mean())
        ss_w = float((within ** 2).sum())
        rows.append({
            "field": c, "n": len(g),
            "n_groups": int(g["_ind"].nunique()),
            "sd_total": round(g[c].std(), 2),
            "sd_within": round(within.std(), 2),
            "within_pct": round(ss_w / ss_tot * 100, 0),
            "between_pct": round((ss_tot - ss_w) / ss_tot * 100, 0),
        })
    return pd.DataFrame(rows)


def score(value: float, pig_class: str, field: str,
          baseline: pd.DataFrame, centers: pd.DataFrame) -> float | None:
    """현장 측정값 하나를 z 로 바꾼다.

    문턱은 필드마다 다르다 — baseline 의 `z` 열과 비교할 것. 3 을 일괄로
    쓰면 back_temp 는 한 건도 안 걸리고 latent_heat 는 10% 가 걸린다.
    """
    b = baseline[baseline["field"] == field]
    m = centers[(centers["field"] == field)
                & (centers[CENTER_BY] == pig_class)]
    if b.empty or m.empty:
        return None
    rsd = float(b["robust_sd"].iloc[0])
    if not rsd:
        return None
    return (value - float(m["mean"].iloc[0])) / rsd


def centers_table(clips: pd.DataFrame,
                  fields: list[str] | None = None) -> pd.DataFrame:
    """성장단계별 평균 — 편차를 되돌리거나 현장값을 채점할 때 쓴다."""
    fields = fields or BIO_FIELDS
    rows = []
    for c in fields:
        if c not in clips.columns or CENTER_BY not in clips.columns:
            continue
        for cls, g in clips.dropna(subset=[c]).groupby(CENTER_BY):
            rows.append({"field": c, CENTER_BY: cls, "n": len(g),
                         "mean": round(g[c].mean(), 3),
                         "median": round(g[c].median(), 3)})
    return pd.DataFrame(rows)


def report_quantile(clips: pd.DataFrame) -> int:
    """분위 문턱 경로. z 문턱이 전량 데이터에서 죽었을 때 쓰는 대안이다."""
    b = build_baseline_quantile(clips)
    print()
    print("=== 분위 문턱 기준표 (성장단계 중심화 · 상하위 "
          + str(TAIL_PCT) + "% 절단) ===")
    print(b.to_string(index=False))
    # 알림률은 설계상 2×TAIL_PCT 로 고정이다. 볼 것은 문턱이 산포 대비 어디냐다.
    for _, r in b[b["sd_ratio"] < 2.0].iterrows():
        print("  ⚠️ " + str(r["field"]) + ": 문턱이 robust_sd 의 "
              + str(r["sd_ratio"]) + "배뿐 — '이상'이 아니라 "
              "'이 집단의 끝'으로 읽을 것")
    os.makedirs("competition/outputs", exist_ok=True)
    out = "competition/outputs/71763_baseline_quantile.csv"
    b.to_csv(out, index=False, encoding="utf-8-sig")
    centers_table(clips).to_csv("competition/outputs/71763_centers.csv",
                                index=False, encoding="utf-8-sig")
    print()
    print("→ " + out + " · competition/outputs/71763_centers.csv")
    print("주의: 분위 문턱은 정상만 있어도 항상 "
          + str(2 * TAIL_PCT) + "% 를 부른다. 알림은 '이상'이 아니라")
    print("      '이 성장단계에서 가장 끝'이라는 뜻이다 — sd_ratio 를 함께 볼 것.")
    return 0


def main() -> int:
    args = sys.argv[1:]
    quantile = "--quantile" in args
    if quantile:
        args.remove("--quantile")
    clips_csv = None
    if "--clips" in args:
        i = args.index("--clips")
        clips_csv = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]

    if clips_csv:
        clips = pd.read_csv(clips_csv)
    else:
        label_dir = args[0] if args else None
        if not (label_dir and os.path.isdir(label_dir)):
            print("사용: bio_baseline_71763.py <라벨디렉터리> | --clips clips.csv")
            return 1
        clips = parse_aihub.aggregate_71763_clips(
            parse_aihub.parse_71763(label_dir))

    # VL_A/VL_B 는 VL 의 분할 사본이라 그대로 두면 검증셋을 두 번 센다.
    if "split" in clips.columns and (clips["split"] == "VL").any():
        clips = clips[~clips["split"].isin(["VL_A", "VL_B"])]
    clips = clips[clips.get("modality") == "호흡량"]
    print(f"호흡량 클립 {len(clips)}개 · 챔버×개체 "
          f"{clips.groupby(['chamber', 'pig_number']).ngroups}조합\n")

    v = variance_split(clips)
    print("=== 분산분해 — 편차에 신호가 남는가 ===")
    print(v.to_string(index=False))
    weak = v[v["within_pct"] < 25]["field"].tolist()
    if weak:
        print(f"  ⚠️ 개체내 비중 25% 미만이라 편차가 잡음에 가까운 필드: "
              f"{', '.join(weak)}")

    if quantile:
        return report_quantile(clips)

    b = build_baseline(clips)
    print("\n=== 평균 0 기준표 (성장단계 중심화 · 필드별 z 자동 조정) ===")
    print(b.to_string(index=False))
    for _, r in b[b["usable"] != "쓸만함"].iterrows():
        print(f"  ⚠️ {r['field']}: {r['usable']} "
              f"({r['n_flagged']}건 / {r['flagged_pct']}%)")

    os.makedirs("competition/outputs", exist_ok=True)
    b.to_csv("competition/outputs/71763_baseline_z.csv",
             index=False, encoding="utf-8-sig")
    centers_table(clips).to_csv("competition/outputs/71763_centers.csv",
                                index=False, encoding="utf-8-sig")
    print("\n→ competition/outputs/71763_baseline_z.csv · 71763_centers.csv")
    print("\n주의: 이 기준표는 챔버 4개·챔버×개체 89조합(고유 개체번호 71)에서 나왔다. 실농장에 쓰려면")
    print("      농장별로 다시 중심화해야 한다 — sd 를 그대로 가져다 쓰지 말 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
