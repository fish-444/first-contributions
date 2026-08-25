"""돈사 환경 −1~1 편차 스케일 + 지침 위험 표시 — 71763 클립 CSV 를 먹는다.

돈사(챔버)마다 센서 설치 위치·보정이 달라 **절대값을 돈사끼리 비교하면
센서 차이를 사육환경 차이로 착각한다.** 그래서 센서값을 돈사·센서별 자기
기준선의 편차로 바꾸되, 눈금이 읽히도록 **−1~1** 로 접는다:

    scaled = z / cut          (z = (값−중앙값)/(IQR/1.349), cut = 자기 이력
                               경보율 0.5~5% 역산 컷)

**±1 이 곧 그 돈사의 경보 경계다.** −1~1 안이면 평소 범위, 넘으면 `주의`.
0.8 은 "경보 경계의 8할까지 왔다"로 읽힌다 — 눈금 자체가 뜻을 갖는다.
경계를 −1~1 로 두려고 값을 우겨 넣는 시그모이드·최소최대 정규화를 쓰지
않는 이유이기도 하다: 그런 변환은 경계에 뜻이 없다.

**문턱을 발명하지 않았다.** z 산포는 `behavior_baseline._robust`(IQR),
컷은 `_calibrate_cut`(자기 이력 경보율 역산) — `barn_env_control.baseline`
을 그대로 부른다. 같은 문제에 같은 답이고, 식을 두 벌 두면 갈린다.

## 위험 층 — 사육 설명서(지침)의 절대 대역

편차는 "평소와 다른가"만 안다. "위험한가"는 지침이 정한다:
성장단계별 적온·습도 대역과 NH₃ 15ppm — 정본은
`barn_env_control.TEMP_GUIDE / RH_GUIDE / NH3_LIMIT`(국립축산과학원
「환절기 돼지 사양관리」 + 제공 지침표)이고 여기서 다시 적지 않는다.

두 층이 어긋날 때가 정보다: **지침 위반인데 편차가 평소 수준(|scaled|<1)
이면 "센서 치우침·교정 확인"** — 그 돈사는 늘 그 값이었다는 뜻이라,
환경이 나빠진 게 아니라 센서가 치우쳤을 공산이 크다.

## 한계 (미리 적는다)

- 71763 의 온·습·환기는 **4개 챔버의 실험 설계값**이다. 전수 파싱
  3,944클립의 실측 분포가 그것을 말한다(71763_PARSE 4절):

        temp_c    24.5 ~ 27.1  (폭 2.6℃)
        humidity  16.1 ~ 35.8 %
        nh3_ppm    7.2 ~  9.3
        ventilation 1.3 ~  2.8

  실농장이면 온도가 계절·주야로 20℃ 이상 흔들리고 습도는 통상 50~80% 다.
  **여기서 뽑은 σ 로 문턱을 잡아 농장에 옮기면 거의 모든 시점이 이상으로
  찍힌다** — 챔버 2.6℃ 산포가 분모이기 때문이다. 이 모듈의 값어치는
  수치가 아니라 **배선**이다: 농장 센서 로그가 오면 같은 명령이 그대로
  돈다. 아래 `span_note` 가 이 상황을 스스로 찍는다.
- pig_class → 지침 단계 대응은 **체중 기준 우리 해석**이다(조문·지침에
  같은 낱말이 없다). 출력이 그 사실을 달고 다닌다.

실행:
    python competition/src/env_scale.py --clips <출력>/clips_71763.csv
    python competition/src/env_scale.py --clips barn_log.csv --key barn
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import barn_env_control as bec  # noqa: E402

#: 스케일 대상 센서. 지침 sensor 키가 None 이면 편차 층만 본다(지침 대역 없음).
SENSORS = {
    "temp_c": "temp_c",
    "humidity_pct": "rh_pct",
    "nh3_ppm": "nh3_ppm",
    "co2_ppm": None,
    "ventilation": None,
}
#: 71763 pig_class → 지침 성장단계. **체중 기준 우리 해석**이다 — 라벨의
#  체중 중앙(이유 17.7 / 33.2 / 59.8 / 88.5kg)을 지침표의 구간에 맞췄다.
STAGE_MAP = {
    "weaningpig": "이유자돈",
    "piglet": "육성초기",
    "growing-pig": "육성후기",
    "porker": "비육돈",
}


def span_note(values: pd.Series, stage: str, gsensor: str,
              guide: dict) -> str | None:
    """이력 폭이 **지침 대역보다 좁으면** 실농장 센서가 아니라고 말한다.

    한 돈사의 온도 이력 전체가 적온 대역(예: 비육돈 16~21℃, 폭 5℃)보다
    좁게 움직인다면, 그건 계절도 주야도 없는 곳이다 — 통제 챔버이거나
    센서가 굳은 것이다. 어느 쪽이든 **그 산포로 잡은 문턱을 농장에 옮기면
    안 된다.**

    자를 여기서 만들지 않았다. 비교 대상이 지침 대역 폭 그 자체다 —
    "농장이 정상으로 오갈 수 있다고 지침이 인정한 폭"보다도 안 움직였다는
    뜻이라, 이 비교에는 발명한 상수가 없다.

    71763 이 정확히 이 경우다: temp_c 폭 2.6℃ < 적온 대역 폭 5℃.
    """
    if gsensor not in ("temp_c", "rh_pct") or not isinstance(stage, str):
        return None
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return None
    band = guide["temp"] if gsensor == "temp_c" else guide["rh"]
    lo, hi = band.get(stage, band["임신돈·웅돈"])
    span, width = float(x.max() - x.min()), float(hi - lo)
    if span >= width:
        return None
    unit = "℃" if gsensor == "temp_c" else "%"
    return (f"이력 폭 {span:.1f}{unit} < 지침 대역 폭 {width:.1f}{unit} — "
            "통제 환경이거나 센서가 굳었다. 이 산포로 잡은 문턱을 농장에 "
            "옮기지 말 것")


def scale_key(values: pd.Series) -> dict:
    """한 돈사·한 센서의 값들 → {center, spread, cut, scaled}.

    기준선 미형성(이력 부족·산포 없음)이면 scaled 는 전부 NaN — 0 으로
    채우지 않는다. 0 은 "평소 그대로"라는 주장인데 근거가 없다.
    """
    x = pd.to_numeric(values, errors="coerce")
    b = bec.baseline(x.dropna().tolist())
    out = {"n": b["n"], "formed": b["formed"], "center": b["center"],
           "spread": b["spread"], "cut": b["cut"],
           "scaled": pd.Series(np.nan, index=values.index)}
    if not b["formed"] or b["cut"] is None:
        out["why"] = ("이력 부족 — 기준선 미형성" if not b["formed"]
                      else "산포 없음 — 컷을 못 만든다(상수 센서)")
        return out
    out["scaled"] = (x - b["center"]) / b["spread"] / b["cut"]
    return out


def run(df: pd.DataFrame, key: str = "chamber") -> pd.DataFrame:
    """클립 표 → 행마다 센서별 scaled(−1~1 눈금) + 위험/주의/센서점검.

    반환 열: `<센서>_scaled`(±1=자기 경보 경계) · `<센서>_flag`
    (위험=지침 위반 / 주의=|scaled|≥1 / 센서점검=위반인데 편차 평소 /
    빈 값=이상 없음 또는 판정 불가).
    """
    d = df.copy()
    d["_stage"] = d.get("pig_class", pd.Series(index=d.index)).map(STAGE_MAP)
    guide = {"temp": bec.TEMP_GUIDE, "rh": bec.RH_GUIDE,
             "nh3": bec.NH3_LIMIT, "h2s": bec.H2S_LIMIT}
    spans: dict = {}
    for col, gsensor in SENSORS.items():
        if col not in d.columns:
            continue
        d[col + "_scaled"] = np.nan
        d[col + "_flag"] = ""
        for k, g in d.groupby(key, dropna=False):
            s = scale_key(g[col])
            d.loc[g.index, col + "_scaled"] = s["scaled"].round(3)
            if gsensor:
                st = g["_stage"].dropna()
                note = span_note(g[col], st.iloc[0] if len(st) else None,
                                 gsensor, guide)
                if note:
                    spans[(k, col)] = note
            for i in g.index:
                val = pd.to_numeric(g.at[i, col], errors="coerce")
                if pd.isna(val):
                    continue
                sc = s["scaled"].get(i, np.nan)
                over_dev = bool(np.isfinite(sc) and abs(sc) >= 1.0)
                state = "적정"
                if gsensor is not None and isinstance(d.at[i, "_stage"], str):
                    state, _band = bec._guide_state(
                        d.at[i, "_stage"], gsensor, float(val), guide)
                if state != "적정":
                    # 지침 위반인데 편차가 평소면 센서 쪽을 먼저 의심한다
                    d.at[i, col + "_flag"] = (
                        f"위험({state})" if over_dev or not np.isfinite(sc)
                        else f"위험({state})·센서 치우침·교정 확인")
                elif over_dev:
                    d.at[i, col + "_flag"] = "주의(평소 범위 밖)"
    d.attrs["span_notes"] = spans
    return d


def summary(d: pd.DataFrame, key: str = "chamber") -> list:
    """돈사별 요약 — 표시된 행만 센다."""
    rows = []
    for k, g in d.groupby(key, dropna=False):
        row = {key: k, "n": len(g)}
        for col in SENSORS:
            f = col + "_flag"
            if f in g.columns:
                row[col + "_위험"] = int(g[f].str.startswith("위험").sum())
                row[col + "_주의"] = int((g[f] == "주의(평소 범위 밖)").sum())
        rows.append(row)
    return rows


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--key", default="chamber")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    d = run(pd.read_csv(a.clips, encoding="utf-8-sig"), key=a.key)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.clips)),
                                "env_scaled.csv")
    d.to_csv(out, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("  돈사 환경 −1~1 편차 + 지침 위험 표시")
    print("=" * 72)
    print("  ±1 = 그 돈사 자기 이력의 경보 경계(경보율 0.5~5% 역산) — "
          "돈사 간 비교는 이 눈금으로만 한다")
    for row in summary(d, key=a.key):
        parts = [f"{c.rsplit('_', 1)[0]} 위험{row[c]}"
                 for c in row if c.endswith("_위험") and row[c]]
        parts += [f"{c.rsplit('_', 1)[0]} 주의{row[c]}"
                  for c in row if c.endswith("_주의") and row[c]]
        print(f"  {a.key}={row[a.key]}  n={row['n']}  "
              + (" · ".join(parts) if parts else "표시 없음"))
    for (k, col), note in (d.attrs.get("span_notes") or {}).items():
        print(f"  ⚠ {a.key}={k} · {col}: {note}")
    print(f"\n  → {out}")
    print("  ⚠ 지침 정본: barn_env_control(TEMP_GUIDE·RH_GUIDE·NH3 15ppm)")
    print("  ⚠ pig_class→단계 대응은 체중 기준 우리 해석이다")
    print("  ⚠ 71763 환경값은 챔버 실험 설계값 — 이 수치는 배선 확인용이고 "
          "농장 정상 범위가 아니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
