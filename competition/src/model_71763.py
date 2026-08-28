"""71763 양돈 생체 에너지 — 파싱→모델링 실행 스크립트.

환경센서(온도·습도·CO2·NH3) + 개체온도 + 체중 + 사양관리로부터 생체에너지
지표(호흡수·현열량·잠열량·증발량)를 예측하는 회귀 베이스라인.

**분석 단위는 프레임이 아니라 클립이다.** 실라벨은 한 클립(폴더) 안에서
타깃과 환경·개체 값이 전부 상수이고, 프레임마다 변하는 것은 키포인트 거리와
호흡국면(inspiratory/expiratory)뿐이다. 프레임 행(40만)으로 회귀를 돌리면
같은 답을 수십 번 세는 것이라 R² 가 부풀려진다. 그래서 여기서는 클립으로
접어서 학습하고, --frames 를 주면 부풀려진 쪽 수치도 나란히 찍어 대조한다.

모달리티가 둘이고 타깃이 다르다.
    호흡량(pig)   : breath_rate · sensible_heat · latent_heat   (개체 있음)
    증발량(floor) : evaporation · sensible_heat · latent_heat   (개체 **없음**)

실행:
    python competition/src/model_71763.py <라벨디렉터리>       # 실데이터
    python competition/src/model_71763.py --clips clips.csv    # 이미 접어 둔 표
    python competition/src/model_71763.py                      # 합성 시연
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_aihub  # noqa: E402
import train as train_mod  # noqa: E402  (make_pipeline 재사용)

#: 모달리티별 타깃. 세 개가 겹치는 것이 아니라 첫 항목만 서로 다르다.
TARGETS_BY_MODALITY = {
    "호흡량": ["breath_rate", "sensible_heat", "latent_heat"],
    "증발량": ["evaporation", "sensible_heat", "latent_heat"],
}
#: 피처에서 반드시 빼야 하는 것 — 식별자, 타깃, 그리고 집계 부산물.
ID_COLS = ["split", "modality", "chamber", "clip_id", "frame_file",
           "timestamp", "video_category", "videoid", "pig_number",
           "date", "time", "measure_date", "measure_time", "n_frames",
           "breathing_type"]
ALL_TARGETS = parse_aihub.TARGETS_71763


def dedup_splits(df: pd.DataFrame) -> pd.DataFrame:
    """VL_A / VL_B 는 VL 을 둘로 나눈 것이라 그대로 두면 검증셋을 두 번 센다.

    실측: VL_A ∪ VL_B == VL, VL_A ∩ VL_B == ∅ 이고 해당 파일들의 md5 가 VL 과
    바이트 단위로 같다(40,374개). 서로 다른 라벨러의 이중 어노테이션이
    아니므로 라벨 신뢰구간을 얻을 수도 없다. VL 만 남긴다.
    """
    if "split" not in df.columns:
        return df
    if (df["split"] == "VL").any() and df["split"].isin(["VL_A", "VL_B"]).any():
        n0 = len(df)
        df = df[~df["split"].isin(["VL_A", "VL_B"])].copy()
        print(f"  VL_A/VL_B 는 VL 의 분할 사본 — 제거: {n0} → {len(df)}클립")
    return df


def load(label_dir: str | None, clips_csv: str | None) -> pd.DataFrame:
    """클립 단위 테이블을 만든다(또는 읽는다)."""
    if clips_csv:
        df = pd.read_csv(clips_csv)
        print(f"클립 표 읽음: {clips_csv} → {len(df)}클립")
        return dedup_splits(df)
    if label_dir and os.path.isdir(label_dir):
        frames = parse_aihub.parse_71763(label_dir)
        print(f"실데이터 파싱: {label_dir} → 프레임 {len(frames)}행")
    else:
        tmp = tempfile.mkdtemp()
        parse_aihub.generate_synthetic_71763(tmp, n=3000)
        frames = parse_aihub.parse_71763(tmp)
        print(f"(합성 시연) 프레임 {len(frames)}행 — "
              f"실행: model_71763.py <라벨디렉터리>")
    clips = parse_aihub.aggregate_71763_clips(frames)
    print(f"클립으로 접음: {len(frames)}프레임 → {len(clips)}클립")
    _report_constancy(clips)
    return clips


def _report_constancy(clips: pd.DataFrame) -> None:
    """클립 내 상수 가정이 깨진 필드를 먼저 보고한다(부정 결과를 앞에)."""
    broken = {c[:-6]: int((clips[c] > 1).sum())
              for c in clips.columns
              if c.endswith("_nuniq") and (clips[c] > 1).any()}
    if broken:
        print("  ⚠️ 클립 안에서 값이 하나가 아닌 필드 — 상수 가정이 깨졌다:")
        for k, v in sorted(broken.items(), key=lambda x: -x[1]):
            print(f"       {k}: {v}클립")
    else:
        print("  클립 내 상수 가정: 모든 필드에서 유지됨")


def features(sub: pd.DataFrame) -> list[str]:
    drop = set(ID_COLS) | set(ALL_TARGETS)
    return [c for c in sub.columns
            if c not in drop and not c.endswith("_nuniq")
            and sub[c].notna().any() and sub[c].nunique(dropna=True) > 1]


def evaluate(sub: pd.DataFrame, target: str, tag: str,
             groups: np.ndarray | None) -> dict | None:
    """GroupKFold 로 교차검증하고 기준선과 나란히 보고한다."""
    y = sub[target].to_numpy(dtype=float)
    feats = features(sub)
    num = [c for c in feats if pd.api.types.is_numeric_dtype(sub[c])]
    cat = [c for c in feats if c not in num]
    if not feats or len(y) < 30:
        print(f"  [{target}] 표본 {len(y)} · 피처 {len(feats)} — 건너뜀")
        return None
    X = sub[num + cat]

    pipe = train_mod.make_pipeline(
        num, cat, GradientBoostingRegressor(random_state=42))

    n_groups = len(set(groups)) if groups is not None else 0
    if groups is not None and n_groups >= 5:
        cv, cvname = GroupKFold(n_splits=5), f"GroupKFold({n_groups}그룹)"
        pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cvname = f"KFold — 그룹 {n_groups}개뿐이라 누수 방지 불가"
        pred = cross_val_predict(pipe, X, y, cv=cv)

    mae = mean_absolute_error(y, pred)
    base = mean_absolute_error(y, np.full_like(y, float(np.mean(y))))
    r2 = r2_score(y, pred)
    print(f"\n  === [{tag}] {target} ({cvname}) ===")
    print(f"    기준선(평균 예측)  MAE {base:.3f}          ← 먼저 본다")
    print(f"    모델              MAE {mae:.3f}  R² {r2:.3f}"
          f"   ({1 - mae / base:+.1%})")
    print(f"    표본 {len(y)}클립 · 피처 {len(feats)}개 · "
          f"타깃 범위 {y.min():.2f}~{y.max():.2f}")
    if mae >= base:
        print("    ⚠️ 평균만 찍는 것보다 못하다 — 이 타깃은 이 피처로 설명되지 않는다.")
    if n_groups < 5:
        print("    ⚠️ 그룹이 5개 미만이라 개체/세션 누수를 막지 못했다. 수치를 믿지 말 것.")
    return {"target": target, "modality": tag, "n": len(y),
            "n_groups": n_groups, "mae": mae, "baseline_mae": base, "r2": r2}


def run(clips: pd.DataFrame) -> list[dict]:
    out = []
    for modality, targets in TARGETS_BY_MODALITY.items():
        sub_all = clips[clips.get("modality") == modality]
        if sub_all.empty:
            continue
        print(f"\n{'=' * 68}\n{modality}  —  {len(sub_all)}클립\n{'=' * 68}")
        for target in targets:
            if target not in sub_all.columns:
                continue
            sub = sub_all[sub_all[target].notna()].copy()
            if len(sub) < 30:
                print(f"  [{target}] 클립 {len(sub)}개 — 건너뜀")
                continue
            groups = parse_aihub.group_key_71763(sub).to_numpy()
            r = evaluate(sub, target, modality, groups)
            if r:
                out.append(r)
                pipe_name = f"71763_{modality}_{target}"
                try:
                    feats = features(sub)
                    num = [c for c in feats
                           if pd.api.types.is_numeric_dtype(sub[c])]
                    cat = [c for c in feats if c not in num]
                    pipe = train_mod.make_pipeline(
                        num, cat, GradientBoostingRegressor(random_state=42))
                    pipe.fit(sub[num + cat], sub[target].to_numpy(dtype=float))
                    train_mod.save_importances(pipe, num, cat, pipe_name)
                except Exception as e:  # noqa: BLE001
                    print(f"    (중요도 저장 실패: {e})")
    return out


def main() -> int:
    args = [a for a in sys.argv[1:]]
    clips_csv = None
    if "--clips" in args:
        i = args.index("--clips")
        clips_csv = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]
    label_dir = args[0] if args else None
    os.makedirs("competition/outputs", exist_ok=True)
    clips = dedup_splits(load(label_dir, clips_csv))
    res = run(clips)
    if res:
        print(f"\n{'=' * 68}\n요약 — 기준선 대비 개선율\n{'=' * 68}")
        for r in res:
            gain = 1 - r["mae"] / r["baseline_mae"]
            mark = "개선" if gain > 0.02 else ("기준선과 같음" if gain > -0.02
                                             else "기준선보다 나쁨")
            print(f"  {r['modality']:4s} {r['target']:14s} "
                  f"n={r['n']:5d} 그룹={r['n_groups']:4d} "
                  f"R²={r['r2']:+.3f} {gain:+6.1%}  [{mark}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
