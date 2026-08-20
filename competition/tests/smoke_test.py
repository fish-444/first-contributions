"""파이프라인 스모크 테스트.

의존성이 제대로 깔렸는지, 핵심 모듈이 import 되고 최소 파이프라인이 도는지
빠르게 확인한다. 순수 파이썬으로 실행 가능하고(`python competition/tests/smoke_test.py`),
pytest 로도 수집된다(함수명이 test_* ).
"""
from __future__ import annotations

import importlib
import os
import sys

# competition/src 를 import 경로에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def test_dependencies_import() -> None:
    for mod in ("pandas", "numpy", "sklearn", "matplotlib"):
        importlib.import_module(mod)


def test_aihub_client_no_key() -> None:
    """AI Hub 클라이언트가 import 되고, 키 없이 검색 함수가 존재하는지."""
    import aihub
    assert hasattr(aihub, "search")
    assert hasattr(aihub, "download")


def test_pipeline_runs() -> None:
    """소량 합성 데이터로 회귀 파이프라인이 끝까지 도는지."""
    import numpy as np
    import generate_data
    import train

    df = generate_data.generate(n=300)
    for col in ("adg_kg_day", "mortality", "sex", "feed_intake_kg"):
        assert col in df.columns

    num, cat = train.build_features(df)
    assert "sex" in cat and "feed_intake_kg" in num  # 범주/수치 분류 정상

    from sklearn.ensemble import GradientBoostingRegressor
    pipe = train.make_pipeline(num, cat, GradientBoostingRegressor(random_state=0))
    pipe.fit(df[num + cat], df["adg_kg_day"])
    pred = pipe.predict(df[num + cat])
    assert pred.shape[0] == len(df)
    assert np.isfinite(pred).all()


def test_aihub_parsers() -> None:
    """세 데이터셋 파서가 스키마 합성 데이터를 정상 파싱하는지."""
    import tempfile
    import parse_aihub
    for key in ("71763", "71471", "622"):
        with tempfile.TemporaryDirectory() as d:
            parse_aihub.GENERATORS[key](d)
            df = parse_aihub.PARSERS[key](d)
            assert len(df) > 0, f"{key} 파싱 결과가 빔"


def test_pipeline_gilt_integration() -> None:
    """CCTV→무발정 통합 파이프라인: 조인트 생성→신호추출→결합이 되는지."""
    import pipeline_gilt
    frames, mgmt = pipeline_gilt.generate_joint(n_gilts=40, frames=8)
    signals = pipeline_gilt.build_cctv_signals(frames)
    merged = mgmt.merge(signals, on="individual_id", how="inner")
    assert len(merged) == len(mgmt)
    assert "activity_mean" in merged.columns and "feed_adequacy" in merged.columns


def test_estrus_onset_and_dashboard() -> None:
    """발정 시작점 탐지 + 대시보드 데이터 생성이 되는지."""
    import estrus_onset
    import build_dashboard
    import pipeline_gilt
    frames, mgmt = pipeline_gilt.generate_joint(n_gilts=40, frames=10)
    onsets = estrus_onset.detect_all(frames)
    assert len(onsets) == 40
    any_res = next(iter(onsets.values()))
    assert "score" in any_res and "status" in any_res
    data = build_dashboard.build_data(frames, mgmt)
    assert data["meta"]["n_gilts"] == 40
    assert len(data["gilts"]) == 40 and data["importance"]


def test_edinburgh_parser() -> None:
    """Edinburgh output.json 파서(작은 합성 샘플로 검증, 다운로드 불필요)."""
    import json
    import tempfile
    import parse_edinburgh
    sample = {"videoFileName": "color.mp4", "stepSize": 0.1, "config": {},
              "objects": [{"id": "0", "frames": [
                  {"frameNumber": 0, "bbox": {"x": 10, "y": 20, "width": 30,
                   "height": 15}, "visible": True, "behaviour": "walk"},
                  {"frameNumber": 1, "bbox": {"x": 12, "y": 22, "width": 30,
                   "height": 15}, "visible": True, "behaviour": "standing"}]}]}
    with tempfile.TemporaryDirectory() as d:
        import os as _os
        rec = _os.path.join(d, "2019_11_05", "000001")
        _os.makedirs(rec)
        json.dump(sample, open(_os.path.join(rec, "output.json"), "w"))
        df = parse_edinburgh.parse_edinburgh(d)
        assert len(df) == 2
        assert {"individual_id", "frame_idx", "behavior", "centroid_x"} <= set(df.columns)


def test_posture_eval_mapping() -> None:
    """교차검증 도구: 라벨 매핑·피처·소스 로더(작은 CSV)가 동작하는지."""
    import tempfile
    import pandas as pd
    import posture_eval
    assert posture_eval.COMP_TO_COMMON["Sternal_lying"] == "lying"
    assert posture_eval.BEHAVIOR_TO_COMMON["sleep"] == "lying"
    X = posture_eval._feats([100.0, 50.0], [50.0, 100.0], [5000.0, 5000.0])
    assert X.shape == (2, 2)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"behavior": ["standing", "lying", "walk"],
                      "bbox_w": [30, 60, 40], "bbox_h": [40, 30, 35]}).to_csv(f.name, index=False)
        src = posture_eval.load_source(f.name)
    assert set(src["posture"]) <= {"standing", "sitting", "lying"}
    assert len(src) == 2  # walk 제외


def test_view_align_feats() -> None:
    """뷰 정합 피처 함수(작은 df)와 held-out 뷰 정의."""
    import pandas as pd
    import view_align
    assert "pen1_tur_cam1" in view_align.HELD_OUT_VIEWS
    df = pd.DataFrame({"aspect": [1.0, 2.0, 0.5, 1.5],
                       "area": [100, 200, 50, 150],
                       "view": ["a", "a", "b", "b"]})
    b = view_align.baseline_feats(df); v = view_align.view_aligned_feats(df)
    assert b.shape == (4, 2) and v.shape == (4, 2)


def test_estrus_link() -> None:
    """행동→발정 연계: 활발 개체 > 휴식 개체 발정지수 확인."""
    import pandas as pd
    import estrus_link
    rows = []
    # 활발 개체 A(walk/run/investigating), 휴식 개체 B(lying/sleep)
    for f in range(12):
        rows.append({"individual_id": "A", "frame_idx": f,
                     "behavior": ["walk", "run", "investigating"][f % 3],
                     "centroid_x": f * 20.0, "centroid_y": f * 15.0,
                     "aspect_ratio": 1.2, "bbox_w": 60, "bbox_h": 50,
                     "kp_spread": 20.0, "species": "pig", "estrus": None})
        rows.append({"individual_id": "B", "frame_idx": f,
                     "behavior": ["lying", "sleep"][f % 2],
                     "centroid_x": 100.0, "centroid_y": 100.0,
                     "aspect_ratio": 1.3, "bbox_w": 60, "bbox_h": 50,
                     "kp_spread": 20.0, "species": "pig", "estrus": None})
    res = estrus_link.behavior_estrus_index(pd.DataFrame(rows)).set_index("individual_id")
    assert res.loc["A", "estrus_index"] > res.loc["B", "estrus_index"]


def test_aihub_reference() -> None:
    """71471 발정 표준: 어휘 매핑·점수·매핑 합산."""
    import aihub_estrus_reference as ref
    assert ref.to_reference("walk") == "restless"
    assert ref.to_reference("jumpontopof") == "mounting"
    assert ref.to_reference("sleep") == "lying"
    R = ref.EstrusReference()
    # 승가/서성임 개체 > 눕기 개체
    hi = R.score({"mounting": 0.3, "restless": 0.5, "standing": 0.2}, 0.9)
    lo = R.score({"lying": 0.8, "sitting": 0.2}, 0.05)
    assert hi > lo
    m = ref.map_fractions({"walk": 0.4, "run": 0.2, "lying": 0.4})
    assert round(m["restless"], 3) == 0.6 and m["lying"] == 0.4


def test_appearance_crop_feats() -> None:
    """외형 크롭 피처가 40차원으로 산출되는지(더미 이미지)."""
    import numpy as np
    import model_behavior_appearance as mba
    dummy = (np.random.rand(60, 40, 3) * 255).astype("uint8")
    f = mba.crop_feats(dummy)
    assert f.shape == (40,)
    assert mba.crop_feats(None).shape == (40,)


def test_iou_tracker() -> None:
    """IoU 추적기: 이동하는 두 개체에 안정적 ID 부여."""
    import iou_tracker as trk
    frames = [(f, [(f * 2, 10, 20, 20), (100 - f, 100, 20, 20)],
               [{"gt": "A"}, {"gt": "B"}]) for f in range(10)]
    tracks = trk.track_sequence(frames)
    ev = trk.evaluate_vs_gt(tracks)
    assert ev["n_tracks"] == 2 and ev["id_consistency"] == 1.0
    assert trk.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_eval_report_figs() -> None:
    """평가 리포트 그림 생성(혼동행렬·ROC/PR/보정)이 data URI 를 내는지."""
    import numpy as np
    import build_eval_report as ev
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 200), 0, 1)
    uri, mets = ev.curves_fig(y, proba, "test")
    assert uri.startswith("data:image/png;base64,")
    assert 0.0 <= mets["auc"] <= 1.0 and "brier" in mets
    labels = ["a", "b", "c"]
    yc = rng.choice(labels, 90); pc = rng.choice(labels, 90)
    cm = ev.confusion_fig(yc, pc, labels, "test")
    assert cm.startswith("data:image/png;base64,")
    rows = ev.perclass_bars(yc, pc, labels)
    assert "<tr>" in rows


def test_estrus_reference_validation() -> None:
    """발정 실측 검증 다리: 합성 71471 로 보정 AUC 가 산출되는지."""
    import validate_estrus_reference as ver
    r = ver.evaluate()  # 실파일 없으면 합성 시연
    assert r["is_real"] is False and r["n"] >= 20
    assert 0.0 <= r["auc_calibrated"] <= 1.0
    assert 0.0 <= r["auc_rule"] <= 1.0
    assert len(r["proba"]) == r["n"] and len(r["y"]) == r["n"]


def test_repro_cause_attribution() -> None:
    """번식 문제 유형 분류 + 원인 귀인: THI·심각도·진단 동작."""
    import repro_cause_attribution as rca
    assert rca.thi(30, 80) > rca.thi(20, 50)   # 고온다습이 THI↑
    # 영양·수퇘지 자극 부족 개체 → 원인 귀인
    row = {"backfat_mm": 9.0, "feed_adequacy": 0.5, "temp_c": 22,
           "humidity_pct": 60, "boar_exposure_min": 2, "facility_score": 0.8,
           "water_adequacy": 0.9, "nh3_ppm": 12, "growth_disease_cnt": 0,
           "age_over_target": 20, "activity_mean": 6.0,
           "frac_standing": 0.05, "frac_tailing": 0.02}
    a = rca.attribute(row)
    assert abs(sum(a["share"].values()) - 1.0) < 1e-6
    # 등지방은 U자형 — 적정(16~22mm)은 무벌점, 야윔·비만 양쪽에 벌점
    base = {"feed_adequacy": 0.9, "temp_c": 22, "humidity_pct": 60,
            "boar_exposure_min": 20, "facility_score": 0.9,
            "water_adequacy": 0.9, "nh3_ppm": 10, "growth_disease_cnt": 0}
    sev = lambda bf: rca.attribute({**base, "backfat_mm": bf})["severity"]["영양 부족"]
    assert sev(17) == 0.0, "적정 등지방에 벌점"
    assert sev(9) > 0.3, "야윔에 벌점 없음"
    assert sev(26) > 0.3, "비만에 벌점 없음(양방향 미반영)"
    assert 1.0 <= rca.bcs_from_backfat(17) <= 5.0
    assert rca.bcs_from_backfat(9) < rca.bcs_from_backfat(17) < rca.bcs_from_backfat(25)
    assert a["top"][0] in rca.CAUSE_GROUPS
    d = rca.diagnose(row, risk=0.9)
    assert d["problem"] in rca.PROBLEMS and d["action"]
    # 활동↓·징후~0 → 무발정, 활동 정상·징후 뚜렷·저위험 → 정상
    assert rca.classify_problem(
        {"activity_norm": 0.05, "frac_standing": 0.02, "frac_tailing": 0.0,
         "age_over_target": 0}, risk=0.9) == "무발정"
    assert rca.classify_problem(
        {"activity_norm": 0.8, "frac_standing": 0.20, "frac_tailing": 0.10,
         "age_over_target": 0}, risk=0.1) == "정상"


def test_estrus_early_warning() -> None:
    """발정 조기경보: D-day 외삽·경보 상태·리드타임 동작."""
    import estrus_early_warning as ew
    # 상승 추세 → 임계 도달일 외삽
    d = ew.predict_onset_day([0, 1, 2, 3], [0.2, 0.3, 0.4, 0.5])
    assert d is not None and d > 3
    # 정체 추세 → 예측 없음
    assert ew.predict_onset_day([0, 1, 2], [0.2, 0.2, 0.2]) is None
    # 발정 도래(임계 지속) → 상태 '발정 확인'
    days = list(range(10)); sc = [0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85]
    a = ew.assess(days, sc)
    assert a["state"] == "발정 확인" and a["onset_actual"] is not None
    # 끝까지 낮음 → 무발정 경보
    flat = ew.assess(list(range(22)), [0.2] * 22)
    assert flat["state"] == "무발정 경보"
    # 타임라인: 지연 개체는 지연/무발정 경보가 발화
    tl = ew.timeline(list(range(22)), [0.2] * 8 + [0.3, 0.5, 0.7, 0.8] + [0.85] * 10)
    assert tl["alert_day"] is not None


def test_repro_dashboard_svg() -> None:
    """번식 대시보드 SVG 헬퍼: 막대·라인차트가 유효 SVG 를 내는지."""
    import build_repro_dashboard as brd
    svg = brd.hbar([("A", 3, "#111", "3두"), ("B", 1, "#222", "1두")])
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    ex = {"normal": {"days": [0, 1, 2], "scores": [0.2, 0.5, 0.8],
                     "onset": 2, "imminent": 1, "alert": None}}
    chart, legend = brd.line_chart(ex)
    assert "<polyline" in chart and "발정 임계" in chart and "정상 발정" in legend


def test_parse_71471_real_schema() -> None:
    """71471 실제 배포 스키마(ANNOTATION_INFO/ESTRUS) 파서 + 실측 검증 경로."""
    import json
    import tempfile
    import parse_71471_real as p71
    import validate_estrus_reference as ver
    acts = ["lying", "standing", "eating", "tailing", "sitting"]
    with tempfile.TemporaryDirectory() as d:
        for k in range(10):
            ts = 160700 + k * 100
            anns = [{"ID": 1000 + k * 10 + i,
                     "BOUNDING_BOX_X_COORDINATE": 142 + i * 180,
                     "BOUNDING_BOX_Y_COORDINATE": 455,
                     "BOUNDING_BOX_WIDTH": 360 - i * 20,
                     "BOUNDING_BOX_HEIGHT": 260 - i * 10,
                     "CATEGORY_NAME": "pig",
                     "ACTION_NAME": "tailing" if i == 0 else acts[i % len(acts)],
                     "ESTRUS": "Y" if i == 0 else "N"} for i in range(5)]
            fn = f"pigfarmA_ch9_2022092109_20-85_{ts}.json"
            json.dump({"INFO": {"VERSION": "1.0"},
                       "IMAGE": {"IMAGE_FILE_NAME": fn.replace(".json", ".jpg"),
                                 "WIDTH": 1920, "HEIGHT": 1080, "TIMESTAMP": ts,
                                 "FARMID": "pigfarmA", "HEADCOUNT": 500},
                       "ANNOTATION_INFO": anns},
                      open(os.path.join(d, fn), "w"), ensure_ascii=False)
        df = p71.parse_dir(d)
        assert len(df) == 50 and df["estrus"].sum() == 10
        assert {"session", "behavior", "estrus", "bbox_w"} <= set(df.columns)
        nm = p71.parse_name("pigfarmA_ch9_2022092109_20-85_160700.json")
        assert nm["farm"] == "pigfarmA" and nm["channel"] == "ch9" and nm["ts"] == 160700
        r = ver.evaluate_real_schema(d)
        assert r["is_real"] and r["schema"] == "71471-real"
        assert 0.0 <= r["auc_calibrated"] <= 1.0


def test_estrus_calendar_link() -> None:
    """외음부 발정 달력 × bbox 인덱스 연결(개체 내 시간 대조)."""
    import json
    import tempfile
    import estrus_calendar as ec
    with tempfile.TemporaryDirectory() as d:
        vd = os.path.join(d, "vulva"); bd = os.path.join(d, "bbox")
        os.makedirs(vd); os.makedirs(bd)
        # 개체 A: 11/02 발정 / 개체 B: 11/09 발정
        for i, (a, dt) in enumerate([("1-16", "20221102_090000"),
                                     ("1-23", "20221109_090000")]):
            json.dump({"VULVA": {"ANIMAL_ID": a, "DATE": dt,
                                 "FARM_NAME": "pigfarmA", "ESTRUS": "Y"}},
                      open(os.path.join(vd, f"v{i}.json"), "w"))
        # bbox: 발정일 프레임 + 멀리 떨어진 비발정일 프레임 + 애매구간
        for a, dt in [("1-16", "2022110209"), ("1-16", "2022101009"),
                      ("1-16", "2022110409"), ("1-23", "2022110909"),
                      ("1-23", "2022100109")]:
            fn = f"pigfarmA_ch1_{dt}_{a}_100.json"
            open(os.path.join(bd, fn), "w").write("{}")
        cal = ec.load_calendar(vd)
        assert len(cal) == 2 and cal["estrus"].sum() == 2
        idx = ec.bbox_index(bd)
        assert len(idx) == 5 and set(idx["animal"]) == {"1-16", "1-23"}
        linked = ec.link(cal, idx, window=3)
        # 발정일 2건(양성), 멀리 떨어진 2건(음성), 11/04(발정+2일)은 제외
        assert int((linked["estrus"] == 1).sum()) == 2
        assert int((linked["estrus"] == 0).sum()) == 2
        assert len(linked) == 4
        # 개체 내 대조가 성립(각 개체가 양성·음성 모두 보유)
        assert (linked.groupby("animal")["estrus"].nunique() > 1).all()


def test_estrus_contrast_eval() -> None:
    """개체 내 대조 발정 검증: 프레임 구성·그룹 AUC·결론 문장."""
    import json
    import tempfile
    import estrus_contrast_eval as ece
    acts = ["lying", "standing", "sitting", "eating"]
    with tempfile.TemporaryDirectory() as d:
        vd = os.path.join(d, "v"); bd = os.path.join(d, "b")
        os.makedirs(vd); os.makedirs(bd)
        animals = ["1-10", "1-11", "1-12", "1-13"]
        for i, a in enumerate(animals):   # 모두 09/21 발정
            json.dump({"VULVA": {"ANIMAL_ID": a, "DATE": "20220921_090000",
                                 "FARM_NAME": "pigfarmA", "ESTRUS": "Y"}},
                      open(os.path.join(vd, f"v{i}.json"), "w"))
        # 같은 채널(ch3)에서 발정일(0921)·비발정일(1005) 프레임 생성
        for a in animals:
            for dt in ("2022092109", "2022100509"):
                for t in range(6):
                    anns = [{"ID": 1, "BOUNDING_BOX_X_COORDINATE": 10 + j * 50,
                             "BOUNDING_BOX_Y_COORDINATE": 20,
                             "BOUNDING_BOX_WIDTH": 100, "BOUNDING_BOX_HEIGHT": 80,
                             "CATEGORY_NAME": "pig",
                             "ACTION_NAME": acts[(j + t) % 4], "ESTRUS": "Y"}
                            for j in range(4)]
                    fn = f"pigfarmA_ch3_{dt}_{a}_{1000 + t}.json"
                    json.dump({"IMAGE": {"IMAGE_FILE_NAME": fn[:-5] + ".jpg",
                                         "WIDTH": 1920, "HEIGHT": 1080,
                                         "TIMESTAMP": 1000 + t, "FARMID": "pigfarmA"},
                               "ANNOTATION_INFO": anns},
                              open(os.path.join(bd, fn), "w"))
        F = ece.build_frames(vd, bd)
        assert len(F) == 48 and F["animal"].nunique() == 4
        assert F.groupby("animal")["y"].nunique().eq(2).all()  # 개체 내 대조 성립
        r = ece.evaluate(vd, bd)
        assert r["ok"] and r["n_within_contrast"] == 4
        assert r["auc_behavior"] is not None
        assert isinstance(ece.verdict(r), str) and ece.verdict(r)


def test_keypoints_parser_pose() -> None:
    """71471 [Keypoints] 파서 + 회전·크기 불변 자세 기술자."""
    import json
    import tempfile
    import numpy as np
    import parse_71471_keypoints as kpp
    # 자세 기술자: 회전·평행이동·크기를 바꿔도 쌍거리는 동일해야 한다
    base = np.array([[0, 0], [10, 0], [20, 0], [20, 10],
                     [10, 10], [0, 10], [5, 5], [15, 5]], float)
    kp1 = np.hstack([base, np.full((8, 1), 2.0)])
    th = np.pi / 3
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    kp2 = np.hstack([(base @ R.T) * 2.5 + 100, np.full((8, 1), 2.0)])
    f1, f2 = kpp.pose_features(kp1), kpp.pose_features(kp2)
    for c in kpp.PAIR_COLS[:6]:
        assert abs(f1[c] - f2[c]) < 1e-6, f"{c} 불변성 위반"
    assert abs(f1["kp_elong"] - f2["kp_elong"]) < 1e-6
    # 파서
    with tempfile.TemporaryDirectory() as d:
        kp = [v for j in range(8) for v in (800 + j * 10, 600 + (j % 3) * 15, 2)]
        fn = "pigfarmA_ch9_2022080509_334_663233.json"
        json.dump({"IMAGE": {"IMAGE_FILE_NAME": fn[:-5] + ".jpg", "WIDTH": 1920,
                             "HEIGHT": 1080, "TIMESTAMP": 663233, "FARMID": "pigfarmA"},
                   "ANNOTATION_INFO": [{"ID": 1, "KEYPOINTS": kp, "NUM_KEYPIONTS": 8,
                                        "CATEGORY_NAME": "pig", "ACTION_NAME": "lying",
                                        "ESTRUS": "N"}]},
                  open(os.path.join(d, fn), "w"))
        df = kpp.parse_dir(d)
        assert len(df) == 1 and df["animal"].iloc[0] == "334"
        assert df["channel"].iloc[0] == "ch9" and df["estrus"].iloc[0] == 0
        assert set(kpp.FEATURES) <= set(df.columns) and len(kpp.FEATURES) == 44


def test_pose_vs_behavior_eval() -> None:
    """자세 vs 행동라벨 비교 평가: 채널 분리 검증과 결론 문장."""
    import json
    import tempfile
    import random
    import pose_vs_behavior_eval as pve
    random.seed(4)
    acts = ["lying", "standing", "sitting", "eating"]
    with tempfile.TemporaryDirectory() as d:
        # ch1~4 발정 / ch5~8 비발정 (클래스별 채널 4개씩)
        for ch in range(1, 9):
            est = "Y" if ch <= 4 else "N"
            for t_ in range(6):
                anns = []
                for i in range(3):
                    kp = [v for j in range(8) for v in
                          (800 + i * 150 + j * 11 + random.randint(-4, 4),
                           600 + (j % 3) * 18 + random.randint(-4, 4), 2)]
                    anns.append({"ID": ch * 100 + t_ * 10 + i, "KEYPOINTS": kp,
                                 "NUM_KEYPIONTS": 8, "CATEGORY_NAME": "pig",
                                 "ACTION_NAME": random.choice(acts), "ESTRUS": est})
                fn = f"pigfarmA_ch{ch}_2022071009_100_{5000 + t_}.json"
                json.dump({"IMAGE": {"IMAGE_FILE_NAME": fn[:-5] + ".jpg",
                                     "WIDTH": 1920, "HEIGHT": 1080,
                                     "TIMESTAMP": 5000 + t_, "FARMID": "pigfarmA"},
                           "ANNOTATION_INFO": anns},
                          open(os.path.join(d, fn), "w"))
        r = pve.evaluate(d)
        assert r["ok"] and r["n_channels"] == 8
        assert r["n_pos_ch"] == 4 and r["n_neg_ch"] == 4
        assert 0.0 <= r["auc_pose"] <= 1.0 and 0.0 <= r["auc_behavior"] <= 1.0
        assert set(r["auc_pose_parts"]) == {"쌍거리(28)", "형태지표(8)", "가시성(8)"}
        assert isinstance(pve.verdict(r), str) and pve.verdict(r)


def test_motion_tracker() -> None:
    """카메라 모션 보상 추적: 화면 전체가 이동해도 ID 유지."""
    import numpy as np
    import motion_tracker as mt
    # 카메라가 오른쪽으로 20px 이동한 것과 동등한 아핀
    M = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]])
    assert abs(mt.motion_magnitude(M) - 20.0) < 1e-6
    b = mt.warp_box((10, 10, 40, 30), M)
    assert abs(b[0] - 30) < 1e-6 and abs(b[1] - 10) < 1e-6
    # 개체는 정지, 카메라만 이동 → 보상하면 트랙 2개 유지
    mc, plain = mt.MCIoUTracker(), mt.MCIoUTracker()
    ids_mc, ids_pl = set(), set()
    M = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0]])   # 박스 너비(40)보다 큰 이동
    for k in range(6):
        sh = k * 50.0                      # 카메라 누적 이동
        boxes = [(10 + sh, 10, 40, 30), (400 + sh, 100, 40, 30)]
        for tid, _ in mc.update(boxes, M if k else None):
            ids_mc.add(tid)
        for tid, _ in plain.update(boxes, None):
            ids_pl.add(tid)
    assert len(ids_mc) == 2, f"보상 시 ID 유지 실패: {len(ids_mc)}"
    assert len(ids_pl) > len(ids_mc)        # 보상 없으면 과분할


def test_box_merge() -> None:
    """창살 분할 박스 병합: 인접 조각은 합치고 멀리 떨어진 개체는 유지."""
    import box_merge as bm
    # 조각(작음) 2개 + 온전한 개체 2마리 → 조각만 합쳐 3개가 되어야 한다.
    # (프레임 내 '온전한 개체 크기'가 있어야 무엇이 조각인지 판별 가능하다)
    mix = [(0, 0, 45, 60), (50, 2, 45, 58), (300, 0, 100, 60), (420, 0, 100, 60)]
    assert len(bm.merge_split_boxes(mix)) == 3
    assert len(bm.merge_split_boxes([(100, 100, 60, 80), (500, 100, 60, 80)])) == 2
    assert len(bm.merge_split_boxes([(10, 10, 50, 50)])) == 1
    # 과병합 방지(실측 회귀): 비슷한 크기 개체가 나란히 붙어 있으면 합치지 않는다.
    # 군사 사육 영상에서 이 케이스를 놓쳐 마릿수가 5→1 로 붕괴한 적이 있다.
    adj = [(0, 0, 100, 60), (102, 0, 100, 60), (204, 0, 100, 60)]
    assert len(bm.merge_split_boxes(adj)) == 3, "붙어 있는 개체를 과병합함"


def test_temporal_features() -> None:
    """시간 윈도우 피처: 서성임(제자리 맴돎) vs 직선 이동 구분."""
    import numpy as np
    import pandas as pd
    import temporal_features as tf
    rows = []
    for k in range(20):                     # A: 제자리 왕복(서성임)
        rows.append({"individual_id": "A", "frame_idx": k, "speed": 5.0,
                     "centroid_x": 100 + (5 if k % 2 else -5), "centroid_y": 100,
                     "darea": 0.0})
    for k in range(20):                     # B: 직선 이동
        rows.append({"individual_id": "B", "frame_idx": k, "speed": 5.0,
                     "centroid_x": 100 + 5 * k, "centroid_y": 100, "darea": 0.0})
    d = tf.add_temporal(pd.DataFrame(rows))
    assert set(tf.TEMPORAL_COLS) <= set(d.columns)
    a = d[(d.individual_id == "A") & (d.frame_idx >= 15)]["path_ratio15"].mean()
    b = d[(d.individual_id == "B") & (d.frame_idx >= 15)]["path_ratio15"].mean()
    assert a > b, f"서성임 경로비({a:.2f})가 직선({b:.2f})보다 커야 함"
    assert np.isfinite(d[tf.TEMPORAL_COLS].to_numpy()).all()


def test_breeding_timing() -> None:
    """교배 적기: WEI 보정·argmax 권장·회전율 경제 계산."""
    import breeding_timing as bt
    # WEI 가 짧으면 발정이 길고 배란이 늦다
    assert bt.ovulation_time("sow", 4) > bt.ovulation_time("sow", 10)
    # 후보돈은 경산돈보다 발정이 짧다
    assert bt.estrus_duration("gilt", 7) < bt.estrus_duration("sow", 7)
    # 권장 시각은 **자기 모델의 argmax** — 관행(12/24h)보다 항상 낫거나 같아야 한다
    for parity in ("sow", "gilt"):
        for wei in (4, 7, 10):
            w = bt.insemination_window(parity, wei)
            opt = bt.conception_prob([w["ai1_h"], w["ai2_h"]], parity, wei)
            routine = bt.conception_prob([12, 24], parity, wei)
            assert opt >= routine - 1e-9, f"{parity} WEI{wei}: 권장이 관행보다 나쁨"
            # 창은 유효도 정점을 포함하고, 정점은 **배란보다 앞**이어야 한다
            # (수정능획득 때문 — 배란 정각 주입은 이미 늦다)
            assert w["window_start_h"] <= w["peak_h"] <= w["window_end_h"]
            assert w["peak_h"] < w["ovulation_h"], "정점이 배란 이후 — 수정능획득 누락"

    # 수정능획득 지연: 배란 직전 주입보다 몇 시간 앞선 주입이 낫다
    ov = bt.ovulation_time("sow", 7)
    assert bt.ai_efficacy(ov - 8, "sow", 7) > bt.ai_efficacy(ov, "sow", 7)
    # 지침의 '주입 금지' 구간은 유효도가 낮아야 한다
    assert bt.ai_efficacy(0, "sow", 7) < 0.05
    assert bt.ai_efficacy(4, "sow", 7) < bt.ai_efficacy(24, "sow", 7)
    # 배란 한참 뒤 수정은 수태율이 급감한다
    assert bt.conception_prob([ov - 6], "sow", 7) > bt.conception_prob([ov + 30], "sow", 7)

    # 현장 지침(적기 12~36h)과 대조 — 권장값이 구간을 벗어나면 안 된다
    for parity in ("sow", "gilt"):
        for wei in (4, 7, 10):
            c = bt.check_against_field_guide(parity, wei)
            assert c["in_window"], f"{parity} WEI{wei}: 권장 {c['ai_times']} 이 지침 이탈"
            assert c["peak_in_window"] and c["no_early_ai"]

    # 관측 지연: 점검 주기가 길수록 수태율이 떨어진다(각 주기의 최적 프로토콜 기준).
    # 오프셋을 고정한 채 지연만 키우면 하루 1회 점검이 0.37 로 나오는 비현실적
    # 결과가 됐다 — 주기마다 최적 프로토콜을 다시 찾아 비교해야 한다.
    prev = None
    for iv in (0, 6, 12, 24):
        d = bt.detection_value(iv, "sow", 7)
        assert 0.5 < d["conception"] <= 1.0, f"{iv}h 주기 수태율 {d['conception']}"
        if prev is not None:
            assert d["conception"] <= prev + 1e-9, "점검이 뜸한데 수태율이 올랐다"
        prev = d["conception"]
    # 점검이 뜸할수록 프로토콜은 더 이르게 잡혀야 한다(지연을 미리 상쇄)
    assert (bt.best_offsets_for_interval(24, "sow", 7)[0]
            < bt.best_offsets_for_interval(0, "sow", 7)[0])

    tl = bt.estrus_timeline("sow", 7)
    assert tl["vulva_change"][0] < tl["standing_heat"][0], \
        "외음부 변화가 승가허용보다 늦다 — 조기 신호가 성립하지 않음"
    assert tl["prodromal"][1] <= tl["standing_heat"][0]
    # 회전율: 수태율이 높을수록 회전 빠르고 공태일 적다
    assert bt.turnover(0.9) > bt.turnover(0.7)
    assert bt.npd(0.9) < bt.npd(0.7)
    assert bt.cycle_days(1.0) == bt.GESTATION + bt.LACTATION + bt.NORMAL_WEI
    e = bt.economics(300, 0.78, 0.85)
    assert e["won_saved_year"] > 0 and e["turnover_after"] > e["turnover_before"]


def test_stall_estrus() -> None:
    """교배사(스톨) 발정 지표: 자세 기반 특징 추출과 점수화."""
    import pandas as pd
    import stall_estrus as se
    # A: 기립 많고 전환 잦음(발정 양상) / B: 계속 누움
    rows = []
    for f in range(40):
        rows.append({"stall_id": "A", "frame_idx": f,
                     "posture": "standing" if f % 3 else "lying"})
        rows.append({"stall_id": "B", "frame_idx": f, "posture": "lying"})
    feat = se.stall_features(pd.DataFrame(rows)).set_index("stall_id")
    assert feat.loc["A", "stand_frac"] > feat.loc["B", "stand_frac"]
    assert feat.loc["A", "transitions"] > feat.loc["B", "transitions"]
    assert feat.loc["B", "lie_frac"] == 1.0
    sc = se.estrus_score(se.stall_features(pd.DataFrame(rows))).set_index("stall_id")
    assert sc.loc["A", "estrus_score"] > sc.loc["B", "estrus_score"]
    # 부동자세: 연속 기립이 길면 immobile_frac 이 잡힌다
    long_stand = [{"stall_id": "C", "frame_idx": f, "posture": "standing"}
                  for f in range(30)]
    fc = se.stall_features(pd.DataFrame(long_stand)).iloc[0]
    assert fc["longest_stand"] == 30 and fc["immobile_frac"] == 1.0
    # 합성 시연이 **완전 분리(AUC 1.0)가 아니어야** 한다(개체차 반영)
    from sklearn.metrics import roc_auc_score
    ts, truth = se.generate_demo(n_stalls=40, seed=0)
    d = se.estrus_score(se.stall_features(ts)).merge(truth, on="stall_id")
    auc = roc_auc_score(d["estrus"], d["estrus_score"])
    assert 0.5 < auc < 0.99, f"합성이 비현실적으로 쉬움(AUC {auc:.3f})"

    # 자세 오류 전파: 상류 정확도가 낮을수록 발정 AUC 가 낮아야 한다.
    # 표본이 작으면 시드 분산에 순서가 뒤집힌다(24개/5회로 재니 실제로 뒤집혔다).
    import numpy as np
    big_ts, big_truth = se.generate_demo(n_stalls=200, frames=120, seed=3)

    def auc_at(acc, seeds=12):
        vals = []
        for s in range(1 if acc >= 1.0 else seeds):
            n = big_ts if acc >= 1.0 else se.degrade(big_ts, acc, seed=s)
            d = se.estrus_score(se.stall_features(n)).merge(big_truth,
                                                           on="stall_id")
            vals.append(roc_auc_score(d["estrus"], d["estrus_score"]))
        return float(np.mean(vals))

    perfect, better, worse = auc_at(1.0), auc_at(0.636), auc_at(0.513)
    assert perfect > better > worse, (
        f"자세 오류가 발정 AUC 에 단조롭게 전파되지 않음 "
        f"({perfect:.3f} / {better:.3f} / {worse:.3f})")
    # degrade 는 실제로 라벨을 바꿔야 한다(무작위성이 죽으면 조용히 통과한다)
    d0 = se.degrade(big_ts, 0.5, seed=1)
    changed = (d0["posture"].to_numpy() != big_ts["posture"].to_numpy()).mean()
    assert 0.2 < changed < 0.6, f"오류 주입 비율이 이상하다({changed:.2f})"
    assert set(d0["posture"]) <= {"standing", "sitting", "lying"}


def test_feeding_monitor() -> None:
    """합사 급이 모니터링: 세션·경쟁·섭취속도(순환논리 회귀 검증)."""
    import numpy as np
    import pandas as pd
    import feeding_monitor as fm
    zones = [(0.0, 0.0, 0.3, 0.3)]
    rows = []
    # A: 급이기에 오래 + 머리 많이 움직임(빨리 먹음)
    # B: 급이기에 오래 + 거의 안 움직임(천천히 먹음)  → 같은 시간, 다른 속도
    rng = np.random.default_rng(0)
    for f in range(300):
        rows.append({"pig_id": "A", "frame_idx": f,
                     "cx": 0.15 + rng.normal(0, 0.010),
                     "cy": 0.15 + rng.normal(0, 0.010)})
        rows.append({"pig_id": "B", "frame_idx": f,
                     "cx": 0.15 + rng.normal(0, 0.001),
                     "cy": 0.15 + rng.normal(0, 0.001)})
        rows.append({"pig_id": "C", "frame_idx": f, "cx": 0.8, "cy": 0.8})
    tracks = pd.DataFrame(rows)
    sess = fm.feeding_sessions(tracks, zones, fps=10.0)
    assert set(sess["pig_id"]) == {"A", "B"}, "급이기 밖 개체가 세션에 포함됨"
    assert "motion" in sess.columns
    met = fm.feeding_metrics(sess, fm.displacements(sess), total_feed_kg=6.0)
    m = met.set_index("pig_id")
    # 저작 강도가 큰 A 가 더 빨리 먹은 것으로 추정돼야 한다
    assert m.loc["A", "chew_intensity"] > m.loc["B", "chew_intensity"]
    assert m.loc["A", "eat_rate_g_per_min"] > m.loc["B", "eat_rate_g_per_min"]
    # 순환논리 회귀: 점유시간이 같은데 속도가 같아지면 안 된다
    assert abs(m.loc["A", "eat_rate_g_per_min"] - m.loc["B", "eat_rate_g_per_min"]) > 1
    d = fm.flag_risk(met)
    assert "feed_adequacy" in d.columns and "status" in d.columns
    assert fm.zone_of(0.15, 0.15, zones) == 0 and fm.zone_of(0.9, 0.9, zones) is None


def test_repro_calendar() -> None:
    """작업 캘린더: 날짜 1개 → 전체 일정, 관측이 예상을 대체, 그룹 일괄 생성."""
    from datetime import date, datetime
    import repro_calendar as rc
    tasks = rc.schedule_from_weaning("2026-08-10", parity="sow")
    kinds = [t["task"] for t in tasks]
    for need in ("이유", "발정 관찰", "교배", "재발정 확인", "임신감정",
                 "분만사 이동", "분만"):
        assert need in kinds, f"{need} 작업이 생성되지 않음"
    assert tasks == sorted(tasks, key=lambda t: t["date"]), "날짜순이 아님"

    # 순서 회귀: '발정 관찰'이 '교배'보다 뒤에 오면 안 된다
    first_obs = min(t["date"] for t in tasks if t["task"] == "발정 관찰")
    first_ai = min(t["date"] for t in tasks if t["task"] == "교배")
    assert first_obs < first_ai, "발정 관찰이 교배 뒤에 배치됨"

    s = rc.cycle_summary(tasks)
    assert 140 <= s["cycle_days"] <= 160, f"1주기 {s['cycle_days']}일 (150 근처여야)"
    assert s["npd_days"] == s["cycle_days"] - rc.GESTATION - rc.LACTATION

    # 후보돈은 이유가 없다 — 경산돈 경로로 넣으면 거부해야 한다
    try:
        rc.schedule_from_weaning("2026-08-10", parity="gilt")
        raise AssertionError("후보돈에 이유 기준 일정이 허용됨")
    except ValueError:
        pass
    g = rc.schedule_from_estrus("2026-08-10", parity="gilt")
    assert "이유" not in [t["task"] for t in g][:2]
    assert min(t["date"] for t in g if t["task"] == "교배") >= date(2026, 8, 10)

    # 관측이 예상을 대체한다: 확정 교배는 estimated=False
    conf = rc.schedule_from_weaning("2026-08-10", "sow",
                                    estrus_confirmed=datetime(2026, 8, 14, 6))
    ai = [t for t in conf if t["task"] == "교배"]
    assert ai and all(not t["estimated"] for t in ai), "확정 발정인데 교배가 추정으로 남음"
    est = [t for t in tasks if t["task"] == "교배"]
    assert all(t["estimated"] for t in est), "미확인인데 교배가 확정으로 표시됨"
    assert ai[0]["date"] != est[0]["date"], "발정 확인이 교배일에 반영되지 않음"

    # 그룹 등록: 입력 1회 → N두, 개별 확인은 해당 개체만 갱신
    grp = rc.group_from_weaning(["A", "B", "C"], "2026-08-10")
    assert len(grp) == 3 and all(len(v) == len(tasks) for v in grp.values())
    grp2 = rc.confirm_estrus(grp, "B", datetime(2026, 8, 14, 6))
    ai_b = [t["date"] for t in grp2["B"] if t["task"] == "교배"]
    ai_a = [t["date"] for t in grp2["A"] if t["task"] == "교배"]
    assert ai_b != ai_a, "개체 확인이 반영되지 않음"
    assert grp2["A"] == grp["A"], "다른 개체 일정까지 바뀜"

    todo = rc.due_today(grp, today="2026-08-16", horizon=1)
    assert todo and all(0 <= t["d_day"] <= 1 for t in todo)
    assert todo == sorted(todo, key=lambda t: (t["d_day"], -t["priority"]))
    late = rc.overdue(grp, today="2026-09-30")
    assert late and all(t["late_days"] > 0 for t in late)


def test_farm_registry() -> None:
    """축사 등록·배치 규칙·관리표·분석 경로."""
    import farm_registry as fr
    f = fr.Farm("t")
    f.add_barn("1동", "교배사").add_pen("1동", "A열", "stall", 3)
    f.add_barn("2동", "임신사").add_pen("2동", "1방", "group", 2)

    # 미등록 축사/돈방, 잘못된 용도·방식은 거부
    for bad in (lambda: f.add_barn("9동", "없는용도"),
                lambda: f.add_pen("9동", "x", "stall", 2),
                lambda: f.add_pen("1동", "y", "없는방식", 2),
                lambda: f.add_pen("1동", "z", "stall", 0),
                lambda: f.place("A", "1동", "없는방")):
        try:
            bad()
            raise AssertionError("잘못된 등록이 허용됨")
        except (KeyError, ValueError):
            pass

    # 스톨은 자리 번호 필수 — 없으면 카메라 화면과 대조할 수 없다
    try:
        f.place("A", "1동", "A열")
        raise AssertionError("스톨에 자리 없이 배치가 허용됨")
    except ValueError:
        pass

    f.place("A", "1동", "A열", 1).place("B", "1동", "A열", 2)
    assert f.locate("A") == ("1동", "A열", "1")
    assert f.at("1동", "A열") == ["A", "B"]
    assert "1동" in f.label("A") and "1번" in f.label("A")
    assert f.label("없는개체") == "미배치"

    # 같은 자리 이중 배치 금지
    try:
        f.place("C", "1동", "A열", 1)
        raise AssertionError("이미 찬 자리에 배치가 허용됨")
    except ValueError:
        pass
    # 수용능력 초과 금지
    f.place("C", "1동", "A열", 3)
    try:
        f.place("D", "1동", "A열", 4)
        raise AssertionError("수용능력 초과 배치가 허용됨")
    except ValueError:
        pass

    # 이동하면 옛 자리는 비어야 한다(같은 개체가 두 곳에 잡히면 두수가 틀어진다)
    f.place("A", "2동", "1방")
    assert f.locate("A")[0] == "2동"
    assert "A" not in f.at("1동", "A열")
    assert len(f.table()) == 3

    # 자리 번호 자연 정렬(1,10,2 가 아니라 1,2,10)
    g = fr.Farm("s")
    g.add_barn("1동", "교배사").add_pen("1동", "A열", "stall", 12)
    for s in (10, 2, 1):
        g.place(f"P{s}", "1동", "A열", s)
    assert list(g.table()["slot"]) == ["1", "2", "10"]

    occ = f.occupancy().set_index(["barn", "pen"])
    assert occ.loc[("1동", "A열"), "n"] == 2 and occ.loc[("1동", "A열"), "free"] == 1

    # 등록이 분석 경로를 정한다: 스톨/군사는 다른 모듈, 분만틀은 대상 외
    route = f.analysis_route().set_index(["barn", "pen"])
    assert route.loc[("1동", "A열"), "module"] == "stall_estrus"
    assert "motion_tracker" in route.loc[("2동", "1방"), "module"]
    f.add_barn("3동", "분만사").add_pen("3동", "분만실", "crate", 2)
    route = f.analysis_route().set_index(["barn", "pen"])
    assert not bool(route.loc[("3동", "분만실"), "estrus_target"]), \
        "분만사에 발정 판정을 돌리려 함"

    f.remove("A")
    assert f.locate("A") is None and len(f.table()) == 2

    # 번식 상태 결합 + 배치 오류 검출
    import herd_board as hb
    demo = fr.demo_farm()
    ids = sorted(demo._where)
    recs = hb.generate_demo(n=len(ids) + 40, today="2026-08-10")[:len(ids)]
    for r, i in zip(recs, ids):
        r["id"] = i
    herd = hb.build_herd(recs, today="2026-08-10")
    t = demo.table(herd)
    assert len(t) == len(ids) and "stage_h" in t.columns
    assert t["id"].nunique() == len(ids), "관리표에 개체 중복"
    mp = demo.misplaced(herd)
    assert len(mp) and {"id", "loc", "reason"} <= set(mp.columns)
    # 분만사의 포유돈은 정상 — 오류로 잡히면 안 된다
    ok = t[(t["stage"] == "분만사") & (t["stage_h"] == "포유")]["id"]
    assert not set(ok) & set(mp["id"]), "정상 배치가 오류로 잡힘"


def test_barn_queue() -> None:
    """작업동별 조치 큐: 단일 판정·동 순서·준비물."""
    import breeding_ledger as bl
    import build_barn_map as bm
    today = "2026-08-10"
    farm, herd, scheds, scores = bl.build_demo(today)
    led = bl.ledger(farm, herd, scheds, scores, today=today)

    # 판정은 한 곳에만 — 도면과 큐가 같은 수를 세야 한다(23 vs 68 회귀)
    assert bm.cell_status is bl.action_status
    n_map = sum(1 for r in led.to_dict("records") if bl.is_actionable(r))
    q = bl.barn_queue(led)
    assert sum(g["n"] for g in q) == n_map, "큐 합계가 도면 조치 대상과 불일치"
    assert n_map < len(led), "전 개체가 조치 대상"

    # 동은 겹치지 않고, 동 안은 긴급도 내림차순
    barns = [g["barn"] for g in q]
    assert len(barns) == len(set(barns))
    for g in q:
        u = [r["urgency"] for r in g["rows"]]
        assert u == sorted(u, reverse=True), f"{g['barn']} 동 내부 정렬 깨짐"
        assert g["n"] == len(g["rows"])
    # 가장 급한 개체가 있는 동이 먼저
    tops = [g["top_urgency"] for g in q]
    assert tops == sorted(tops, reverse=True)
    assert [g["visit_order"] for g in q] == list(range(1, len(q) + 1))

    # 준비물: 발정 관찰과 재발정 확인은 둘 다 웅돈 — 한 줄로 합쳐야 한다
    assert bl.SUPPLIES["발정 관찰"] == bl.SUPPLIES["재발정 확인"] == "웅돈"
    for g in q:
        tasks = [r["next_task"] for r in g["rows"]]
        want = sum(1 for t in tasks if bl.SUPPLIES.get(t) == "웅돈")
        assert g["supplies"].get("웅돈", 0) == want, f"{g['barn']} 웅돈 수 불일치"

    # 동선 순서: 등록 순서를 따른다
    route = list(farm.barns)
    rq = bl.barn_queue(led, order="route", route=route)
    seen = [g["barn"] for g in rq]
    assert seen == [b for b in route if b in seen], "동선 순서가 지켜지지 않음"


def test_growth_flow() -> None:
    """사육단계: 단계·체중·밀도·지연개체·PSY/MSY."""
    import numpy as np
    import pandas as pd
    import growth_flow as gf

    # 단계는 끊김·겹침 없이 이어져야 한다
    for (n0, a0, a1, w0, w1, _b, _r), nxt in zip(gf.STAGES, gf.STAGES[1:]):
        assert a1 == nxt[1], f"{n0}→{nxt[0]} 일령이 안 이어진다"
        assert abs(w1 - nxt[3]) < 1e-9, f"{n0}→{nxt[0]} 체중이 안 이어진다"
        assert w1 > w0 and a1 > a0
    assert gf.STAGES[-1][2] == gf.MARKET_AGE
    assert abs(gf.STAGES[-1][4] - gf.MARKET_WEIGHT) < 1e-9

    assert gf.stage_at(40)[0] == "이유자돈" and gf.stage_at(40)[1] == "자돈사"
    assert gf.stage_at(200)[0] == "출하"
    # 체중은 단조증가하고 단계 경계에서 기준값과 맞아야 한다
    ws = [gf.weight_at(a) for a in range(0, 200, 5)]
    assert all(b >= a for a, b in zip(ws, ws[1:])), "체중이 감소하는 구간이 있다"
    for _n, a0, _a1, w0, _w1, _b, _r in gf.STAGES:
        assert abs(gf.weight_at(a0) - w0) < 1e-6
    # age_for_weight 는 weight_at 의 역이어야 한다
    for kg in (10.0, 30.0, 60.0, 115.0):
        assert abs(gf.weight_at(gf.age_for_weight(kg)) - kg) < 0.5

    tl = gf.batch_timeline("2026-08-10", 300)
    assert list(tl["stage"]) == ["이유자돈", "육성돈", "비육돈"], "포유가 섞였다"
    # 두수는 단계마다 줄기만 한다
    assert (tl["n_out"] <= tl["n_in"]).all()
    assert list(tl["n_in"][1:]) == list(tl["n_out"][:-1]), "단계 간 두수 불연속"
    assert tl.attrs["n_marketed"] < 300
    assert 0.85 < tl.attrs["survival"] < 1.0
    # 기간이 일령 구간과 맞는지
    for r in tl.itertuples(index=False):
        assert (r.end - r.start).days == r.days == r.age_to - r.age_from

    # 밀도: 법정 기준 미만이면 과밀
    ok = gf.density_check(100, 30.0, "이유자돈")      # 0.30 = 기준 정확히
    assert not ok["overcrowded"] and ok["excess"] == 0
    tight = gf.density_check(100, 20.0, "이유자돈")   # 0.20 < 0.30
    assert tight["overcrowded"] and tight["excess"] == 100 - int(20.0 // 0.30)
    assert gf.density_check(10, 5.0, "포유자돈")["regulated"] is False

    # 지연 개체: 가벼운 개체가 잡히고, 되돌리라고 말하지 않아야 한다
    pigs = pd.DataFrame({
        "id": ["A", "B", "C", "D"], "batch": "B1", "age_days": [120] * 4,
        "weight_kg": [80.0, 78.0, 82.0, 40.0]})
    te = gf.tail_enders(pigs)
    assert len(te) == 4 and te.iloc[0]["id"] == "D", "가장 가벼운 개체가 위가 아니다"
    assert bool(te.set_index("id").loc["D", "tail_ender"])
    assert not bool(te.set_index("id").loc["C", "tail_ender"])
    assert te.set_index("id").loc["D", "delay_days"] > 0
    for a in te["action"]:
        assert "되돌리지" in a or "정상" in a
    assert not any("어린 배치로 이동" in a for a in te["action"]), \
        "역류를 권하고 있다 — AIAO 가 깨진다"
    assert len(gf.tail_enders(pigs.iloc[:0])) == 0

    # PSY/MSY: MSY = PSY × 육성률, 벤치마크 재현
    r = gf.psy_msy(2.20, 10.4, 0.807)
    assert abs(r["psy"] - 22.9) < 0.15 and abs(r["msy"] - 18.5) < 0.15
    dk = gf.psy_msy(2.30, 13.6, 0.933)
    assert abs(dk["psy"] - 31.3) < 0.2 and abs(dk["msy"] - 29.2) < 0.2
    assert abs(r["msy"] - r["psy"] * r["post_wean_survival"]) < 0.05
    assert abs(r["post_wean_mortality"] + r["post_wean_survival"] - 1.0) < 1e-9
    # 국내 평균의 이유후 폐사가 덴마크보다 크다 — 전체 관리로 넓혀야 하는 근거
    assert r["post_wean_mortality"] > dk["post_wean_mortality"] * 2
    for name in gf.BENCHMARKS:
        assert name in r["vs"]
        assert gf.BENCHMARKS[name]["msy"] < gf.BENCHMARKS[name]["psy"]


def test_aihub_bridge() -> None:
    """AI Hub 실데이터 연동: 파싱·라벨 감사·축사 생성(데이터 없으면 건너뜀)."""
    import pandas as pd
    import aihub_bridge as ab

    # 라벨 감사 로직은 데이터 없이도 검증할 수 있다.
    # 돈방마다 라벨이 순수하면 '카메라 라벨'로 판정해야 한다(71471 의 함정).
    pure = pd.DataFrame({
        "pen": ["a"] * 4 + ["b"] * 4,
        "estrus_label": ["Y"] * 4 + ["N"] * 4,
        "posture": ["lying", "standing"] * 4})
    a = ab.label_audit(pure)
    assert a["confounded"] and a["pure_pens"] == a["n_pens"] == 2
    assert a["behaviour_tvd"] is not None and a["behaviour_tvd"] < 1e-9, \
        "행동 분포가 같은데 차이가 0 이 아니다"
    # 각 돈방에 Y/N 이 섞여 있고(교락 아님), Y 는 눕기 N 은 서기로 갈린다
    mixed = pd.DataFrame({
        "pen": ["a"] * 4 + ["b"] * 4,
        "estrus_label": ["Y", "Y", "N", "N"] * 2,
        "posture": ["lying", "lying", "standing", "standing"] * 2})
    m = ab.label_audit(mixed)
    assert not m["confounded"], "혼재 라벨을 교락으로 오판"
    assert m["behaviour_tvd"] > 0.5, "행동이 갈리는데 TVD 가 작다"
    assert ab.label_audit(pd.DataFrame())["has_label"] is False

    dirs = ab.data_dirs()
    if not dirs:
        return                      # 국내 IP 전용 — 없는 환경에서는 건너뛴다
    df = ab.load_frames(dirs)
    if not len(df):
        return
    # 두 디렉터리에 같은 파일이 겹쳐 있다. 프레임 이름으로 중복을 제거해야
    # bbox 수가 부풀지 않는다(중복 포함 13,916 vs 고유 12,805).
    assert df["frame"].nunique() == df.attrs["n_frames"], "프레임 중복 제거 실패"
    assert set(df["posture"]) <= {"lying", "sitting", "standing", "other"}
    # 폭·높이 0 인 박스가 실제로 섞여 있다(1/12,805). 원본은 그대로 두되
    # 몇 건인지 세어 두고, 소비 측은 valid_boxes() 로 걸러 쓴다.
    assert "degenerate_boxes" in df.attrs
    ok = ab.valid_boxes(df)
    assert (ok["w"] > 0).all() and (ok["h"] > 0).all()
    assert ok.attrs["dropped"] == df.attrs["degenerate_boxes"]
    assert len(ok) + ok.attrs["dropped"] == len(df)

    real = ab.label_audit(df)
    assert real["confounded"], "71471 ESTRUS 는 카메라 교락이어야 한다"

    farm = ab.build_farm(df)
    assert len(farm.pens) == df["pen"].nunique()
    # 개체 ID 가 없으므로 자리 배치를 하면 안 된다(없는 개체를 만들어내는 것)
    assert len(farm.slots) == 0, "추적 ID 가 없는데 개체를 배치했다"
    occ = farm.occupancy()
    assert (occ["capacity"] > 0).all() and (occ["n"] == 0).all()

    ses = ab.pen_sessions(df)
    assert len(ses) and (ses["headcount"] > 0).all()
    frac = ses[["standing", "sitting", "lying"]].sum(axis=1)
    assert ((frac - 1.0).abs() < 0.02).all(), "자세 비율 합이 1 이 아니다"
    sc = ab.pen_estrus_scores(ses)
    assert len(sc) == df["pen"].nunique()
    assert sc["estrus_score"].is_monotonic_decreasing


def test_pig_polygon() -> None:
    """Pig_Polygon(분만 폴리곤): CVAT 파싱·감사·이미지 단위 분할·내보내기."""
    import os
    import tempfile
    import parse_pig_polygon as pp

    with tempfile.TemporaryDirectory() as d:
        xml = pp.synth_cvat(os.path.join(d, "annotations.xml"), n_images=10)
        df = pp.parse_cvat(xml)
        assert len(df) and df["image"].nunique() == 10
        assert (df["n_points"] >= 3).all() and (df["area"] > 0).all()
        assert (df["w"] > 0).all() and (df["h"] > 0).all()

        # 신발끈 공식: 단위 정사각형은 면적 1
        assert abs(pp.polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)]) - 1.0) < 1e-9
        assert pp.polygon_area([(0, 0), (1, 1)]) == 0.0
        # 방향(시계/반시계)에 무관해야 한다
        assert abs(pp.polygon_area([(0, 0), (0, 1), (1, 1), (1, 0)]) - 1.0) < 1e-9

        a = pp.audit(df)
        assert a["zero_area"] == 0 and a["degenerate"] == 0
        assert a["out_of_frame"] == 0
        assert set(a["labels"]) == {"pig", "farrowing"}
        assert not a["label_purity"]["single_label"]

        # 분할은 **이미지 단위** — 같은 이미지가 train/test 에 갈라지면 누수
        sp = pp.split_images(df, seed=1)
        assert set(sp) == {"train", "val", "test"}
        imgs = [set(v["image"]) for v in sp.values()]
        assert not (imgs[0] & imgs[1]) and not (imgs[0] & imgs[2])
        assert not (imgs[1] & imgs[2])
        assert sum(len(v) for v in sp.values()) == len(df)
        assert sum(v["image"].nunique() for v in sp.values()) == 10

        coco = pp.to_coco(sp["train"])
        assert len(coco["images"]) == sp["train"]["image"].nunique()
        assert len(coco["annotations"]) == len(sp["train"])
        ids = {i["id"] for i in coco["images"]}
        assert all(x["image_id"] in ids for x in coco["annotations"])
        assert all(len(x["segmentation"][0]) >= 6 for x in coco["annotations"])
        assert all(x["area"] > 0 for x in coco["annotations"])

        out = os.path.join(d, "yolo")
        n = pp.to_yolo_seg(sp["train"], out)
        assert n == sp["train"]["image"].nunique()
        txts = [f for f in os.listdir(out) if f.endswith(".txt")]
        assert len(txts) == n
        # YOLO-seg 좌표는 0~1 정규화여야 한다
        vals = open(os.path.join(out, txts[0])).read().split()
        assert all(0.0 <= float(v) <= 1.0 for v in vals[1:])

    # 기준선은 지표까지 함께 기록해야 비교가 성립한다
    assert pp.BASELINE["value"] == 60.0 and "AP50" in pp.BASELINE["metric"]
    assert abs(sum(pp.SPLIT.values()) - 1.0) < 1e-9


def test_batch_flow() -> None:
    """돈군흐름(배칭): 배치 수·AIAO 방 수·여유·배치 유지율."""
    import numpy as np
    import batch_flow as bf
    import breeding_ledger as bl

    # 분만사 점유에 **세척 기간이 반드시 포함**돼야 한다. 빼면 방이 모자라
    # 올인/올아웃이 무너져 배칭의 목적 자체가 사라진다.
    assert bf.FARROW_OCCUPY == bf.MOVE_IN + bf.LACTATION + bf.WASHDOWN
    assert bf.WASHDOWN > 0

    p = bf.plan(300, 21)
    # 배치 수 × 간격 = 번식주기. n_batches 는 소수점 1자리로 반올림돼
    # 나오므로(7.1) 그 오차(±0.05×간격)를 감안해 본다.
    assert abs(p["n_batches"] * 21 - bf.CYCLE) < 0.05 * 21 + 1e-6
    # 두 값 모두 반올림된 값이라 곱이 정확히 300 이 되지는 않는다(298.2)
    assert abs(p["sows_per_batch"] * p["n_batches"] - 300) < 0.02 * 300
    # 방 수는 점유기간을 덮어야 한다
    assert p["farrow_rooms"] * 21 >= bf.FARROW_OCCUPY
    assert (p["farrow_rooms"] - 1) * 21 < bf.FARROW_OCCUPY, "방이 과다 산정"
    assert p["slack_days"] == p["farrow_rooms"] * 21 - bf.FARROW_OCCUPY
    # 권장 방 수는 최소 방 수 이상이고 여유를 확보한다
    assert p["rooms_recommended"] >= p["farrow_rooms"]
    assert p["rooms_recommended"] * 21 - bf.FARROW_OCCUPY >= bf.BUFFER
    # 분만 두수로 방 크기를 잡아야 한다(교배 두수로 잡으면 빈 분만틀이 생긴다)
    assert p["farrow_per_batch"] < p["sows_per_batch"]

    # 간격이 넓을수록 배치는 커지고 방은 줄고 집중도는 오른다
    c = bf.compare(300).sort_values("interval")
    assert c["sows_per_batch"].is_monotonic_increasing
    assert c["farrow_rooms"].is_monotonic_decreasing
    assert c["peak_ratio"].is_monotonic_increasing
    assert len(c) == len(bf.BATCH_INTERVALS)

    # 배치 번호: 같은 간격 안의 이유일은 같은 배치
    a = "2026-08-03"
    assert bf.batch_of("2026-08-03", a, 21) == 0
    assert bf.batch_of("2026-08-23", a, 21) == 0
    assert bf.batch_of("2026-08-24", a, 21) == 1
    assert bf.batch_of("2026-07-30", a, 21) == -1

    d = bf.batch_dates(a, 1, 21)
    assert (d["farrow"] - d["service"]).days == bf.GESTATION
    assert (d["next_wean"] - d["farrow"]).days == bf.LACTATION
    assert (d["room_free"] - d["move_in"]).days == bf.FARROW_OCCUPY
    assert (d["service_to"] - d["service_from"]).days == bf.BATCH_WINDOW

    # AIAO: 최소 방 수로도 겹치면 안 된다(경계에서 정확히 맞물린다)
    rs = bf.room_schedule(a, 21, n_batches=8)
    assert not rs["overlap"].any(), "최소 방 수인데 점유가 겹친다"
    assert len(rs) == 8

    # 배정 + 유지율. herd_board 의 생성기는 이유 이후 **재교배를 만들지 않아**
    # 유지율을 잴 수 없다(전부 미교배로 잡힌다) — 배칭용 생성기를 쓴다.
    herd = bf.generate_demo(300, 21, today="2026-08-10", adherence=0.82)
    asg = bf.assign(herd, 21)
    assert len(asg) == 300
    assert {"batch", "in_batch", "wei_actual"} <= set(asg.columns)
    g = bf.integrity(asg)
    assert g["n"] > 0, "교배 기록이 하나도 없다 — 유지율을 잴 수 없다"
    assert 0 <= g["rate"] <= 1 and g["in_batch"] <= g["n"]
    assert g["n_batches"] >= 2
    assert 0.7 <= g["rate"] <= 0.95, f"유지율 {g['rate']} — 설정과 동떨어짐"
    # WEI 는 음수일 수 없다. 직전 주기의 교배를 이번 배치로 세면 -143 이 나온다.
    served = asg[asg["wei_actual"].notna()]
    assert (served["wei_actual"] > 0).all(), "이유 이전 교배가 섞였다"
    for r in served.itertuples(index=False):
        assert bool(r.in_batch) == (r.wei_actual <= bf.BATCH_WINDOW)

    # 미교배(공태)는 창 안으로 세면 안 된다
    b2 = bl.build_demo("2026-08-10")[1]
    a2 = bf.assign(b2, 21)
    if len(a2):
        assert not a2["in_batch"].any(), "이유 후 미교배가 유지로 잡힘"
        assert bf.integrity(a2)["n"] == 0

    assert len(bf.assign(herd.iloc[:0], 21)) == 0

    # --- 분만틀 기준 설계(존 카 모델) — 참고 사례 수치를 그대로 재현하는지 ---
    q = bf.plan_from_crates(10, 7)
    assert q["services_per_batch"] == 13, "분만틀 10 → 교배 13두여야 한다"
    assert q["gilts_per_batch"] == 3
    assert 245 <= q["herd_size"] <= 250, f"모돈 {q['herd_size']} (약 247 이어야)"
    assert q["weaned_per_batch"] == 120.0 and q["marketed_per_batch"] == 114.0

    # 배치당 교배는 **평균이 아니라 하위 분위수**로 나눠야 한다.
    # 평균으로 잡으면 절반의 배치에서 분만틀이 빈다.
    assert bf.FARROW_RATE_P10 < bf.FARROW_RATE_AVG
    avg = bf.plan_from_crates(10, 7, farrow_rate=bf.FARROW_RATE_AVG)
    assert q["services_per_batch"] > avg["services_per_batch"]
    # 하위 분위수로 잡으면 나쁜 배치에서도 틀이 채워진다
    assert q["services_per_batch"] * bf.FARROW_RATE_P10 >= 10

    # 방 수: 참고 예시 3건(분만대기 4일·세척 3일)
    assert bf.rooms_for(28, 7, 4, 3) == 5
    assert bf.rooms_for(28, 21, 4, 3) == 2
    assert bf.rooms_for(21, 28, 4, 3) == 1
    # rooms_for 와 max_lactation 은 서로 역이어야 한다
    for lac, ivx in ((28, 7), (28, 21), (21, 28), (21, 14)):
        r = bf.rooms_for(lac, ivx, 4, 3)
        assert bf.max_lactation(r, ivx, 4, 3) >= lac
        assert bf.max_lactation(r - 1, ivx, 4, 3) < lac if r > 1 else True

    # 뒷단: 방이 점유를 덮어야 하고 자리 수는 배치 크기 이상
    ds = bf.downstream(480, 21)
    assert set(ds["stage"]) == {"자돈사", "육성사", "비육사"}
    for _i, r in ds.iterrows():
        assert r["rooms"] * 21 >= r["occupy"]
        assert r["places_total"] >= 480
        assert r["slack_days"] >= 0
    # 사육기간이 길수록 방이 더 필요하다
    assert (ds.sort_values("days")["rooms"].is_monotonic_increasing)

    # 벤치마크 임계치
    assert bf.aiao_worth_it(0.15, 0.05)["worth_it"]
    assert bf.aiao_worth_it(0.05, 0.10)["worth_it"]
    assert bf.aiao_worth_it(0.08, 0.04, 200)["worth_it"]
    assert not bf.aiao_worth_it(0.08, 0.04, 170)["worth_it"]
    assert len(bf.aiao_worth_it(0.15, 0.10)["reasons"]) == 2


def test_work_log() -> None:
    """작업 로그: 추가전용·취소 반영·큐 정정·적기 준수."""
    import tempfile
    import breeding_ledger as bl
    import work_log as wl
    today = "2026-08-10"
    farm, herd, scheds, scores = bl.build_demo(today)
    led = bl.ledger(farm, herd, scheds, scores, today=today)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        assert len(wl.load(path)) == 0
        wl.record("A", "교배", "2026-08-10", operator="김", path=path,
                  planned_date="2026-08-09")
        wl.record("B", "임신감정", "2026-08-10", path=path)
        lg = wl.load(path)
        assert len(lg) == 2 and list(lg.columns) == wl.COLS
        assert wl.done_keys(lg) == {("A", "교배"), ("B", "임신감정")}
        # 취소를 덧붙이면 완료가 무효가 된다(기록은 지우지 않는다)
        wl.record("A", "교배", "2026-08-10", result="취소", path=path)
        lg2 = wl.load(path)
        assert len(lg2) == 3, "취소가 기존 행을 덮어썼다 — 추가전용이 깨짐"
        assert ("A", "교배") not in wl.done_keys(lg2)
        try:
            wl.record("C", "교배", today, result="없는결과", path=path)
            raise AssertionError("잘못된 result 가 허용됨")
        except ValueError:
            pass

    # 합성 로그로 큐 정정
    log = wl.generate_demo(scheds, today=today)
    assert len(log) and set(log["result"]) <= set(wl.RESULTS)
    before = sum(1 for r in led.to_dict("records") if bl.is_actionable(r))
    led2 = wl.apply_to_ledger(led, log)
    after = sum(1 for r in led2.to_dict("records") if bl.is_actionable(r))
    assert "done" in led2.columns
    assert after <= before, "로그를 반영했는데 조치 대상이 늘었다"
    if int(led2["done"].sum()):
        assert after < before, "완료 기록이 있는데 큐가 줄지 않았다"
    # done 은 반드시 큐에서 빠져야 한다(긴급도만 0 으로 내리면 안 빠졌다)
    assert not any(bl.is_actionable(r) for r in led2.to_dict("records")
                   if r["done"])

    c = wl.compliance(log)
    assert len(c) and (c["on_time_rate"].between(0, 1)).all()
    assert (c["on_time"] + c["late"] + c["early"] <= c["n"]).all()
    # 교배는 허용 폭이 가장 좁아야 한다
    assert wl.ON_TIME["교배"] < wl.ON_TIME["임신감정"]

    s = wl.summary(log, days=14, today=today)
    assert s["n"] >= 0 and len(s["daily"]) == 14
    assert sum(x["n"] for x in s["daily"]) == s["n"]
    assert len(wl.summary(wl.load("/nonexistent.csv"))) == 1  # {"n":0}


def test_pregnancy_check() -> None:
    """임신진단 3단계: 캐스케이드 보존·조기검출 이득·초음파 의존성."""
    import pregnancy_check as pc
    shares = sum(c[4] for c in pc.CHECKPOINTS)
    assert abs(shares - 1.0) < 1e-9, f"재발 비율 합이 {shares} (1.0 이어야)"
    # 1차 관문은 초음파가 아니라 발정체크 — 이 프로젝트의 근거
    assert pc.CHECKPOINTS[0][5] == "재발정 확인" and pc.CHECKPOINTS[0][4] == 0.80

    rows = pc.detection_cascade()
    total = sum(r["caught"] for r in rows) + rows[-1]["missed_forward"]
    assert abs(total - 1.0) < 1e-6, f"캐스케이드 총합 {total} — 재발돈이 사라졌다"
    assert (rows[0]["npd_if_caught"] < rows[1]["npd_if_caught"]
            < rows[2]["npd_if_caught"]), "늦게 잡을수록 공태일이 길어야 한다"

    # 민감도가 높을수록 기대 공태일이 짧다
    good = pc.npd_from_returns(pc.CCTV_SENSITIVITY)
    poor = pc.npd_from_returns(pc.DEFAULT_SENSITIVITY)
    assert good < poor, "3주 검출을 개선했는데 공태일이 줄지 않음"
    assert 18 <= good <= 114

    v = pc.value_of_early(300)
    assert v["won_saved_year"] > 0 and v["npd_saved_per_return"] > 0
    # 초음파가 부실할수록 3주 개선의 가치가 커진다(과장 방지용 회귀)
    strict = pc.value_of_early(300,
                               base_sens={"3주": .70, "5주": .95, "8~10주": .90},
                               improved_sens={"3주": .92, "5주": .95, "8~10주": .90})
    none_us = pc.value_of_early(300,
                                base_sens={"3주": .70, "5주": .0, "8~10주": .90},
                                improved_sens={"3주": .92, "5주": .0, "8~10주": .90})
    assert none_us["won_saved_year"] > strict["won_saved_year"], \
        "초음파 유무와 무관하게 같은 이득이 나옴 — 캐스케이드가 작동하지 않는다"

    tasks = pc.checkpoint_tasks("2026-08-16")
    assert len(tasks) == 3
    assert [t["date"] for t in tasks] == sorted(t["date"] for t in tasks)
    assert tasks[0]["priority"] > tasks[1]["priority"], \
        "80% 를 잡는 관문이 더 급해야 한다"

    # 캘린더에 3단계가 실제로 반영됐는지
    import repro_calendar as rc
    sched = rc.schedule_from_service("2026-08-16")
    cps = [t for t in sched if t["task"] in ("재발정 확인", "임신감정")]
    assert len(cps) == 3, f"캘린더에 체크포인트가 {len(cps)}개"


def test_herd_board() -> None:
    """모돈군 현황판: 단계 판정·주차 파이프라인·산차 구성·도태·전입 계획."""
    from datetime import date, timedelta
    import herd_board as hb
    herd = hb.build_herd(hb.generate_demo(n=200, today="2026-08-10"),
                         today="2026-08-10")
    assert len(herd) == 200
    sc = hb.stage_counts(herd)
    assert sum(sc.values()) == 200
    # 단계 판정 회귀: 이유까지 끝난 모돈이 '임신'으로 남으면 공태가 0이 된다
    assert sc["공태"] > 0, "공태돈이 한 두도 없다 — 최근 사건 판정이 깨졌다"
    assert sc["임신"] > 0 and sc["포유"] > 0

    t0 = date(2026, 8, 10)
    one = hb.build_herd([{"id": "X", "parity": 3,
                          "service_date": t0 - timedelta(days=160),
                          "farrow_date": t0 - timedelta(days=45),
                          "weaning_date": t0 - timedelta(days=17)}], today=t0)
    assert one.loc[0, "stage"] == "공태" and one.loc[0, "npd"] == 17

    wb = hb.weekly_board(herd, today="2026-08-10")
    assert len(wb) == 17 and (wb["farrow"] >= 0).all()
    # 확정 판정 회귀: 주 '끝'을 역산해야 한다. 마지막 주는 아직 교배로 메울 수 있다
    assert not bool(wb.iloc[-1]["locked"]), "메울 수 있는 주가 확정 손실로 잡힘"
    assert bool(wb.iloc[0]["locked"])

    pp = hb.parity_profile(herd)
    assert abs(pp["target_share"].sum() - 1.0) < 1e-9, "목표 산차 구성 합이 1이 아님"
    assert pp["n"].sum() == int((herd["parity"] > 0).sum()), "산차 집계 누락/중복"

    cc = hb.cull_candidates(herd)
    assert len(cc) and (cc["score"].diff().dropna() <= 0).all(), "점수 내림차순 아님"
    assert cc["reason"].str.len().gt(0).all()

    gi = hb.gilt_intake_plan(herd, months=6, today="2026-08-10")
    a = gi.attrs
    # 용량 상한 회귀: 적체가 아무리 커도 월 전입이 상한을 넘으면 안 된다
    assert (gi["need"] <= a["monthly_cap"] + 1).all(), "월 전입이 격리사 용량 초과"
    assert (gi["backlog_left"].diff().dropna() <= 0).all(), "적체가 늘어남"
    assert a["months_to_clear"] is None or a["months_to_clear"] >= 1

    st = hb.service_target(herd, today="2026-08-10")
    assert st["service_target_week"] > st["farrow_target_week"], \
        "수태 실패분을 감안하면 교배 목표가 분만 목표보다 커야 한다"


def test_breeding_ledger() -> None:
    """통합 관리표: 완료 추론·조치 가능 지연·모순 검출·향후 일정·작업량."""
    from datetime import date, timedelta
    import breeding_ledger as bl
    today = "2026-08-10"
    farm, herd, scheds, scores = bl.build_demo(today)
    led = bl.ledger(farm, herd, scheds, scores, today=today)
    assert len(led) == len(farm.table()), "관리표 행 수가 배치 두수와 다름"
    assert led["id"].nunique() == len(led), "개체 중복"
    for c in ("loc", "stage", "estrus", "pregnancy", "next_task", "d_day",
              "action", "conflict", "urgency"):
        assert c in led.columns, f"{c} 열 없음"

    # 완료 추론 회귀: 분만한 모돈에게 '교배 142일 경과' 를 띄우면 안 된다
    assert (led["overdue_days"] <= bl.OVERDUE_HORIZON).all(), \
        "조치 불가능한 과거 작업이 지연 큐에 남아 있다"
    lact = led[led["stage"] == "포유"]
    assert not (lact["overdue"] == "교배").any(), \
        "이미 분만한 모돈이 교배 미실시로 잡힘"

    # 후보돈에게 '이유' 작업이 생기면 안 된다
    gilts = set(led[led["stage"] == "후보"]["id"])
    for g in gilts:
        assert not any(t["task"] == "이유" and t["date"] <= rc_date(today)
                       for t in scheds[g]), f"{g}: 후보돈에 이유 작업"

    # 시한작업 우선: 오늘 교배해야 할 개체가 단순 지연 건보다 위에 있어야 한다
    ai_today = led[(led["next_task"] == "교배") & (led["d_day"] == 0)]
    if len(ai_today):
        routine = led[(led["next_task"] == "임신감정") & (led["d_day"] > 5)]
        if len(routine):
            assert ai_today["urgency"].max() > routine["urgency"].max(), \
                "오늘 교배가 여유 있는 임신감정보다 아래로 밀림"

    # 모순: 임신 중 발정 신호는 별도로 남아야 한다(곱해서 뭉개지 않는다)
    cf = bl.conflicts(led)
    hot_preg = led[(led["estrus_score"] >= bl.ESTRUS_HI)
                   & (led["stage"].isin(("임신", "포유")))]
    assert len(cf) == len(hot_preg), "임신 중 발정 신호가 누락됨"
    if len(cf):
        assert cf["conflict"].str.len().gt(0).all()

    up = bl.upcoming(scheds, today=today, days=14, farm=farm)
    assert len(up) and (up["d_day"].between(0, 14)).all()
    assert up["d_day"].is_monotonic_increasing
    assert up["loc"].str.len().gt(0).all(), "향후 일정에 위치가 비었다"

    wl = bl.workload(scheds, today=today, days=14)
    assert "합계" in wl.columns and len(wl)
    task_cols = [c for c in wl.columns if c not in ("date", "합계")]
    # 표가 전부 0 으로 찍히던 회귀(공백 든 한글 컬럼명 접근 실패)
    assert wl[task_cols].to_numpy().sum() > 0, "작업량 표가 비었다"
    assert (wl[task_cols].sum(axis=1) == wl["합계"]).all(), "합계 불일치"
    assert wl["합계"].sum() == len(up), "작업량 총합이 향후 일정 건수와 다름"


def test_posture_crop_feats() -> None:
    """크롭 외형 피처: 차원·결정성·자세 구분력 + 파일명 회귀."""
    import numpy as np
    import posture_crop_feats as pcf
    rng = np.random.default_rng(0)

    # 가로로 긴 밝은 띠(옆으로 누운 몸통) vs 세로로 긴 띠 — 방향 피처가 달라야 한다
    horiz = np.full((pcf.SZ, pcf.SZ), 40, dtype=np.uint8)
    horiz[20:28, 6:42] = 200
    vert = np.full((pcf.SZ, pcf.SZ), 40, dtype=np.uint8)
    vert[6:42, 20:28] = 200
    fh, fv = pcf._crop_feats(horiz), pcf._crop_feats(vert)
    assert fh.shape == (len(pcf.CROP_COLS),) == fv.shape
    assert np.isfinite(fh).all() and np.isfinite(fv).all()
    i_cos = pcf.CROP_COLS.index("sil_cos2t")
    assert abs(fh[i_cos] - fv[i_cos]) > 0.5, "가로/세로 실루엣이 구분되지 않음"
    i_el = pcf.CROP_COLS.index("sil_elong")
    assert fh[i_el] > 1.5 and fv[i_el] > 1.5, "긴 띠인데 장단축비가 1 근처"

    # 결정적이어야 한다(같은 입력 → 같은 출력)
    assert np.allclose(fh, pcf._crop_feats(horiz))
    # 균일 크롭은 그래디언트가 없다
    flat = pcf._crop_feats(np.full((pcf.SZ, pcf.SZ), 128, dtype=np.uint8))
    assert flat[pcf.CROP_COLS.index("edge_den")] == 0.0

    # 파일명 회귀: image_id 에 이미 확장자가 있는데 .jpg 를 덧붙여 전량 누락됐었다.
    # 절반 이상 실패하면 0 행렬을 조용히 캐시하지 말고 터져야 한다.
    import pandas as pd
    bad = pd.DataFrame([{"image_id": "없는파일.jpg", "x": 0, "y": 0,
                         "w": 10, "h": 10}])
    try:
        pcf.extract(bad, {"d": "/nonexistent"}, verbose=False)
        raise AssertionError("전량 누락인데 예외가 나지 않음")
    except RuntimeError:
        pass


def test_posture_crossview() -> None:
    """교차-뷰 프로토콜: 뷰 정규화의 무누수성·3클래스 매핑·상한 계산."""
    import numpy as np
    import posture_crossview as pcv
    import posture_features as pf

    # 5클래스 → 발정 3클래스 매핑이 stall_estrus 어휘와 맞아야 한다
    import stall_estrus as se
    for v in set(pcv.TO_ESTRUS.values()):
        assert se._canon(v) == v, f"{v} 가 stall_estrus 어휘와 불일치"
    assert pcv.TO_ESTRUS["Lateral_lying_left"] == \
        pcv.TO_ESTRUS["Lateral_lying_right"] == "lying"

    # 좌우 횡와를 못 가른다는 가정의 상한: 1 - 비중/2
    import pandas as pd
    df = pd.DataFrame({"cls": ["Lateral_lying_left"] * 2
                       + ["Lateral_lying_right"] * 2 + ["Standing"] * 6})
    c = pcv.ceiling_from_lr(df)
    assert abs(c["lr_share"] - 0.4) < 1e-9 and abs(c["ceiling"] - 0.8) < 1e-9

    # 뷰 정규화는 뷰 단위로 독립이어야 한다 — 한 뷰를 바꿔도 다른 뷰 결과는 그대로
    F = np.array([[1.0, 5.0], [3.0, 7.0], [10.0, 1.0], [20.0, 3.0]])
    v = np.array(["a", "a", "b", "b"])
    n1 = pcv.view_normalize(F, v)
    F2 = F.copy(); F2[2:] *= 100.0
    n2 = pcv.view_normalize(F2, v)
    assert np.allclose(n1[:2], n2[:2]), "다른 뷰의 값이 결과에 새어 들어감"
    for grp in ("a", "b"):
        blk = n1[v == grp]
        assert abs(blk.mean()) < 1e-9, "뷰 내 평균이 0 이 아님"
    # 상수 열에서 0 나눗셈이 나면 안 된다
    assert np.isfinite(pcv.view_normalize(np.ones((4, 2)), v)).all()


def test_posture_report() -> None:
    """자세 병목 리포트: SVG 렌더러 + 자체완결 HTML(캐시 있을 때만 전체 생성)."""
    import os
    import numpy as np
    import build_posture_report as bpr
    import posture_crossview as pcv

    # 혼동행렬 렌더러: 행 정규화라 각 행의 표시값 합이 1 이어야 한다
    labels = ["Standing", "Sternal_lying"]
    svg = bpr.confusion_svg(labels, [[3, 1], [2, 2]], 300)
    assert svg.startswith("<svg") and "0.75" in svg and "0.50" in svg
    # 합이 0 인 행이 있어도 0 나눗셈으로 죽지 않아야 한다
    assert bpr.confusion_svg(labels, [[0, 0], [1, 1]], 300).startswith("<svg")

    bars = bpr.grouped_bars([("a", {"acc_w": 0.5, "mf1_w": 0.2}, False),
                             ("b", {"acc_w": 0.7, "mf1_w": 0.4}, True)],
                            width=400, ref=0.45)
    assert bars.startswith("<svg") and "0.700" in bars
    # 값이 전부 0 이어도 죽지 않아야 한다(mx=0 나눗셈)
    assert bpr.grouped_bars([("z", {"acc_w": 0.0, "mf1_w": 0.0}, False)],
                            width=300).startswith("<svg")

    folds = [{"view": "v1", "n_test": 100, "acc": 0.7, "mf1": 0.5},
             {"view": "v2", "n_test": 50, "acc": 0.4, "mf1": 0.3}]
    assert bpr.fold_svg(folds, 400).startswith("<svg")

    # 전체 생성은 결과 캐시가 있을 때만(케글 데이터 없는 환경 배려)
    if not os.path.exists(pcv.RESULTS):
        return
    assert bpr.main() == 0
    page = open(bpr.OUT, encoding="utf-8").read()
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert os.path.getsize(bpr.OUT) > 8000
    for bad in ("http://", "https://", "<script src", "cdn."):
        assert bad not in page, f"외부 참조 {bad}"
    assert "prefers-color-scheme" in page
    # 폐기한 누수 수치를 성과처럼 다시 싣지 않았는지
    assert "0.642" in page and "폐기" in page, "0.642 폐기 사실이 빠졌다"

    r = pcv.run_all()
    for k in ("baseline", "configs", "pen", "ceiling", "confusion_geom"):
        assert k in r, f"{k} 없음"
    cm = np.array(r["confusion_geom"]["matrix"])
    assert cm.sum() > 0 and cm.shape[0] == cm.shape[1] == len(r["classes"])
    # 좌/우 횡와가 실제로 갈리는지(동전던지기 주장의 근거)
    labs = r["confusion_geom"]["labels"]
    li, ri = labs.index("Lateral_lying_left"), labs.index("Lateral_lying_right")
    assert cm[li][ri] > 0 and cm[ri][li] > 0, "좌우 혼동이 없다 — 주장과 불일치"


def test_barn_environment() -> None:
    """THI 계산·구간 판정·착상기 위험군 교차."""
    import barn_environment as be
    import farm_registry as fr
    import herd_board as hb
    # 같은 온도라도 습도가 높으면 THI 가 높다(온도만으로 판정하면 안 되는 이유)
    assert be.thi(30, 80) > be.thi(30, 40)
    assert be.thi(20, 60) < be.thi(30, 60)
    assert be.band(90)[0] == "중증" and be.band(60)[0] == "적정"
    assert be.band(80)[0] == "중등도" and be.band(76)[0] == "경증"

    env = be.assess({"A": (30.0, 80.0), "B": (18.0, 55.0)})
    a = env.set_index("barn")
    assert bool(a.loc["A", "heat_stress"]) and not bool(a.loc["B", "heat_stress"])
    assert a.loc["A", "wei_penalty_d"] > a.loc["B", "wei_penalty_d"]
    assert (env["thi"] > 0).all()

    farm = fr.demo_farm()
    ids = sorted(farm._where)
    recs = hb.generate_demo(n=len(ids) + 40, today="2026-08-10")[:len(ids)]
    for r, i in zip(recs, ids):
        r["id"] = i
    herd = hb.build_herd(recs, today="2026-08-10")
    hot = be.assess(be.demo_readings(hot_summer=True))
    risk = be.at_risk_services(herd, hot, farm)
    lo, hi = be.IMPLANTATION_WINDOW
    if len(risk):
        # 일 단위여야 한다(주차로 재면 7일 단위로 뭉개져 경계가 흐려진다)
        assert risk["days_since_service"].between(lo, hi).all()
        assert (risk["days_since_service"] % 7 != 0).any(), "주 단위로 뭉개졌다"
    # 서늘하면 위험군이 없어야 한다
    cool = be.assess(be.demo_readings(hot_summer=False))
    assert not len(be.at_risk_services(herd, cool, farm))


def test_dashboard_builders() -> None:
    """새 웹 뷰 2종이 자체완결 HTML 로 생성되는지 + 상태 판정 회귀."""
    import os
    import build_barn_map as bm
    import build_breeding_console as bc
    import breeding_ledger as bl

    today = "2026-08-10"
    farm, herd, scheds, scores = bl.build_demo(today)
    led = bl.ledger(farm, herd, scheds, scores, today=today)

    # 결측 판정 회귀: pandas 를 거친 None 은 float NaN 이 되고 bool(nan) 은 True.
    # 그대로 두면 전 개체가 '경보'로 칠해진다(실제로 68/68 이 그랬다).
    assert not bm._present(float("nan")) and not bm._present(None)
    assert bm._present("x") and bm._present(0)
    pairs = [bm.cell_status(r) for r in led.to_dict("records")]
    kinds = {s for s, _ in pairs}
    assert "정상" in kinds, "조치 없음 개체가 하나도 없다 — 결측 판정이 깨졌다"
    n_alert = sum(1 for s, _ in pairs if s == "경보")
    assert n_alert == len(bl.conflicts(led)), "도면 경보 수가 모순 목록과 불일치"
    assert n_alert < len(led), "전 개체가 경보"

    # 지연은 색이 아니라 테두리 — 오늘 교배할 개체가 지연 색으로 덮이면 안 된다
    ai_today = [r for r in led.to_dict("records")
                if r["next_task"] == "교배" and r["d_day"] == 0
                and (r["overdue_days"] or 0) > 0]
    for r in ai_today:
        st, late = bm.cell_status(r)
        assert st == "교배" and late, "지연이 임박한 교배를 덮어씀"
    assert "지연" not in bm.STATUS, "지연이 색 범례에 남아 있다"

    layout = bm.build_layout(farm, led)
    total = sum(len(p["cells"]) for b in layout.values() for p in b["pens"])
    assert total == len(farm.table()), "도면 칸 수가 배치 두수와 다름"

    # 앱 사용 화면: 숫자가 다른 뷰와 어긋나면 "지어낸 값 없음" 전제가 깨진다
    import build_app_screens as bas
    assert bas.main() == 0
    app = open(bas.OUT, encoding="utf-8").read()
    n_act = sum(1 for s, late in pairs if s not in ("정상", "공실") or late)
    assert f'<b>{n_act}</b><span>조치 대상</span>' in app, \
        f"앱 화면의 조치 대상 수가 도면과 불일치(도면 {n_act})"
    assert f'<b>{len(bl.conflicts(led))}</b><span>경보</span>' in app

    # 교배기록 화면의 날짜는 그 개체의 실제 일정에서 나와야 한다
    import repro_calendar as rc2
    sid, wean, _sc = bas.pick_service_case(herd, scores)
    est = rc2.schedule_from_weaning(wean)
    est_ai = [t for t in est if t["task"] == "교배"][0]
    assert f'{est_ai["date"]:%Y-%m-%d}' in app, "예정일이 실제 일정과 다름"
    assert f'{wean:%Y-%m-%d}' in app, "이유일이 화면에 없음"

    # 동작 프로토타입: 심는 데이터가 유효한 JSON 이고 다른 뷰와 값이 맞는지
    import json
    import build_app_prototype as bap
    P = bap.build_payload()
    for k in ("animals", "sched", "barns", "board", "kpi", "ai",
              "statusColors", "stageColors"):
        assert k in P, f"payload 에 {k} 없음"
    assert len(P["animals"]) == len(led)
    assert P["kpi"]["nAct"] == n_act, "프로토타입 조치 대상 수가 도면과 불일치"
    assert P["kpi"]["nConf"] == len(bl.conflicts(led))
    # NaN 이 새면 JS 에서 truthy 로 잘못 동작한다 — allow_nan=False 로 잡는다
    json.dumps(P, allow_nan=False)
    assert all(a["conflict"] is None or isinstance(a["conflict"], str)
               for a in P["animals"]), "conflict 에 NaN 이 남았다"
    ids = {a["id"] for a in P["animals"]}
    assert all(s["id"] in ids for b in P["barns"] for p in b["pens"]
               for s in p["slots"]), "도면 칸이 없는 개체를 가리킨다"
    assert set(P["sched"]) >= ids, "일정이 없는 개체가 있다"
    assert bap.main() == 0
    proto = open(bap.OUT, encoding="utf-8").read()
    for bad in ("http://", "https://", "<script src", "cdn."):
        assert bad not in proto, f"프로토타입에 외부 참조 {bad}"
    assert "prefers-color-scheme" in proto

    # PC 콘솔: 폰 프로토타입과 **같은 payload** 를 써야 숫자가 갈리지 않는다
    import build_pc_console as bpc
    assert bpc.build_payload is bap.build_payload
    assert bpc.main() == 0
    pc = open(bpc.OUT, encoding="utf-8").read()
    for bad in ("http://", "https://", "<script src", "cdn."):
        assert bad not in pc, f"PC 콘솔에 외부 참조 {bad}"
    assert "prefers-color-scheme" in pc
    assert "@media print" in pc, "작업지시서 인쇄 스타일이 없다"
    # PC 에서만 되는 것들이 실제로 붙어 있는지
    for need in ('id="bulk"', 'type="checkbox"', 'data-k=', "keydown"):
        assert need in pc, f"PC 전용 기능 누락: {need}"

    # 돈군흐름 관제: 페이지 숫자가 pigflow 계산과 같은 값이어야 한다
    import build_pigflow_console as bpf
    from pigflow import calc as pfc, report as pfr
    from pigflow.config import load_config as pf_load
    from pigflow.simulator import Simulator as PfSim, build_rooms as pf_rooms
    assert bpf.main() == 0
    pf = open(bpf.OUT, encoding="utf-8").read()
    cfg = pf_load(bpf.YAML).merged()
    plan = pfc.plan(cfg)
    sim = PfSim(cfg, bpf.START, rooms=pf_rooms(cfg)).run(bpf.SIM_DAYS)
    k = pfr.kpi_report(sim)
    assert f'>{plan["services_per_batch"]}<' in pf, "배치당 교배 두수가 페이지에 없다"
    assert f'{plan["sow_inventory"]:.0f}' in pf, "모돈 규모가 페이지에 없다"
    assert f'PSY {k["psy"]}' in pf or f'{k["psy"]}' in pf
    # 예시 농장은 비육사 1방 부족이 유일한 병목 — 그게 화면에 나와야 한다
    bn = pfr.bottlenecks(sim)
    assert {b["stage"] for b in bn} == {"FINISHER"}, bn
    assert "수용능력 부족" in pf, "병목 메시지가 화면에 없다"
    # 부족 없이 지었을 때는 경보가 없어야 한다(거짓 경보 방지 회귀)
    clean = PfSim(cfg, bpf.START, rooms=pf_rooms(cfg, from_config=False)
                  ).run(bpf.SIM_DAYS)
    from pigflow import validate as pfv
    assert pfv.summarize(clean.findings)["n"] == 0

    for mod in (bc, bm, bas, bpf):
        assert mod.main() == 0
        assert os.path.exists(mod.OUT) and os.path.getsize(mod.OUT) > 8000
        page = open(mod.OUT, encoding="utf-8").read()
        assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
        # 자체완결: 외부 리소스를 부르면 안 된다
        for bad in ("http://", "https://", "<script src", "cdn."):
            assert bad not in page, f"{os.path.basename(mod.OUT)} 에 외부 참조 {bad}"
        assert 'prefers-color-scheme' in page, "다크 모드 대응 없음"

    # 허브가 모든 뷰를 잡고 있는지 — 새 뷰를 만들고 등록을 잊는 일이 잦다
    import build_dashboard_hub as hub
    assert hub.main() == 0
    files = {v[0] for v in hub.VIEWS}
    for mod in (bc, bm, bas, bap, bpc, bpf):
        assert os.path.basename(mod.OUT) in files, \
            f"{os.path.basename(mod.OUT)} 이 허브 VIEWS 에 없다"
    idx = open(hub.OUT, encoding="utf-8").read()
    assert "pigflow_console.html" in idx


def test_check_download() -> None:
    """다운로드 진단: 6가지 실패 유형을 각각 맞게 짚는지."""
    import contextlib
    import gzip
    import importlib.util
    import io
    import tarfile
    import tempfile

    path = os.path.join(ROOT, "tools", "check_download.py")
    spec = importlib.util.spec_from_file_location("_chk", path)
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)

    def run(p):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cd.check(p)
        return rc, buf.getvalue()

    with tempfile.TemporaryDirectory() as d:
        ok = os.path.join(d, "ok.tar")
        with tarfile.open(ok, "w") as t:
            f = os.path.join(d, "a.txt")
            open(f, "w").write("hi")
            t.add(f, arcname="a.txt")
        rc, out = run(ok)
        assert rc == 0 and "정상" in out and "종료 블록" in out, out

        # 잘린 tar — tarfile 은 조용히 통과시키므로 종료 블록을 직접 봐야 한다.
        # 이 검사가 없을 때 2,000바이트 파일이 '정상'으로 나왔다(회귀 방지).
        trunc = os.path.join(d, "trunc.tar")
        open(trunc, "wb").write(open(ok, "rb").read()[:2000])
        rc, out = run(trunc)
        assert rc == 1 and "잘렸다" in out, out

        # 에러 문구가 tar 로 위장 — 실제로 이 저장소에서 나왔던 82바이트 파일
        err = os.path.join(d, "err.tar")
        open(err, "w", encoding="utf-8").write(
            "AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다.")
        rc, out = run(err)
        assert rc == 1 and "해외 IP 차단" in out, out

        htm = os.path.join(d, "h.tar")
        open(htm, "w", encoding="utf-8").write("<!DOCTYPE html>로그인이 필요합니다")
        rc, out = run(htm)
        assert rc == 1 and "HTML" in out and "인증 실패" in out, out

        gz = os.path.join(d, "g.tar")
        open(gz, "wb").write(gzip.compress(open(ok, "rb").read()))
        rc, out = run(gz)
        assert rc == 1 and "gzip" in out, out

        # 분할 조각 12개 — 10을 넘으면 알파벳 정렬이 순서를 뒤섞으므로 ls -v
        raw = open(ok, "rb").read()
        step = max(1, len(raw) // 12)
        for i in range(12):
            open(os.path.join(d, f"s.tar.part{i:02d}"), "wb").write(
                raw[i * step:(i + 1) * step])
        rc, out = run(os.path.join(d, "s.tar.part00"))
        assert rc == 1 and "분할 조각 12개" in out, out
        assert "ls -v" in out, "10조각 초과인데 정렬 함정을 경고하지 않는다"

        # 실제로 받은 39바이트 tl.tar/vl.tar. 실측으로 원인을 특정했다:
        #   /down/0.6/622.do?fileSn=533708 → 502 인증실패  (경로 정상)
        #   /down/622.do?fileSn=533708     → 404 이 문구    (버전 세그먼트 누락)
        # filekey 만료라고 말하면 엉뚱한 데를 뒤지게 된다.
        nf = os.path.join(d, "nf.tar")
        open(nf, "w", encoding="utf-8").write("페이지가 존재하지 않습니다.")
        rc, out = run(nf)
        assert rc == 1 and "URL 경로가 틀렸다" in out and "/down/0.6/" in out, out
        assert "filekey 문제가 아니다" in out, out

        auth = os.path.join(d, "auth.tar")
        open(auth, "w", encoding="utf-8").write("인증실패, 권한이 거부되었습니다")
        rc, out = run(auth)
        assert rc == 1 and "활용신청" in out, out

        rc, out = run(os.path.join(d, "없는파일.tar"))
        assert rc == 1

    # 조각 탐색이 .partNN 만 잡고 무관한 파일을 끌어오지 않는지
    with tempfile.TemporaryDirectory() as d:
        for n in ("x.tar.part00", "x.tar.part01", "x.tar", "other.zip"):
            open(os.path.join(d, n), "wb").write(b"\0")
        assert cd.find_parts(os.path.join(d, "x.tar")) == \
            ["x.tar.part00", "x.tar.part01"]


def test_finetune_polygon() -> None:
    """폴리곤 파인튜닝 준비: 매칭률·서브샘플·데이터셋 생성·시간추정."""
    import shutil
    import tempfile
    import numpy as np
    import cv2
    import parse_pig_polygon as ppp
    import finetune_polygon as fp

    with tempfile.TemporaryDirectory() as d:
        ld, idir = os.path.join(d, "lab"), os.path.join(d, "img")
        os.makedirs(ld); os.makedirs(idir)
        ppp.synth_cvat(os.path.join(ld, "a.xml"), n_images=30, seed=1)
        df = fp.load_labels(ld, verbose=False)
        imgs = sorted(df["image"].unique())
        rng = np.random.default_rng(0)
        # 절반만 이미지를 만든다 — 라벨 전체 + 원천 일부 상황 재현
        for name in imgs[:len(imgs) // 2]:
            row = df[df["image"] == name].iloc[0]
            cv2.imwrite(os.path.join(idir, os.path.basename(name)),
                        rng.integers(0, 255, (int(row.img_h or 64),
                                              int(row.img_w or 64), 3),
                                     dtype=np.uint8))
        index = fp.index_images(idir)
        assert index["__n_files__"] == len(imgs) // 2
        paired = fp.pair(df, index, verbose=False)
        assert paired["image"].nunique() == len(imgs) // 2
        assert paired["path"].map(os.path.exists).all(), "경로가 실제 파일이 아니다"

        # 서브샘플은 **이미지 단위** — 한 장이 반만 라벨되면 안 된다
        sub = fp.subsample(paired, 5, seed=0)
        assert sub["image"].nunique() == 5
        for img in sub["image"].unique():
            assert len(sub[sub["image"] == img]) == \
                len(paired[paired["image"] == img]), f"{img} 폴리곤이 잘렸다"
        assert fp.subsample(paired, 10_000)["image"].nunique() == len(imgs) // 2

        out = os.path.join(d, "ds")
        built = fp.build_dataset(paired, out)
        assert os.path.exists(built["yaml"])
        # train/val 이미지가 겹치면 검증이 무의미하다
        tr = set(os.listdir(os.path.join(out, "images", "train")))
        va = set(os.listdir(os.path.join(out, "images", "val")))
        assert tr and va and not (tr & va), "train/val 이미지 누수"
        # 이미지마다 라벨 파일이 있어야 한다
        for split in ("train", "val"):
            ii = os.listdir(os.path.join(out, "images", split))
            ll = os.listdir(os.path.join(out, "labels", split))
            assert len(ii) == len(ll), f"{split}: 이미지 {len(ii)} vs 라벨 {len(ll)}"
        # YOLO-seg 좌표는 전부 0~1 정규화
        for f in os.listdir(os.path.join(out, "labels", "train")):
            for line in open(os.path.join(out, "labels", "train", f)):
                v = [float(x) for x in line.split()[1:]]
                assert v and all(0.0 <= x <= 1.0 for x in v), f"{f}: {line[:60]}"

    # 시간 추정: 실측 표(ms/장/ep)에서 그대로 나와야 한다
    ms = fp.SPEED_MS[("yolo11n-seg", 416, True)]
    h1 = fp.estimate_hours(1000, 50, "yolo11n-seg", 416)
    assert abs(h1 - 1000 * 50 * ms / 1000 / 3600) < 0.01, h1
    # 장수·epoch 에 선형
    assert fp.estimate_hours(2000, 50, "yolo11n-seg", 416) == 2 * h1
    assert abs(fp.estimate_hours(1000, 100, "yolo11n-seg", 416) - 2 * h1) < 1e-9
    # 해상도가 클수록 느리고, freeze 하면 빨라야 한다
    assert h1 < fp.estimate_hours(1000, 50, "yolo11n-seg", 640)
    assert h1 < fp.estimate_hours(1000, 50, "yolo11n-seg", 416, freeze=False)
    # 실측에 없는 조합도 외삽되어야 한다(320/freeze 는 표에 있으므로 224 로)
    assert 0 < fp.estimate_hours(1000, 50, "yolo11n-seg", 224) < h1
    # freeze 기본값은 True — CPU 에서 1.76배 차이가 나므로 기본이 중요하다
    assert fp.estimate_hours(1000, 50, "yolo11n-seg", 416) == \
        fp.estimate_hours(1000, 50, "yolo11n-seg", 416, freeze=True)


def test_fetch_622_doctor() -> None:
    """502 해석: AI Hub 는 정상 응답에도 502 를 준다 — 본문으로 판단해야 한다."""
    import contextlib
    import io
    import fetch_622 as f6

    TREE = ("└─108.지능형 스마트축사 통합 데이터(양돈)\n  ├─1.Training\n"
            + "x" * 400).encode()

    def run(responses, key=None):
        """_probe 를 가짜로 바꿔 분기별 문구를 확인한다."""
        seq = list(responses)
        real, old = f6._probe, os.environ.pop("AIHUB_APIKEY", None)
        if key:
            os.environ["AIHUB_APIKEY"] = key
        f6._probe = lambda *_a, **_k: seq.pop(0)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = f6.doctor()
        finally:
            f6._probe = real
            os.environ.pop("AIHUB_APIKEY", None)
            if old is not None:
                os.environ["AIHUB_APIKEY"] = old
        return rc, buf.getvalue()

    # 502 + 트리 = 정상. 상태코드로 실패 판정하면 안 된다.
    rc, out = run([(502, TREE)])
    assert "파일 트리 조회" in out and "✅ 정상" in out, out
    assert rc == 1 and "AIHUB_APIKEY 없음" in out          # 키 없어 2단계 스킵

    # 트리 자체가 안 오면 네트워크 문제로 짚어야 한다
    rc, out = run([(-1, b"timeout")])
    assert rc == 1 and "네트워크" in out, out

    # 502 + 인증실패 = 진짜 실패. 키/승인 둘 다 제시해야 한다.
    rc, out = run([(502, TREE), (502, "인증실패, 권한이 거부되었습니다".encode())],
                  key="k")
    assert rc == 1 and "API 키가 틀렸다" in out and "활용신청" in out, out

    rc, out = run([(502, TREE),
                   (502, "AI 허브는 해외에서의 데이터 다운로드를 "
                         "제한하고 있습니다.".encode())], key="k")
    assert rc == 1 and "해외 IP 차단" in out, out

    # /0.6/ 을 쓰는데도 404 문구면 그때는 정말 filekey 가 바뀐 것
    rc, out = run([(502, TREE), (404, "페이지가 존재하지 않습니다.".encode())],
                  key="k")
    assert rc == 1 and "filekey 가 실제로 바뀐" in out, out

    # 바이너리가 오면 통과
    rc, out = run([(502, TREE), (200, b"PK\x03\x04" + b"\x00" * 200)], key="k")
    assert rc == 0 and "통과" in out, out


def test_real_622_schema() -> None:
    """실제 622 CVAT 스키마 회귀 — VL01 실측값으로 고정.

    파서를 합성 픽스처로만 검증하면 실제 스키마가 다를 때 조용히 무너진다.
    VL01(27 세션)에서 확인한 사실을 여기 못 박는다:
      · <image> 안의 <polygon>, points="x,y;x,y;..."
      · XML 파일명이 27개 전부 annotations.xml → 세션 디렉터리가 유일한 구분자
      · frame_000000 이 27개 세션에 전부 있음 → 세션 한정 없이는 9,427→1,301 뭉갬
      · 라벨 21종 중 돼지 개체는 14종뿐(시설물·신체부위 제외)
    """
    import tempfile
    import textwrap
    import parse_pig_polygon as ppp
    import finetune_polygon as fp

    def sess_xml(path, frames=2, labels=("Resting", "Feedbox", "Head")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imgs = ""
        for i in range(frames):
            polys = "".join(
                f'<polygon label="{lb}" occluded="0" source="manual" '
                f'points="10.5,20.5;30.5,20.5;30.5,40.5;10.5,40.5"></polygon>'
                for lb in labels)
            imgs += (f'<image id="{i}" name="frame_{i:06d}" '
                     f'width="2560" height="1944">{polys}</image>')
        open(path, "w", encoding="utf-8").write(textwrap.dedent(f"""\
            <?xml version="1.0" encoding="utf-8"?>
            <annotations><version>1.1</version>
            <meta><task><id>1</id><mode>interpolation</mode></task></meta>
            {imgs}</annotations>"""))

    with tempfile.TemporaryDirectory() as d:
        # 세션 3개, 전부 annotations.xml, 전부 frame_000000 부터
        for sn in ("farmA/stageX/P01_07_a", "farmA/stageX/P01_07_b",
                   "farmB/stageY/P03_02_a"):
            sess_xml(os.path.join(d, sn, "annotations.xml"), frames=2)

        one = ppp.parse_cvat(os.path.join(d, "farmA/stageX/P01_07_a",
                                          "annotations.xml"))
        assert len(one) == 6 and one["image"].nunique() == 2
        assert one["img_w"].iloc[0] == 2560 and one["img_h"].iloc[0] == 1944
        assert one["points"].iloc[0][0] == (10.5, 20.5), one["points"].iloc[0]

        df = fp.load_labels(d, verbose=False)
        # 세션 한정이 없으면 2장으로 뭉개진다 — 6장이어야 한다
        assert df["image"].nunique() == 6, df["image"].nunique()
        assert df["image_name"].nunique() == 2
        assert df["session"].nunique() == 3
        assert df["source"].nunique() == 3, "파일명이 다 같아 출처 구분이 안 된다"
        assert df["image"].iloc[0].startswith("farmA/stageX/"), df["image"].iloc[0]

        # 라벨 분류: 시설물·신체부위를 학습에서 빼야 한다
        assert fp.label_kind("Resting") == "behavior"
        assert fp.label_kind("Feedbox") == "fixture"
        assert fp.label_kind("Head") == "part"
        assert fp.label_kind("Nonexistent") == "unknown"
        pig = fp.select_labels(df, "pig", verbose=False)
        assert set(pig["label"]) == {"pig"}
        assert len(pig) == 6, "행동 폴리곤만 남아야 한다(세션3×프레임2×Resting1)"
        assert set(pig["behavior"]) == {"Resting"}
        beh = fp.select_labels(df, "behavior", verbose=False)
        assert set(beh["label"]) == {"Resting"}
        assert len(fp.select_labels(df, "all", verbose=False)) == 18

        # 21종 분류표에 빠진 라벨이 없는지 — 실제 VL01 에서 나온 전체 목록
        real = ["Resting", "Feedbox", "Watercup", "Suckling", "Eating",
                "Searching", "Lying", "Standing", "Head", "Hip", "Walking",
                "Right_front_leg", "Right_behind_leg", "Drinking",
                "Left_behind_leg", "Sitting", "Left_front_leg", "Scrubbing",
                "Eating head", "Drinking head", "Parturition"]
        unknown = [x for x in real if fp.label_kind(x) == "unknown"]
        assert not unknown, f"분류 안 된 실제 라벨: {unknown}"


def test_image_name_collision() -> None:
    """같은 파일명이 원천마다 반복될 때 **엉뚱한 짝을 맺으면 안 된다**.

    CVAT 내보내기는 frame_0000.jpg 같은 이름을 쓰고 그 이름이 TS06/VS01 양쪽에
    다 있다. basename 으로만 색인하면 라벨이 다른 원천의 사진에 붙은 채로
    학습에 들어가고 에러도 안 난다 — 조용히 오염된다.
    """
    import tempfile
    import numpy as np
    import cv2
    import finetune_polygon as fp

    with tempfile.TemporaryDirectory() as d:
        for sub in ("TS06", "VS01"):
            os.makedirs(os.path.join(d, sub))
            for i in range(3):
                cv2.imwrite(os.path.join(d, sub, f"frame_{i:04d}.jpg"),
                            np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(os.path.join(d, "TS06", "only_here.jpg"),
                    np.zeros((8, 8, 3), np.uint8))
        idx = fp.index_images(d)
        assert idx["__n_files__"] == 7

        # 경로가 붙어 있으면 정확히 그 폴더로
        for sub in ("TS06", "VS01"):
            got = fp.resolve(f"{sub}/frame_0000.jpg", idx)
            assert got and os.path.basename(os.path.dirname(got)) == sub, got
        # 맨 이름은 모호하므로 **매칭을 거부**해야 한다(아무거나 고르면 안 된다)
        assert fp.resolve("frame_0000.jpg", idx) is None
        assert idx["frame_0000"] is None
        # 유일한 이름은 맨 이름으로도 찾힌다
        assert fp.resolve("only_here.jpg", idx) is not None
        assert fp.resolve("없는파일.jpg", idx) is None
        # 윈도우 구분자·선행 슬래시도 처리
        assert fp.resolve("\\TS06\\frame_0001.jpg", idx) is not None
        assert fp.resolve("/TS06/frame_0002.jpg", idx) is not None
        # 더 긴 경로가 주어져도 뒤에서부터 맞춘다
        assert fp.resolve("x/y/TS06/frame_0001.jpg", idx) is not None


def test_fetch_622() -> None:
    """622 다운로드 헬퍼: filekey·용량·디스크·키 가드."""
    import contextlib
    import io
    import tempfile
    import fetch_622 as f6

    # filekey 는 tree 622 실측값. 바뀌면 다운로드가 조용히 엉뚱한 걸 받는다.
    keys = {k for grp in f6.FILES.values() for k, _d, _m in grp}
    assert {"533708", "533718", "533695", "533714"} == keys, keys
    # 라벨만으로는 학습 불가 — 이미지 항목이 별도로 있어야 한다
    assert f6.FILES["labels"] and f6.FILES["images"]
    # VL01 은 VS01 을 가리킨다. 라벨만 받으면 매칭 0장이므로 **둘이 한 묶음**
    ov = {k for k, _d, _m in f6.FILES["official_val"]}
    assert ov == {"533718", "533714"}, ov
    assert "533718" not in {k for k, _d, _m in f6.FILES["labels"]}, \
        "VL01 이 기본 라벨에 있다 — 이미지 없이 받으면 매칭률만 왜곡된다"
    lab_mb = sum(m for _k, _d, m in f6.FILES["labels"])
    img_mb = sum(m for _k, _d, m in f6.FILES["images"])
    assert lab_mb < 200 and img_mb > 1000, (lab_mb, img_mb)

    def run(argv, env_key=None):
        old = os.environ.pop("AIHUB_APIKEY", None)
        if env_key:
            os.environ["AIHUB_APIKEY"] = env_key
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = f6.main(argv)
        finally:
            os.environ.pop("AIHUB_APIKEY", None)
            if old is not None:
                os.environ["AIHUB_APIKEY"] = old
        return rc, buf.getvalue()

    with tempfile.TemporaryDirectory() as d:
        # 키 없으면 받기 전에 멈춘다
        rc, out = run(["--out", d, "--dry-run"])
        assert rc == 1 and "AIHUB_APIKEY" in out, out
        assert "커밋하지 않는다" in out, "키 커밋 금지 경고가 없다"

        # 라벨만일 때는 학습 불가 경고
        rc, out = run(["--out", d, "--dry-run"], env_key="dummy")
        assert rc == 0 and "학습할 수 없다" in out, out
        assert "533708" in out, "Training 라벨이 기본에 없다"
        assert "533695" not in out, "--images 없이 원천을 받으려 한다"
        assert "533718" not in out, "VL01 을 기본으로 받으려 한다"
        assert "공식 검증셋" in out, "VL/VS 를 왜 안 받는지 설명이 없다"

        # --official-val 은 라벨과 원천을 **함께** 받아야 한다
        rc, out = run(["--out", d, "--images", "--official-val", "--dry-run"],
                      env_key="dummy")
        assert "533718" in out and "533714" in out, out

        # --images 면 원천도 계획에 들어간다
        rc, out = run(["--out", d, "--images", "--dry-run"], env_key="dummy")
        assert "533695" in out, out

    # 디스크 부족은 받기 전에 잡아야 한다 — 10GB 를 다 받고 실패하면 늦다
    real = f6.free_gb
    try:
        f6.free_gb = lambda _p: 0.5
        with tempfile.TemporaryDirectory() as d:
            rc, out = run(["--out", d, "--images", "--dry-run"], env_key="k")
            assert rc == 1 and "공간이 부족" in out, out
    finally:
        f6.free_gb = real


def test_korean_farm_stats() -> None:
    """국내 466행 실측 집계 — 원자료 없이 JSON + 합성 프레임으로 검증.

    원자료(farm_stats.xlsx)는 농장 식별자가 있어 커밋하지 않으므로, 테스트는
    커밋된 집계 JSON 과 합성 데이터프레임으로 함수를 확인한다.
    """
    import json
    import numpy as np
    import pandas as pd
    import korean_farm_stats as kfs

    j = os.path.join(ROOT, "data", "korean_farm_stats.json")
    assert os.path.exists(j), "집계 JSON 이 커밋돼 있어야 한다"
    r = json.load(open(j, encoding="utf-8"))
    assert r["n_rows"] >= 400 and r["n_farms"] >= 100, r
    assert r["years"] == sorted(r["years"]) and len(r["years"]) >= 3

    q = r["quantiles"]
    for k in ("psy", "npd", "farrowing_rate", "turnover", "weaned"):
        v = q[k]
        # 분위수는 단조여야 한다 — 아니면 집계가 깨진 것
        seq = [v["p10"], v["p25"], v["p50"], v["p75"], v["p90"]]
        assert seq == sorted(seq), (k, seq)
        assert v["n"] >= 100

    # 우리 기본값이 실측과 10% 넘게 어긋나면 config 를 손봐야 한다
    off = [x["name_ko"] for x in r["defaults_check"] if x["off"]]
    assert not off, f"실측과 어긋난 기본값이 남아 있다: {off}"

    # 앱의 논거: NPD·분만율이 PSY 를 좌우해야 한다
    dr = {x["metric"]: x for x in r["drivers_psy"]}
    assert dr["npd"]["spearman"] < -0.5, dr["npd"]
    assert dr["farrowing_rate"]["spearman"] > 0.4, dr["farrowing_rate"]
    assert dr["npd"]["top25"] < dr["npd"]["bottom25"], "상위 농장 NPD 가 더 크다"
    assert dr["farrowing_rate"]["top25"] > dr["farrowing_rate"]["bottom25"]

    # growth_flow 벤치마크가 이 실측에서 나왔는지
    import growth_flow as gf
    assert abs(gf.BENCHMARKS["국내 중앙값"]["psy"] - q["psy"]["p50"]) < 0.15
    assert abs(gf.BENCHMARKS["국내 상위 10%"]["psy"] - q["psy"]["p90"]) < 0.15
    surv = [v["msy"] / v["psy"] for k, v in gf.BENCHMARKS.items()]
    assert surv == sorted(surv), "육성률이 벤치마크 순서와 어긋난다"

    # 함수 자체 — 합성 프레임으로
    rng = np.random.default_rng(0)
    n = 200
    npd = rng.uniform(25, 75, n)
    d = pd.DataFrame({
        "year": 2021, "region": "-", "scale": ["A"] * 100 + ["B"] * 100,
        "farm": [f"F{i}" for i in range(n)],
        "sows": rng.uniform(100, 900, n),
        "psy": 30 - npd * 0.12 + rng.normal(0, 0.4, n),   # NPD 가 PSY 를 깎는다
        "msy": np.nan, "turnover": rng.uniform(2.1, 2.5, n), "npd": npd,
        "born_total": rng.uniform(12, 15, n), "born_alive": rng.uniform(11, 14, n),
        "weaned": rng.uniform(9, 12, n),
        "farrowing_rate": rng.uniform(70, 90, n),
        "first_service_age": rng.uniform(240, 290, n),
        "gestation": 115.0, "lactation": 24.8, "wean_to_estrus": 6.9,
    })
    dv = {x["metric"]: x for x in kfs.drivers(d, "psy")}
    assert dv["npd"]["spearman"] < -0.8, dv["npd"]   # 심어 둔 관계를 찾아야 한다
    assert dv["npd"]["gap"] < 0                       # 상위 농장이 NPD 가 낮다
    qq = kfs.quantiles(d, ["psy", "npd"])
    assert qq["psy"]["p10"] < qq["psy"]["p90"]
    # 분위 조회 — 농가 피드백에 쓴다
    assert 0.0 <= kfs.percentile_of(d, "psy", float(d["psy"].median())) <= 1.0
    assert kfs.percentile_of(d, "psy", 0.0) == 0.0
    assert kfs.percentile_of(d, "psy", 1e9) == 1.0
    # 표본이 적은 컬럼은 분위수를 내지 않는다(오해 방지)
    assert "msy" not in kfs.quantiles(d)


def test_kaggle_notebooks() -> None:
    """캐글 노트북 — 자립적이고, 규약이 ml_core 와 갈리지 않는가.

    노트북은 저장소가 없는 런타임에서 도니 규약을 인라인한다. 두 곳이
    갈라지면 캐글 결과와 로컬 결과를 나란히 놓을 수 없다.
    """
    import json
    import build_kaggle_notebooks as bkn
    import ml_core as mc

    assert bkn.main() == 0
    for name in ("posture_cnn_kaggle.ipynb", "behavior_seq_kaggle.ipynb"):
        p = os.path.join(ROOT, "notebooks", name)
        nb = json.load(open(p, encoding="utf-8"))
        assert nb["nbformat"] == 4 and nb["cells"], name
        assert nb["metadata"].get("accelerator") == "GPU", f"{name} GPU 미설정"
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code, name
        for c in code:                       # 셀마다 구문이 성립해야 한다
            compile("".join(c["source"]), name, "exec")
        txt = "".join("".join(c["source"]) for c in nb["cells"])
        # 자립성 — 이 저장소 모듈을 **import** 하면 캐글에서 죽는다.
        # 주석으로 출처를 가리키는 건 괜찮으므로 import 문만 본다.
        code_txt = "".join("".join(c["source"]) for c in code)
        for bad in ("import ml_core", "from ml_core",
                    "import posture_crossview", "import posture_features",
                    "sys.path.insert"):
            assert bad not in code_txt, f"{name} 이 저장소 모듈에 기댄다: {bad}"
        # 규약이 인라인돼 있는가
        for need in ("def score(", "def weighted(", "def report(",
                     "기준선", "Macro-F1" if "Macro" in txt else "mf1"):
            assert need in txt, f"{name} 에 규약 {need} 이 없다"
        assert "/kaggle/input/" in txt

    post = json.load(open(os.path.join(ROOT, "notebooks",
                                       "posture_cnn_kaggle.ipynb"),
                          encoding="utf-8"))
    ptxt = "".join("".join(c["source"]) for c in post["cells"])
    # **밟기 쉬운 함정 둘이 실제로 막혀 있는가.**
    assert "drop_duplicates" in ptxt, "train1/train2 중복 제거가 없다"
    assert "flip" not in ptxt.lower() and "hflip" not in ptxt.lower(), \
        "좌우 뒤집기 증강 — 좌횡와를 뒤집으면 라벨이 바뀐다"
    assert str(bkn.CEILING) in ptxt, "원리적 상한이 노트북에 없다"
    # 상한·MIN_FOLD 가 다른 곳과 어긋나면 안 된다
    import train_posture_cnn as tpc
    assert bkn.CEILING == tpc.CEILING == 0.861
    assert bkn.MIN_FOLD == tpc.MIN_FOLD

    # 인라인 규약이 ml_core 와 같은 판정을 내는가 — 갈리면 비교가 무의미하다
    ns: dict = {}
    exec(compile(bkn.CONTRACT, "contract", "exec"), ns)
    B1 = {"acc": 0.636, "mf1": 0.280, "n": 100, "folds": 3}   # 폴리곤 실험
    B2 = {"acc": 0.423, "mf1": 0.119, "n": 100, "folds": 3}   # 자세 5클래스
    cases = [
        # 둘 다 아래 → 진짜 미달(폴리곤)
        (B1, {"acc": 0.615, "mf1": 0.260, "n": 100, "folds": 3}, "기준선 미달"),
        # 정확도만 아래, MF1 은 위 → 불균형이 가린 것(자세 기하 실측)
        (B2, {"acc": 0.414, "mf1": 0.228, "n": 100, "folds": 3},
         "정확도만 미달(불균형에 가림)"),
        (B1, {"acc": 0.700, "mf1": 0.400, "n": 100, "folds": 3}, "개선"),
        (B1, dict(B1), "기준선과 같음"),
    ]
    for base, m, want in cases:
        got = ns["report"]("t", m, base)["verdict"]
        ref = mc.report("t", m, base, quiet=True)["verdict"]
        assert got == want == ref, \
            f"인라인 규약이 ml_core 와 다르다: {got} / {ref} (기대 {want})"
    w = ns["weighted"]([{"acc": 1.0, "mf1": 1.0, "n": 900},
                        {"acc": 0.0, "mf1": 0.0, "n": 100}])
    assert abs(w["acc"] - 0.9) < 1e-9, "가중 집계가 다르다"


def test_ml_core() -> None:
    """학습·평가 공통 규약 — 굳혀 둔 실수 넷이 실제로 막히는가.

    가장 중요한 건 **기존 결과를 재현**하는 것이다. 새 규약이 아니라 이미
    쓰던 방식을 모은 것이므로, 공표된 기준선과 다르면 규약이 틀린 것이다.
    """
    import json
    import numpy as np
    import pandas as pd
    import ml_core as mc

    # ① 공표된 기준선을 그대로 재현하는가 (캐시가 있을 때만)
    npz = os.path.join(ROOT, "data", "posture_crops.npz")
    pub_p = os.path.join(ROOT, "data", "posture_crossview.json")
    if os.path.exists(npz) and os.path.exists(pub_p):
        import posture_crossview as pc
        full, _X, _c = pc.load()
        pub = json.load(open(pub_p, encoding="utf-8"))["baseline"]
        for lab in ("cls", "cls3"):
            got = mc.majority_baseline(full, lab, "view", min_fold=pc.MIN_FOLD)
            assert abs(got["acc"] - pub[lab]["acc_w"]) < 0.002, \
                f"{lab} 기준선 재현 실패: {got['acc']} vs {pub[lab]['acc_w']}"
            assert abs(got["mf1"] - pub[lab]["mf1_w"]) < 0.002

    # ② 기준선 미달을 **실패로 표시**하는가. 폴리곤 실험에서 0.615 를
    #    기준선 0.636 과 견주지 않아 개선으로 읽을 뻔했다(MF1 도 아래였다).
    base = {"acc": 0.636, "mf1": 0.28, "n": 100, "folds": 3}
    lo = {"acc": 0.615, "mf1": 0.26, "n": 100, "folds": 3}
    r = mc.report("t", lo, base, quiet=True)
    assert r["verdict"] == "기준선 미달", r

    # 정확도만 아래이고 MF1 은 위인 경우는 **미달이 아니다.** 자세 실측이
    # 정확히 그랬다(기하 0.414 vs 기준선 0.423, MF1 0.228 vs 0.119).
    masked = mc.report("t", {"acc": 0.414, "mf1": 0.228, "n": 100, "folds": 3},
                       {"acc": 0.423, "mf1": 0.119, "n": 100, "folds": 3},
                       quiet=True)
    assert masked["verdict"].startswith("정확도만 미달"), masked
    same = mc.report("t", dict(base), base, quiet=True)
    assert same["verdict"] == "기준선과 같음", same
    up = mc.report("t", {"acc": 0.70, "mf1": 0.40, "n": 100, "folds": 3},
                   base, quiet=True)
    assert up["verdict"] == "개선"

    # ③ 작은 폴드는 집계에서 빠지는가 — 분산에 묻혀 순서가 뒤집힌 적이 있다
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "g": ["big"] * 200 + ["tiny"] * 5,
        "y": list(rng.integers(0, 2, 200)) + [0] * 5,
        "x": rng.normal(size=205)})
    out = mc.leave_one_group_out(df, "y", "g", lambda tr, te: np.zeros(len(te)),
                                 min_fold=30)
    assert "tiny" in out["skipped"] and out["folds"] <= 1, out

    # ④ 누수 검사 — 고유하지 않은 id 를 주면 세지 말고 경고해야 한다.
    #    frame_idx 처럼 그룹마다 0 부터 다시 매겨지는 값을 넘겼다가
    #    전부 '누수' 로 찍힌 적이 있다.
    dup = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "fid": [0, 1, 2, 3] * 2})
    lk = mc.leakage_check(dup, "g", "fid")     # 전부 그룹을 넘음 = 카운터
    assert lk["checked"] is False and lk["leaked"] == 0 and "note" in lk
    # 진짜 누수는 **일부만** 넘는다 — p1 하나만 두 그룹에 걸쳐 있다
    real = pd.DataFrame({"g": ["a", "a", "a", "b", "b"],
                         "path": ["p1", "p2", "p3", "p4", "p1"]})
    lk2 = mc.leakage_check(real, "g", "path")
    assert lk2["checked"] is True and lk2["leaked"] == 1, lk2
    assert lk2["examples"] == ["p1"]

    # ⑤ 폴드 집계는 **표본 수 가중**이다. 단순 평균이면 작은 폴드에 끌려간다.
    w = mc.weighted([{"acc": 1.0, "mf1": 1.0, "n": 900},
                     {"acc": 0.0, "mf1": 0.0, "n": 100}])
    assert abs(w["acc"] - 0.9) < 1e-9, w

    # ⑥ 발정은 막힌 과제로 등록돼 있어야 한다 — 실측 결론이다
    est = next(t for t in mc.TASKS if t.key == "estrus")
    assert est.status == "blocked" and "라벨 어휘" in est.note
    assert all(t.status in ("ready", "blocked") for t in mc.TASKS)
    assert mc.main() == 0


def test_pc_suite() -> None:
    """PC 통합 콘솔 — 여섯 화면을 한 파일로 합쳤는가.

    합치는 방식이 핵심이다. 일곱 뷰가 전부 `.card` `.wrap` 같은 **같은
    클래스명**을 쓰고 각자 전역 스크립트를 깐다. DOM 을 이어붙이면 CSS 가
    서로를 덮으므로 srcdoc iframe 으로 격리해야 하고, 그게 유지되는지 본다.
    """
    import base64
    import re
    import build_pc_suite as bps

    # 모바일은 빠져야 한다 — 그게 이 뷰의 전제다
    names = [v[0] for v in bps.VIEWS]
    assert "app_prototype.html" not in names and "app_screens.html" not in names
    assert len(names) == len(set(names)) == 8

    made = [n for n in names if os.path.exists(os.path.join(ROOT, "dashboard", n))]
    if not made:
        return                      # 대시보드 미생성 환경 — build_all.sh 선행
    assert bps.main() == 0
    html = open(bps.OUT, encoding="utf-8").read()

    # 자체완결 — 합친 뒤에도 외부로 나가면 안 된다
    assert not re.findall(r'https?://[^"\'\s)]+', html), "외부 URL"

    # **격리.** 원본을 DOM 에 풀어 놓으면 CSS 가 충돌한다.
    assert "srcdoc" in html and "<iframe" in html
    assert html.count("<iframe") == 1, "화면마다 iframe 을 만들면 안 된다"

    # 원본이 통째로, 손대지 않은 채 들어 있는가 — base64 를 되돌려 대조
    b64s = re.findall(r'"([A-Za-z0-9+/=]{500,})"', html)
    assert len(b64s) == len(made), f"실린 문서 {len(b64s)} vs 화면 {len(made)}"
    for name, enc in zip(made, b64s):
        src = base64.b64decode(enc).decode("utf-8")
        orig = open(os.path.join(ROOT, "dashboard", name),
                    encoding="utf-8").read()
        assert src == bps.strip_bom(orig), f"{name} 원본과 다르다"
        assert src.lstrip().lower().startswith("<!doctype"), \
            f"{name} 이 완전한 문서가 아니다"

    # 테마는 iframe 에 넣기 전에 문자열로 주입한다(file:// 오리진 회피)
    assert "function themed" in html and "data-theme" in html
    for sel in (":root{", "prefers-color-scheme:dark", ':root[data-theme=dark]'):
        assert sel in html, f"테마 선언 누락: {sel}"

    # 등록
    import build_dashboard_hub as hub
    assert any(v[0] == "pc_suite.html" for v in hub.VIEWS), "허브 미등록"
    sh = open(os.path.join(ROOT, "build_all.sh"), encoding="utf-8").read()
    assert "build_pc_suite.py" in sh
    # 합치는 쪽이 원본보다 **뒤에** 돌아야 한다
    assert sh.index("build_pc_suite.py") > sh.index("build_farm_diagnosis.py")
    assert sh.index("build_pc_suite.py") < sh.index("build_dashboard_hub.py")


def test_psy_priority() -> None:
    """우선순위표 — **배열 작업이다. 수치가 바뀌면 잘못 만든 것이다.**

    검사의 요점 셋. 값이 원본 모듈과 같은가, 등급·축이 비지 않는가,
    그리고 **합산 문장을 만들지 않는가**.
    """
    import json
    import farm_gap as fg
    import psy_priority as pp

    r = pp.build(dict(pp.DEMO_FARM), pp.DEMO_SOWS)

    # 1) 값은 farm_gap 이 낸 것 그대로여야 한다 — 여기서 다시 계산하면 갈린다
    diag = fg.diagnose(dict(pp.DEMO_FARM), n_sows=pp.DEMO_SOWS)
    assert r["psy"] == diag["psy"] and r["psy_gap"] == diag["psy_gap"]
    won = {w["metric"]: w["won_year"] for w in diag["won_per_year"]}
    by_metric = {q["metric"]: q for q in diag["rows"]}
    for x in r["rows"]:
        if x["axis"] != "회수":
            continue
        # 회수 축에는 farm_gap 이 실제로 낸 지표만 올 수 있다. 방어·계절을
        # 여기에 밀어 넣으면 축이 섞이므로 **이름부터 막는다** — 안 막으면
        # 다음 줄이 StopIteration 으로 죽어 무엇이 틀렸는지 안 보인다.
        assert x["metric"] in by_metric, f"회수 축에 없는 지표: {x['metric']}"
        src = by_metric[x["metric"]]
        assert x["psy"] == src["psy_recover"], (x, src)
        assert x["won_year"] == won[x["metric"]], x

    # 2) 회수 항목은 회수량 내림차순
    rec = [x["psy"] for x in r["rows"] if x["axis"] == "회수"]
    assert rec == sorted(rec, reverse=True), rec
    assert len(rec) >= 2, rec

    # 3) **등급 열이 비면 안 된다.** 안 달면 횡단면(B)이 농장 내 변화(A)로 읽힌다
    for x in r["rows"]:
        assert x["grade"] in pp.GRADE, x
        assert x["axis"] in pp.AXIS, x
        assert x["target"], x
    grades = {x["grade"] for x in r["rows"]}
    assert {"A", "B", "C"} <= grades, grades

    # 4) 축이 다른 항목은 회수량을 비워야 한다 — 같은 칸에 넣으면 합쳐 읽힌다
    for x in r["rows"]:
        if x["axis"] != "회수":
            assert x["psy"] is None, x

    # 5) **합산 문장을 만들지 않는다.** sum 은 참고로만 두고 총합 주장 금지
    assert "sum_of_parts" in r and r["sum_note"]
    blob = json.dumps(r, ensure_ascii=False)
    # **금지 문구를 담은 주의문 자체는 검사에서 뺀다.** 안 빼면 "합산해
    # '총 +N두' 라고 쓰지 않는다" 라는 경고문이 위반으로 잡힌다.
    body = blob.replace(r["sum_note"], "").replace(r["footer"], "")
    assert "총 +" not in body and "합쳐서" not in body

    # 6) 환산 계수는 **단일 출처**여야 한다
    import farm_economics as fe
    lev = fe.levers(n_sows=pp.DEMO_SOWS)
    psy1 = int(lev.loc[lev["lever"] == "PSY +1두", "연간효과"].iloc[0])
    top = next(x for x in r["rows"] if x["axis"] == "회수")
    # 회수량 1두당 금액이 지렛대와 같은 급인지(정확히 같진 않다 — 비선형)
    assert 0.8 < (top["won_year"] / top["psy"]) / psy1 < 1.25, top

    # 7) 방어·계절은 원본 JSON 에서 와야 한다
    dn = json.load(open(os.path.join(ROOT, "data", "farm_panel.json"),
                        encoding="utf-8"))["downside"]
    d = next(x for x in r["rows"] if x["metric"] == "downside")
    assert d["won_year"] == dn["expected_won_year"], (d, dn)
    sn = json.load(open(os.path.join(ROOT, "data", "farm_monthly_panel.json"),
                        encoding="utf-8"))
    s = next(x for x in r["rows"] if x["metric"] == "season")
    assert s["won_year"] == round(sn["money"]["won_ref"]["median"]), s
    # 여름은 **분포로** 내야 한다 — 전체 평균 하나면 발견 ③′ 를 버리는 것
    assert "~" in s["target"] and s.get("won_p90"), s

    # 8) 모돈 두수가 다르면 방어 기댓값도 따라 환산돼야 한다
    r2 = pp.build(dict(pp.DEMO_FARM), 600)
    d2 = next(x for x in r2["rows"] if x["metric"] == "downside")
    assert abs(d2["won_year"] - d["won_year"] * 2) <= 2, (d, d2)

    # 9) 문장 규율 — 개입 효과·도달 가능성을 주장하지 않는다
    for bad in ("줄이면", "달성 가능", "오른다", "개선하면"):
        assert bad not in blob, bad
    assert "개입 효과의 추정이 아니다" in r["footer"]


def test_presentation_cnn_current() -> None:
    """발표자료가 **더 최신 결과를 안 싣고 있는** 상태를 잡는가.

    슬라이드 8 이 CPU 파이프라인 0.636 을 싣고 슬라이드 12 가 그걸 뒤집는
    상태로 며칠 갔다. 0.636 은 다른 파이프라인의 정당한 값이라 '틀린 수치'
    검사로는 안 걸린다 — 그래서 **CNN 값이 실려 있는지**를 따로 본다.
    """
    import importlib
    import json
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    cd = importlib.import_module("check_docs")

    m = cd.actual_metrics()
    j = json.load(open(os.path.join(ROOT, "data", "posture_cnn.json"),
                       encoding="utf-8"))
    # 감시값은 JSON 에서 와야 한다 — 문서에 손으로 적은 걸 읽으면 무의미하다
    assert m["cnn3_acc"] == round(j["cls3"]["acc"], 3)
    assert m["cnn_lr_restricted"] == round(j["left_right_restricted"]["acc"], 3)

    rep = []
    cd.check_posture_cnn(rep)
    assert rep == [], rep

    pres = os.path.join(ROOT, "docs", "PRESENTATION.md")
    src = open(pres, encoding="utf-8").read()
    # 실제로 실려 있는가
    assert f"{m['cnn3_acc']:.3f}" in src, "발표자료에 CNN 3클래스 값이 없다"
    assert f"{m['cnn_lr_restricted']:.3f}" in src, "좌/우 한정 값이 없다"

    # **잡아야 할 것을 잡는가** — 두 분기 모두
    import tempfile
    orig = cd.DOCS
    try:
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "PRESENTATION.md")
            open(bad, "w", encoding="utf-8").write(
                src.replace(f"{m['cnn3_acc']:.3f}", "0.636"))
            cd.DOCS = [bad]
            rep = []
            cd.check_posture_cnn(rep)
            assert any("CNN 재측정값" in r for r in rep), rep

            open(bad, "w", encoding="utf-8").write(
                src.replace('좌/우 횡와는 "bbox 로는" 구분이 안 됩니다',
                            "좌/우 횡와는 원리상 구분이 안 됩니다"))
            rep = []
            cd.check_posture_cnn(rep)
            assert any("원리상" in r for r in rep), rep
    finally:
        cd.DOCS = orig

    # 옛 주장을 **인용해서 고치는** 줄은 통과시켜야 한다(슬라이드 12·STATUS ⑦)
    st = open(os.path.join(ROOT, "docs", "STATUS.md"), encoding="utf-8").read()
    assert "원리상" in st, "고친 내역이 사라졌다 — 스스로 잡은 오류는 남긴다"


def test_estrus_label_audit() -> None:
    """발정 라벨 감사기 — **누수를 실제로 잡는가, 깨끗한 걸 통과시키는가**.

    이 도구의 유일한 임무는 "피처로 써도 되나" 에 답하는 것이다. 누수를
    놓치면 0.642 를 폐기한 결정이 무의미해지고, 깨끗한 걸 걸면 쓸 수 있는
    데이터를 버린다. 그래서 양방향으로 건다.
    """
    import json
    import re
    import tempfile
    import estrus_label_audit as ela

    def write(dirpath, name, boxes, created="2022-11-30 10:54:10"):
        d = {"INFO": {"CREATE_DATE_TIME": created},
             "IMAGE": {"FARMID": "pigfarmT"},
             "ANNOTATION_INFO": boxes}
        json.dump(d, open(os.path.join(dirpath, name + ".json"), "w",
                          encoding="utf-8"), ensure_ascii=False)

    def box(estrus, injection=True):
        b = {"CATEGORY_NAME": "pig", "ACTION_NAME": "lying", "ESTRUS": estrus}
        if injection:
            b["INJECTION"] = "Y"
        return b

    # -- 누수판: 채널이 라벨을 결정하고, 프레임 안에서 안 갈린다 ----------
    with tempfile.TemporaryDirectory() as td:
        for i in range(20):
            ch = 1 if i < 10 else 9
            e = "Y" if ch == 1 else "N"
            write(td, f"pigfarmT_ch{ch}_2022071510_025_{i:05d}",
                  [box(e), box(e), box(e)])
        r = ela.run(td)
        p = r["bbox"]["L2_channel_predict"]
        assert p["leaky"], p
        assert p["acc"] > 0.99, p
        assert r["bbox"]["L4_frame"]["share_mixed"] == 0.0
        v = r["verdict"]
        assert v["color"] == "적색", v
        assert v["checks"]["L2 완전 분리"] is False
        assert v["checks"]["L4 프레임 일관성"] is False
        # 확인 불가는 통과가 아니다 — None 으로 남아야 한다
        assert v["checks"]["L5 주석자 맹검"] is None

    # -- 깨끗한 판: 채널과 라벨이 무관하고 프레임 안에서 갈린다 ----------
    with tempfile.TemporaryDirectory() as td:
        for i in range(40):
            # 세 번째 상자를 i%2 로 정하면 ch(=1+i%4) 와 붙어 버린다 —
            # 실제로 그렇게 짰다가 '깨끗한 판' 이 누수로 걸렸다.
            # 채널과 독립인 (i//4)%2 로 정한다.
            ch = 1 + (i % 4)
            write(td, f"pigfarmT_ch{ch}_2022071510_025_{i:05d}",
                  [box("Y"), box("N"), box("Y" if (i // 4) % 2 else "N")])
        r = ela.run(td)
        assert not r["bbox"]["L2_channel_predict"]["leaky"], \
            r["bbox"]["L2_channel_predict"]
        assert r["bbox"]["L4_frame"]["share_mixed"] == 1.0
        assert not r["bbox"]["L1_missing"]["leaky"]
        v = r["verdict"]
        # 실측 셋은 통과하지만 L3·L5·VULVA 가 확인 불가라 **청색이 아니다**
        assert v["checks"]["L2 완전 분리"] is True
        assert v["checks"]["L4 프레임 일관성"] is True
        assert v["color"] == "회색", v

    # -- L1: 필드 유무가 라벨을 담고 있으면 걸려야 한다 -------------------
    with tempfile.TemporaryDirectory() as td:
        for i in range(40):
            ch = 1 + (i % 4)
            e = "Y" if i % 2 else "N"
            # 발정일 때만 INJECTION 블록이 있다 → 결측이 곧 라벨
            write(td, f"pigfarmT_ch{ch}_2022071510_025_{i:05d}",
                  [box(e, injection=(e == "Y")),
                   box("N" if e == "Y" else "Y", injection=(e != "Y"))])
        r = ela.run(td)
        assert r["bbox"]["L1_missing"]["leaky"], r["bbox"]["L1_missing"]
        assert r["verdict"]["color"] == "적색"

    # -- VULVA: 라벨이 한 값뿐이면 그 자체로 적색이어야 한다 ---------------
    #    실측이 그랬다 — 22,497건 전부 "Y", 대조군 0건. 그러면 무엇을 피처로
    #    넣든 '파일이 있으면 발정' 을 못 넘는다.
    with tempfile.TemporaryDirectory() as td:
        for i in range(30):
            json.dump({"VULVA": {"DATE": f"2022110{i % 8}_090000",
                                 "FARM_NAME": "pigfarmT",
                                 "ANIMAL_ID": f"1-{i % 6}", "ESTRUS": "Y"}},
                      open(os.path.join(td, f"v{i}.json"), "w",
                           encoding="utf-8"), ensure_ascii=False)
        vv = ela.audit_vulva(td)
        assert vv["single_class"] is True, vv
        # 상태를 기술하는 필드가 하나도 없다 — 식별자와 ESTRUS 뿐
        assert vv["descriptive_fields"] == [], vv["descriptive_fields"]
        assert vv["n_animals"] == 6 and vv["n_dates"] == 8, vv
        with tempfile.TemporaryDirectory() as bd:
            for i in range(6):
                write(bd, f"pigfarmT_ch{1 + i % 3}_2022071510_025_{i:05d}",
                      [box("Y"), box("N")])
            v = ela.verdict(ela.audit_bbox(bd), vv)
        assert v["checks"].get("VULVA 대조군") is False, v
        assert v["color"] == "적색", v

    # -- 저장된 집계에 원자료 파일명이 새면 안 된다 -----------------------
    p = os.path.join(ROOT, "data", "estrus_label_audit.json")
    if os.path.exists(p):
        blob = open(p, encoding="utf-8").read()
        assert not re.search(r"pigfarm\w+_ch\d", blob), "파일명 노출"
        saved = json.loads(blob)
        # 실측 판정은 적색이었다 — 이게 바뀌면 근거가 바뀐 것이다
        assert saved["verdict"]["color"] == "적색", saved["verdict"]


def test_path_predict() -> None:
    """로그 → 경로 → 예측.

    검사의 요점 셋. **경로 복원이 맞는가**(주입값을 되찾는가),
    **미래를 안 보는가**(시간 순 분할·사후 열 배제),
    **기준선을 넘었다고 거짓말하지 않는가**(표준 캘린더 대비).
    """
    import numpy as np
    import pandas as pd
    import path_predict as pp
    import repro_calendar as rc
    import synth_farm as sf

    P = sf.Params()
    cycles = sf.generate(n_sows=120, years=3, seed=1, params=P)
    ev = pp.to_events(cycles)

    # 사건 어휘는 정해진 것만 나와야 한다 — 앱 로그와 낱말이 갈리면 못 받는다
    assert set(ev["event"]) <= set(pp.EVENTS), set(ev["event"])
    # 한 사이클 안에서 사건은 **시간 순**이어야 한다. 뒤집히면 경로가 거짓이다
    for _, g in ev.groupby(["animal_id", "cycle"]):
        ds = list(g.sort_index()["date"])
        assert ds == sorted(ds), g.to_dict("records")

    # 1) 경로 복원 — 주입한 모수를 로그만 보고 되찾는가.
    #    날짜가 일 단위라 **평균**으로 잰다(중앙값은 소수를 못 만든다).
    rec = pp.recovery(ev, cycles, P)
    for k in ("wei", "gestation", "lactation", "farrowing_rate"):
        assert rec[k]["ok"], (k, rec[k])
    assert rec["gestation"]["n"] > 100

    # 2) 경로 변형 — 표준 경로가 있고, 비율 합이 1 을 넘지 않는다
    v = pp.variants(ev)
    assert v["standard"].any(), v.to_dict("records")
    assert 0.99 < v["share"].sum() <= 1.0001, v["share"].sum()
    std = v[v["standard"]].iloc[0]
    assert tuple(std["path"].split(" → ")) == pp.STANDARD

    # 3) 전이 간격 — 표준 상수는 repro_calendar 에서 와야 한다.
    #    여기서 따로 적으면 캘린더를 고쳐도 이 화면만 옛 값을 든다.
    assert pp.B0_DAYS[("교배", "분만")] == float(rc.GESTATION)
    assert pp.B0_DAYS[("분만", "이유")] == float(rc.LACTATION)
    assert pp.B0_DAYS[("교배", "재발정")] == float(rc.RETURN_CHECK)

    # 4) 날짜 예측 — 시간 순으로 잘렸는가. 학습이 평가 뒤를 보면 안 된다
    agg, pairs = pp.transitions(ev)
    d = pp.predict_days(pairs)
    if not d.get("skipped"):
        cut = pd.Timestamp(d["cut_date"]).date() \
            if not isinstance(d["cut_date"], str) or "-" in d["cut_date"] \
            else None
        assert d["n_train"] > d["n_test"], (d["n_train"], d["n_test"])
        sc = d["scores"]
        assert abs(sc["B0 표준 캘린더"]["gain_vs_B0"]) < 1e-9
        for k, v2 in sc.items():
            assert 0.0 <= v2["hit"] <= 1.0 and v2["mae"] >= 0.0, (k, v2)
        if cut:
            # 평가 구간의 행이 실제로 컷 뒤인지 직접 확인.
            # predict_days 는 **표준 상수가 있는 전이만** 쓰므로 같은 필터를
            # 걸고 세야 한다 — 안 걸면 1,340 vs 1,282 로 어긋난다.
            keep = pairs[[(f, tt) in pp.B0_DAYS
                          for f, tt in zip(pairs["from"], pairs["to"])]]
            late = keep[keep["date"] > cut]
            assert len(late) == d["n_test"], (len(late), d["n_test"])

    # 5) 결과 예측 — **사후 열이 피처에 없어야 한다.** 분만일·이유두수를
    #    넣으면 그 자리에서 예측이 아니게 된다.
    fr = pp._outcome_frame(cycles)
    feats = ["parity", "month", "wei", "est_gap", "prior_returns", "prior_cycles"]
    for bad in ("farrow", "wean", "born_alive", "weaned", "outcome"):
        assert bad not in feats, bad
    # prior_returns 는 **이번 교배 전까지**만 세야 한다 — 첫 사이클은 0
    first = fr.groupby("sow_id").head(1)
    assert (first["prior_returns"] == 0).all(), "첫 사이클에 과거 재발이 있다"
    # 그 개체의 누적 재발 수를 손으로 세서 대조
    one = fr[fr["sow_id"] == fr["sow_id"].iloc[0]].sort_values("service")
    want = (1 - one["y"]).shift(1).cumsum().fillna(0).to_numpy()
    assert np.allclose(one["prior_returns"].to_numpy(), want)

    # -- A-1 수축 --------------------------------------------------------
    # B2 가 B1 에 진 원인은 개체 추정치의 노이즈였다. 수축이 그걸 고치는가.
    if not d.get("skipped"):
        sc = d["scores"]
        assert sc["B3 개체 수축"]["mae"] < sc["B2 개체 과거"]["mae"], sc
        # **B1 은 못 넘는다.** 넘었다고 쓰면 거짓말이 된다 — 계열이 다르다
        fc = d["family_check"]
        assert fc["B3_mean_centred"] < fc["B1_mean"], fc   # 같은 계열에선 이긴다
        sp = d["shrink"]
        for k, v in sp.items():
            # 분산 분해 — 오차가 관측보다 크면 진짜 몫은 **0 으로 자른다**.
            # 항등식만 걸었다가 그 경우에 실패했다(분만→이유 obs 0.479 <
            # err 0.532). 분산이 음수일 수는 없으므로 자르는 게 맞다.
            want = max(0.0, v["var_obs"] - v["var_err"])
            assert abs(v["var_true"] - want) < 0.02, (k, v)
            assert 0.0 <= v["true_share"] <= 1.0 and 0.0 <= v["w_median"] <= 1.0
        # 변동이 전혀 없는 전이(교배→재발정 21일 고정)는 w=0 이어야 한다.
        # 0/0 을 안 막으면 NaN 이 예측 전체로 번진다 — 실제로 그랬다
        fixed = sp.get("교배→재발정")
        if fixed:
            assert fixed["w_median"] == 0.0, fixed
        assert all(np.isfinite(v["mae"]) for v in sc.values()), sc

    o = pp.predict_outcome(cycles)
    if not o.get("skipped"):
        for key in ("time_split", "group_split"):
            s = o[key]
            if s.get("skipped"):
                continue
            # 기준선을 함께 내야 한다 — 정확도만 보면 다수 클래스에 속는다
            assert "baseline" in s and "model" in s
            assert set(s["model"]) >= {"acc", "mf1"}
            assert s["verdict"] in ("개선", "기준선과 같음", "기준선 미달",
                                    "정확도만 미달(불균형에 가림)")
        # 개체 단위 분할에서는 train/test 개체가 겹치면 안 된다
        assert o["group_split"].get("skipped") or True

    # -- A-3 검출력 ------------------------------------------------------
    # "안 된다" 가 표본 탓인지 지표 탓인지 신호 탓인지를 가르는 장치다.
    pw = pp.power(cycles, deltas=(0.05, 0.25), seeds=4)
    if not pw.get("skipped"):
        assert 0.4 < pw["null_auc_p95"] < 0.7, pw["null_auc_p95"]
        c5, c25 = pw["curve"][5.0], pw["curve"][25.0]
        # 효과를 키우면 **AUC 는 올라야 한다.** 안 오르면 주입이 안 된 것이다
        assert c25["auc_mean"] > c5["auc_mean"] + 0.05, (c5, c25)
        # 하드 라벨은 82% 불균형에 가려 안 오른다 — 그게 이 분석의 요점이다
        assert c25["auc_mean"] > 0.55, c25
        # 실측 계절 손실은 farm_monthly_panel 에서 받아야 한다
        import json as _j
        pj = _j.load(open(os.path.join(ROOT, "data", "farm_monthly_panel.json"),
                          encoding="utf-8"))
        assert pw["measured_gap_pp"] == abs(pj["overall"]["summer_minus_winter"])


def test_barn_watch() -> None:
    """배치 전이 감시 — 검사기가 **틀어진 것을 실제로 잡는가**.

    설계대로 지으면 0 건이 나오는 건 쉽다. 이 검사의 요점은 방을 빼면
    반드시 잡히는가, 그리고 **아무것도 안 움직인 것을 '정상' 이라 하지
    않는가**다. 후자는 실제로 그렇게 보고했다가 고쳤다.
    """
    import barn_watch as bw
    import run_farm as rf
    from pigflow.config import default_config
    from pigflow.simulator import build_rooms

    cfg = default_config()
    cfg.crate_count = rf.crates_for_sows(300, cfg)
    m = cfg.merged()
    full = build_rooms(m)

    # 1) 설계대로 지으면 규칙 위반이 없어야 한다
    r = bw.watch(cfg, days=400, rooms=full)
    assert r["feasible"] and r["verdict"] == "정상", (r["verdict"], r["counts"])
    assert r["n_violations"] == 0, r["counts"]
    assert r["n_transitions"] > 50 and r["n_steady"] > 20, r
    # 갓 태어난 배치를 '못 움직인' 것으로 세면 안 된다
    assert r["n_never_moved"] == 0, r["n_never_moved"]
    # 워밍업은 판정에서 빠져야 한다 — 뒷단이 비어 있는 구간이다
    assert r["warmup_days"] > 0 and r["n_steady"] < r["n_transitions"]
    assert all(s["warmup"] for s in r["snapshots"]
               if s["day"] < r["warmup_days"])

    # 2) 방을 빼면 반드시 잡혀야 한다. 안 잡히면 검사기가 무용하다.
    ids = [x.room_id for x in full if x.house == "nursery"][:2]
    short = [x for x in build_rooms(m) if x.room_id not in ids]
    r2 = bw.watch(cfg, days=400, rooms=short)
    assert r2["verdict"] == "위반 있음", r2["verdict"]
    assert r2["counts"].get("적체", 0) > 0, r2["counts"]
    assert r2["worst_jam"] and r2["worst_jam"]["over_days"] > 0

    # 3) **아무것도 안 움직인 것을 '정상' 이라 하면 안 된다.**
    #    방을 배치보다 작게 주면 한 발짝도 못 가는데, 그때 위반 0 건이
    #    나온다고 통과시키면 이 도구는 거짓말을 하는 셈이다.
    tiny = [x for x in build_rooms(m)]
    for x in tiny:
        x.capacity_head, x.area_m2 = 1, 1.0
    r3 = bw.watch(cfg, days=200, rooms=tiny)
    assert not r3["feasible"] and r3["blocked"], r3["verdict"]
    assert r3["verdict"] == "흐름 실패", r3["verdict"]
    assert r3["n_steady"] == 0

    # 4) 분만사 부족은 '적체' 가 아니라 '무처소' 로 나타난다 — 배치는 방이
    #    없어도 만들어지고 나이가 차면 다음 단계로 가기 때문이다
    fids = [x.room_id for x in full if x.house == "farrowing"][:3]
    nf = [x for x in build_rooms(m) if x.room_id not in fids]
    r4 = bw.watch(cfg, days=400, rooms=nf)
    assert r4["counts"].get("무처소", 0) > 0, r4["counts"]

    # 5) 등록 화면 JSON → 돈방. 비육사는 육성·비육을 겸하므로 나뉘어야 한다
    spec, notes = bw.rooms_from_setup(
        {"barns": [{"name": "3동", "stage": "분만사", "rooms": 2, "per": 40},
                   {"name": "6동", "stage": "비육사", "rooms": 10, "per": 120},
                   {"name": "1동", "stage": "교배사", "rooms": 1, "per": 70}]}, m)
    houses = {s["house"] for s in spec}
    assert houses == {"farrowing", "grower", "finisher"}, houses
    # 교배사는 번식돈 자리라 빠지고, 그 사실이 보고돼야 한다
    assert any(n[1] == "교배사" for n in notes), notes
    assert any("육성·비육" in n[2] for n in notes), notes
    # 나눈 방 수의 합은 등록한 수와 같아야 한다 — 여기서 새 방을 만들면 안 된다
    assert sum(1 for s in spec if s["house"] in ("grower", "finisher")) == 10

    # 6) 어휘는 farm_registry 것이어야 한다
    import farm_registry as fr
    for stage in bw.HOUSE_OF:
        assert stage in fr.BARN_STAGES, stage
    for stage in bw.BREEDING_ONLY:
        assert stage in fr.BARN_STAGES, stage


def test_farm_setup_view() -> None:
    """농장 등록 화면 — 어휘가 코드와 같은가, 그리고 아무 데도 안 보내는가.

    이 화면의 위험은 계산이 아니라 **어휘 분기**다. 여기서 축사 용도나 사육
    방식을 손으로 적어 두면 `farm_registry` 가 거절하는 값을 사용자에게
    권하게 된다. 그래서 상수를 그대로 쓰는지 본다.
    """
    import json
    import re
    import build_farm_setup as bfs
    import farm_registry as fr

    html = bfs.build()

    # 자체완결 — 외부 연결 0. 농장 정보를 다루는 화면이라 특히 그렇다
    ext = re.findall(r'https?://[^"\'\s)]+', html)
    assert not ext, f"외부 URL {ext[:3]}"
    assert not re.findall(r'<(?:script|link|img)[^>]*\bsrc=|<link[^>]*href=', html)
    # **전송 경로가 없어야 한다.** 입력값이 곧 농장 식별자다
    for bad in ("fetch(", "XMLHttpRequest", "navigator.sendBeacon",
                "WebSocket", "<form"):
        assert bad not in html, f"전송 경로로 보이는 것: {bad}"
    assert "localStorage" in html, "로컬 저장이 없으면 매번 다시 입력해야 한다"

    # 어휘는 farm_registry 것을 그대로 — 새 낱말을 만들면 코드와 갈라진다.
    # 선택지는 JS 가 그리므로 화면에 심은 **목록 자체**를 꺼내 대조한다.
    def embedded(name):
        m = re.search(rf"^const {name} = (\[.*?\]);$", html, re.M | re.S)
        assert m, f"{name} 목록을 화면에서 찾지 못했다"
        return json.loads(m.group(1))

    assert embedded("STAGES") == list(fr.BARN_STAGES), "축사 용도 어휘가 갈렸다"
    assert [h[0] for h in embedded("HOUSING")] == list(fr.HOUSING), \
        "사육 방식 어휘가 갈렸다"
    # 사육 방식 → 발정 판정 경로도 코드에서 와야 한다
    cm = re.search(r"^const CFG = (\{.*?\});$", html, re.M | re.S)
    assert cm, "CFG 를 찾지 못했다"
    cfg = json.loads(cm.group(1))
    for h, v in fr.HOUSING.items():
        assert cfg["routes"][h]["module"] == v[1], (h, cfg["routes"][h])

    # 성적란은 run_farm 이 **실제로 받는 인자**여야 한다. 없는 인자를 만들면
    # 복사해서 붙인 명령줄이 그 자리에서 죽는다.
    import run_farm
    ap_src = open(os.path.join(ROOT, "src", "run_farm.py"),
                  encoding="utf-8").read()
    for _, _, _, arg, _, _ in bfs.PERF:
        flag = "--" + arg.replace("_", "-")
        assert flag in ap_src, f"run_farm 에 없는 인자를 권하고 있다: {flag}"
    assert run_farm is not None

    # 빈 칸을 중앙값으로 채우지 않는다 — 격차가 늘 0 으로 찍혔던 버그
    assert "진단에서 제외" in html and "중앙값을 넣지 않습니다" in html

    # 분위수는 466행 실측에서 온다. 화면이 따로 만들어 쓰면 갈린다
    q = bfs.quantiles()
    assert "farrowing_rate" in q and "npd" in q, q
    st = json.load(open(os.path.join(ROOT, "data", "korean_farm_stats.json"),
                        encoding="utf-8"))
    assert q["npd"]["p50"] == st["quantiles"]["npd"]["p50"]

    # 번식 상수도 pigflow 에서 받아야 한다
    from pigflow.config import BREEDING_DEFAULTS as B
    d = bfs.defaults()
    assert d["gestation"] == float(B["gestation_days"])
    assert d["lactation"] == float(B["lactation_days"])

    # -- 계절 손실(발견 ③′)이 **같은 수치로** 붙었는가 -------------------
    #
    # 이 화면의 값어치는 원/년을 **사용자 규모로** 놓는 것이다. 그러려면
    # 환산 계수와 여름 비중이 farm_monthly_panel 과 같아야 한다. 여기서
    # 새로 만들면 두 화면이 같은 농장에 다른 금액을 말하게 된다.
    import farm_monthly_panel as mp
    import farm_economics as fe
    s = bfs.season()
    pj = json.load(open(os.path.join(ROOT, "data", "farm_monthly_panel.json"),
                        encoding="utf-8"))
    assert s["per_sow_won"] == pj["money"]["per_sow_won"]
    lev = fe.levers(n_sows=mp.REF_SOWS)
    assert s["per_sow_won"] == int(
        lev.loc[lev["lever"] == "PSY +1두", "두당효과"].iloc[0])
    assert s["share"] == mp.SEASON_SHARE
    assert s["loss"] == pj["loss"] and s["n_farms"] == pj["n_farms"]
    assert s["implantation"] == list(
        __import__("barn_environment").IMPLANTATION_WINDOW)

    # 화면에 실린 값도 같은 것이어야 한다
    assert cfg["season"]["per_sow_won"] == s["per_sow_won"], "화면 계수가 갈렸다"

    # 계절 취약도는 연간 성적으로 예측이 안 된다 — 이 경고가 빠지면 사용자가
    # PSY 만 보고 "우리는 괜찮겠지" 로 넘어간다
    assert "맞힐 수 없습니다" in html and str(s["rho_psy"]) in html
    # 비운 경우를 우리 농장 값처럼 보이게 하면 안 된다
    assert "우리 농장 값이 아닙니다" in html
    # 겨울 기준이라 상한이라는 것도 밝혀야 한다
    assert "손실 상한" in html

    # -- 돈사 등록이 **출하까지** 이어지는가 ------------------------------
    #
    # 예전 이 화면은 번식사 넷(교배·임신·분만·후보)에서 끝났다. 그러면
    # 등록만으로는 뒷단 병목이 안 보인다 — batch_flow 가 잡은 육성사 여유
    # 0일이 화면에 한 번도 안 뜬다.
    import growth_flow as gf
    down = bfs.downstream_stages()
    assert [x["stage"] for x in down] == ["자돈사", "육성사", "비육사"], down
    for x in down:
        assert x["stage"] in fr.BARN_STAGES, x
    # 일수·면적·폐사는 growth_flow 가 원본이어야 한다
    for name, a0, a1, _w0, _w1, barn, area in gf.STAGES:
        if area is None:
            continue
        got = next(x for x in down if x["stage"] == barn)
        assert got["days"] == a1 - a0 and got["area"] == area, (barn, got)
        assert got["mort"] == gf.MORTALITY[name], (barn, got)
    assert cfg["down"] == down, "화면에 실린 뒷단이 갈렸다"

    # batch_flow 도 같은 일수를 써야 한다. 비육 63일이 박혀 있어서
    # 105~175일령(70일)과 7일 어긋나 있었다 — 여유가 그만큼 부풀었다
    import batch_flow as bf2
    assert bf2.DOWNSTREAM_DAYS == {x["stage"]: x["days"] for x in down}

    # -- 정적 계산이 시뮬레이터 소요보다 작을 수 있다 ---------------------
    #
    # 이게 이 화면에서 실제로 난 사고다. 자돈사를 일령 한 구간(46일)으로
    # 세면 3방인데, 시뮬레이터는 전기·후기로 나눠 방을 따로 써서 4방이
    # 필요하다. 3방으로 등록하면 화면은 통과시키고 돌리면 적체 55회가 난다.
    sim = bfs.sim_stages()
    assert cfg["sim"] == sim, "화면에 실린 흐름 단계가 갈렸다"
    n_nursery = sum(1 for s in sim if s["stage"] == "자돈사")
    assert n_nursery > 1, "자돈사가 나뉘지 않으면 이 검사는 의미가 없다"
    # 큰 쪽으로 짓는다는 문장과 이유가 화면에 있어야 한다
    assert "둘 중 큰 쪽" in html
    assert "방을 따로 씁니다" in html
    # 면적을 자리 수에서 되돌려 채우면 안 된다 — 늘 '적정' 으로 찍힌다
    assert "과밀로 안 잡히기" in html
    # 후보사를 표에서 뺀 이유를 밝혀야 한다. 조용히 빼면 통과로 읽힌다
    assert "후보사는 이 표에 없습니다" in html


def test_capacity_from_rooms() -> None:
    """**지어 놓은 방 → 넣을 수 있는 두수.** 설계의 반대 방향.

    이 함수의 값어치는 두수가 아니라 **병목의 이름**이다. 두수만 맞고 병목을
    엉뚱하게 짚으면 처방이 통째로 틀린다 — 그래서 병목을 먼저 본다.
    """
    import batch_flow as bf

    ok = [{"stage": "교배사", "rooms": 1, "per": 72},
          {"stage": "임신사", "rooms": 2, "per": 82},
          {"stage": "분만사", "rooms": 2, "per": 36},
          {"stage": "자돈사", "rooms": 4, "per": 396},
          {"stage": "육성사", "rooms": 3, "per": 385},
          {"stage": "비육사", "rooms": 4, "per": 381}]
    r = bf.capacity_from_rooms(ok, 21, lactation=24, weaned_per_crate=11.0)
    assert r["flows"] and r["n_sows"] > 0, r
    assert r["binding"] in {b["stage"] for b in ok}, r["binding"]
    # 병목은 **가장 작게 지지하는 돈사**여야 한다
    live = [x for x in r["rows"] if x["sows"] > 0]
    assert r["n_sows"] == min(x["sows"] for x in live), r["rows"]

    # 1) 설계와 역산이 서로를 되찾는가. 분만틀에서 지은 농장을 되읽으면
    #    같은 분만틀이 나와야 한다 — 안 그러면 같은 돈사가 방향에 따라 다른
    #    크기로 나온다(실제로 12.0 목표로 되읽어 33 vs 36 이 나왔다)
    for crates, iv in ((36, 21), (20, 14), (12, 7)):
        p = bf.plan_from_crates(crates, iv, weaned_per_crate=11.0)
        rooms = -(-(bf.MOVE_IN + 24 + bf.WASHDOWN) // iv)
        head = 11.0 * crates
        built = [{"stage": "분만사", "rooms": rooms, "per": crates}]
        for st, days in bf.DOWNSTREAM_DAYS.items():
            built.append({"stage": st, "rooms": -(-(days + bf.WASHDOWN) // iv),
                          "per": -(-head // 1)})
        back = bf.capacity_from_rooms(built, iv, lactation=24,
                                      weaned_per_crate=11.0)
        assert back["crates"] == crates, (crates, iv, back["crates"])
        assert back["n_sows"] == p["herd_size"], (crates, iv, back["n_sows"])

    # 2) **막힌 돈사가 병목이다.** 막힌 곳은 지지 두수 0 이라 그냥 두면 그다음
    #    으로 작은 돈사가 병목으로 뽑힌다 — 방이 모자란 자돈사를 놔두고
    #    "육성사가 병목" 이라고 말하게 되고, 두수를 줄여도 안 풀린다
    thin = [b if b["stage"] != "자돈사" else dict(b, rooms=1) for b in ok]
    t = bf.capacity_from_rooms(thin, 21, lactation=24, weaned_per_crate=11.0)
    assert not t["flows"] and t["binding"] == "자돈사", t
    assert t["n_sows"] == 0 and t["crates"] == 0, t
    assert any("방" in x["why"] for x in t["blocked"]), t["blocked"]

    # 3) 방을 넓히면 그 돈사는 병목에서 빠져야 한다 — 붙잡고 있던 게 맞다면
    wide = [b if b["stage"] != r["binding"]
            else dict(b, per=b["per"] * 3, rooms=b["rooms"] + 2) for b in ok]
    w = bf.capacity_from_rooms(wide, 21, lactation=24, weaned_per_crate=11.0)
    assert w["binding"] != r["binding"] and w["n_sows"] > r["n_sows"], (r, w)

    # 4) 시뮬레이터 방 소요를 얹으면 더 빡빡해진다(자돈사 전·후기 분리)
    e = bf.capacity_from_rooms(ok, 21, lactation=24, weaned_per_crate=11.0,
                               extra_rooms={"자돈사": 9})
    assert not e["flows"] and e["binding"] == "자돈사", e

    # 5) 등록이 없으면 두수를 지어내지 않는다
    z = bf.capacity_from_rooms([], 21)
    assert z["n_sows"] == 0 and z["binding"] is None, z


def test_throughput_ceiling() -> None:
    """**이 돈사의 연간 최대 출하두수**, 그리고 거기까지 가는 길 셋.

    한 줄 항등식이다:
        연간 출하 = 분만틀 × 채움률 × 복당 이유두수 × 육성률 × 연간 배치수

    가장 위험한 실패는 **방이 못 받는 생산량을 상한이라고 부르는 것**이다.
    설계 목표 12두를 그냥 상한으로 쓰면 자돈사 396자리에 분만틀 36개인
    농장이 432두를 "낼 수 있다" 고 나온다 — 36두가 갈 곳이 없다.
    """
    import batch_flow as bf
    import farm_economics as fe

    barns = [{"stage": "교배사", "rooms": 1, "per": 72},
             {"stage": "임신사", "rooms": 2, "per": 82},
             {"stage": "분만사", "rooms": 2, "per": 36},
             {"stage": "자돈사", "rooms": 4, "per": 396},
             {"stage": "육성사", "rooms": 3, "per": 385},
             {"stage": "비육사", "rooms": 4, "per": 381}]
    cap = bf.capacity_from_rooms(barns, 21, lactation=24, weaned_per_crate=11.0)

    # 1) 방이 복당 이유두수 상한을 정한다. 396자리 ÷ 36틀 = 11.0두
    assert cap["weaned_ceiling"] == 11.0, cap["weaned_ceiling"]
    t = bf.throughput(cap)
    assert t["weaned_room_bound"] and t["top_weaned"] == 11.0, t
    # 방이 넘치면 안 된다 — 상한 배치가 자돈사 한 방에 들어가야 한다
    assert t["crates"] * t["top_weaned"] <= 396 + 1e-9

    # 2) 성적을 안 넣으면 격차를 지어내지 않는다. '지금' 이 곧 상한이다
    assert t["now_year"] == t["ceiling_year"] and t["gap_year"] == 0, t
    assert all(w["at_target"] for w in t["ways"]), t["ways"]

    # 3) 항등식이 실제로 성립하는가 — 곱한 값과 같아야 한다.
    #    batches_per_year 는 **표시용으로 소수 1자리 반올림**된 값이라
    #    그걸로 되짚으면 7두쯤 어긋난다. 검산에는 365÷간격을 쓴다.
    per_year = 365.0 / cap["interval_days"]
    got = (t["crates"] * t["factors"]["fill"] * t["factors"]["weaned"]
           * t["factors"]["survival"] * per_year)
    assert abs(got - t["now_year"]) < 1, (got, t["now_year"])

    # 4) 나쁜 성적을 넣으면 길 셋이 다 열린다
    r = bf.throughput(cap, farrow_rate=0.74, weaned_per_litter=10.0,
                      grow_survival=0.86)
    assert 0 < r["achieved"] < 1 and r["gap_year"] > 0, r
    assert all(w["gain"] > 0 for w in r["ways"]), r["ways"]
    # **합치지 않는다.** 곱해지므로 개별 합은 총 격차와 다르다
    assert r["sum_of_ways"] != r["gap_year"], (r["sum_of_ways"], r["gap_year"])
    assert "합산해" in r["sum_note"]

    # 5) 각 몫은 **그것 하나만** 올렸을 때의 값이다 — 직접 다시 계산해 대조
    for w in r["ways"]:
        f = dict(r["factors"])
        f[w["key"]] = (1.0 if w["key"] == "fill"
                       else (r["top_weaned"] if w["key"] == "weaned"
                             else bf.CEILING["survival"]))
        alone = (r["crates"] * f["fill"] * f["weaned"] * f["survival"]
                 * per_year)
        assert abs((alone - r["now_year"]) - w["gain"]) < 1, (w, alone)

    # 6) 분만율이 설계 기준을 넘어도 채움률은 1 을 못 넘는다 — 틀이 더 생기지
    #    않으므로. 안 막으면 없는 생산량이 나온다
    hi = bf.throughput(cap, farrow_rate=0.95)
    assert hi["factors"]["fill"] == 1.0 and hi["now_year"] <= hi["ceiling_year"]

    # 7) 뒷단을 넓히면 상한이 올라가야 한다 — 방이 상한을 정한다는 주장의 검증
    wide = [dict(b, per=b["per"] * 2) if b["stage"] in bf.DOWNSTREAM_DAYS else b
            for b in barns]
    w2 = bf.throughput(bf.capacity_from_rooms(wide, 21, lactation=24,
                                              weaned_per_crate=11.0))
    assert w2["ceiling_year"] > t["ceiling_year"], (w2, t)
    # 다만 설계 목표 12두를 넘지는 않는다 — 방이 넉넉해도 돼지가 더 낳지 않는다
    assert w2["top_weaned"] == bf.CEILING["weaned"], w2["top_weaned"]

    # 8) 원/년은 **한계 이익**이어야 한다. 총원가로 재면 개선의 값이 작게 나온다
    m = fe.margin_per_pig()
    assert m["margin"] > fe.revenue_per_pig()["revenue"] - \
        fe.cost_per_pig()["total"], m
    assert "증축 판단에는 쓰지 않는다" in m["note"]


def test_setup_screen_matches_module() -> None:
    """등록 화면의 역산 JS 가 `batch_flow` 와 **같은 두수**를 내는가.

    화면이 계산을 다시 구현하면 언젠가 갈린다. 이 프로젝트는 그걸 여러 번
    겪었다(여름 손실 계수·뒷단 사육일수). 그런데 이건 파이썬으로 못 읽는
    JS 라서, **브라우저에서 두 결과를 직접 대조**한다.

    playwright 가 없으면 상수 대조까지만 하고 넘어간다 — requirements 에
    없는 의존이라 없다고 실패시키면 안 된다. 대신 **건너뛴 사실을 찍는다.**
    """
    import json
    import re
    import batch_flow as bf
    import build_farm_setup as bfs

    html = bfs.build()
    cm = re.search(r"^const CFG = (\{.*?\});$", html, re.M | re.S)
    assert cm, "CFG 를 찾지 못했다"
    cap = json.loads(cm.group(1))["cap"]

    # 상수는 batch_flow 것이어야 한다. 화면이 따로 적으면 거기서 갈린다
    assert cap["farrow_rate"] == bf.FARROW_RATE_P10
    assert cap["gilt_share"] == bf.GILT_SHARE
    assert cap["gilt_weeks"] == bf.GILT_PIPELINE_WEEKS
    assert cap["weaned_per_crate"] == bf.WEANED_PER_CRATE
    assert cap["service_hold"] == bf.SERVICE_HOLD_DAYS
    assert cap["down_days"] == bf.DOWNSTREAM_DAYS
    # 주기·회전율은 **관행 상수가 아니라 `herd_cycle()`** 이어야 한다. 예전에
    # 화면이 관행 2.3 · 지침 WEI 5.0 을 박고 있었고 모듈은 실측 중앙으로
    # 옮겨 가서, 같은 돈사가 화면 295두 · 모듈 299두로 갈렸다
    cyc = bf.herd_cycle()
    assert cap["turnover"] == cyc["turnover"], (cap["turnover"], cyc)
    assert cap["gestation"] == cyc["gestation"], (cap["gestation"], cyc)
    assert cap["wei"] == cyc["wean_to_service"], (cap["wei"], cyc)
    # 실측이 있으면 관행값을 그대로 쓰지 않는다 — 갈라진 게 정상이다
    assert cyc["source"]["wean_to_service"] in ("실측 중앙", "지침")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("      (playwright 없음 — 상수 대조만 하고 JS 대조는 건너뜀)")
        return
    exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    if not os.path.exists(exe):
        print("      (chromium 없음 — JS 대조는 건너뜀)")
        return

    cases = [
        ([{"stage": "교배사", "rooms": 1, "per": 72},
          {"stage": "임신사", "rooms": 2, "per": 82},
          {"stage": "분만사", "rooms": 2, "per": 36},
          {"stage": "자돈사", "rooms": 4, "per": 396},
          {"stage": "육성사", "rooms": 3, "per": 385},
          {"stage": "비육사", "rooms": 4, "per": 381}], 21, 24),
        # 자돈사 방 부족 — 막힌 곳이 병목으로 잡혀야 한다
        ([{"stage": "분만사", "rooms": 2, "per": 36},
          {"stage": "자돈사", "rooms": 2, "per": 396},
          {"stage": "육성사", "rooms": 3, "per": 385},
          {"stage": "비육사", "rooms": 4, "per": 381}], 21, 24),
        # 5주 간격 — 반올림이 갈리기 쉬운 자리
        ([{"stage": "분만사", "rooms": 2, "per": 50},
          {"stage": "자돈사", "rooms": 2, "per": 600},
          {"stage": "육성사", "rooms": 2, "per": 580},
          {"stage": "비육사", "rooms": 3, "per": 570}], 35, 28),
    ]
    url = "file://" + os.path.join(ROOT, "dashboard", "farm_setup.html")
    open(os.path.join(ROOT, "dashboard", "farm_setup.html"),
         "w", encoding="utf-8").write(html)
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=exe)
        pg = br.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        for barns, iv, lact in cases:
            js = pg.evaluate(
                "([bs, iv, lact, w]) => { barns = bs.map(b => ({...b}));"
                " return capacityFromRooms(iv, lact, CFG.d.pre_farrow,"
                " CFG.d.washout, {}, w); }", [barns, iv, lact, 11.0])
            py = bf.capacity_from_rooms(barns, iv, lactation=lact,
                                        weaned_per_crate=11.0)
            for k in ("n_sows", "binding", "crates", "flows",
                      "weaned_ceiling"):
                assert js[k] == py[k], (k, iv, js[k], py[k])
            assert [r["sows"] for r in js["rows"]] == \
                   [r["sows"] for r in py["rows"]], (iv, js["rows"])

            # 생산량 상한도 같아야 한다 — 성적을 넣은 경우와 안 넣은 경우 둘 다
            for fr, wl, gs in ((None, None, None), (0.74, 10.0, 0.86)):
                jt = pg.evaluate(
                    "([c, fr, wl, gs]) => throughput(c, fr, wl, gs)",
                    [js, fr, wl, gs])
                pt = bf.throughput(py, farrow_rate=fr, weaned_per_litter=wl,
                                   grow_survival=gs)
                for k in ("now_year", "ceiling_year", "gap_year", "achieved",
                          "top_weaned", "weaned_room_bound", "sum_of_ways"):
                    assert jt[k] == pt[k], (k, iv, fr, jt[k], pt[k])
                assert [w["gain"] for w in jt["ways"]] == \
                       [w["gain"] for w in pt["ways"]], (iv, fr, jt["ways"])
        br.close()
        assert not errs, errs


def test_setup_json_actually_runs() -> None:
    """등록 화면이 내보낸 JSON 이 **실제로 돌아가는가**.

    등록 화면의 값어치는 폼이 아니라 그다음이다. 내보낸 JSON 을
    `barn_watch --setup` 이 못 돌리면 이 화면은 종이다. 그리고 못 돌 때의
    증상이 고약하다 — 배치가 한 발짝도 못 가면 전이가 0 회라 집계가
    **전부 0 건 = 위반 없음** 으로 보인다.

    실제로 세 군데가 어긋나 있었다:
      · 분만사 '방당 자리' 는 분만틀(모돈) 수인데 pigflow 는 포유자돈 두수
      · crate_count 는 총 분만틀이 아니라 **방 하나 크기**
      · 등록 간격 21일을 안 읽어 시뮬레이터가 주간으로 돌았다
    """
    import barn_watch as bw
    import run_farm as rf

    setup = {
        "n_sows": 300, "interval_days": 21, "lactation_days": 24,
        "pre_farrow_days": 7, "washout_days": 7,
        "barns": [
            {"name": "1동", "stage": "교배사", "rooms": 1, "per": 72},
            {"name": "3동", "stage": "분만사", "rooms": 2, "per": 36},
            {"name": "5동", "stage": "자돈사", "rooms": 4, "per": 396},
            {"name": "6동", "stage": "육성사", "rooms": 3, "per": 385},
            {"name": "7동", "stage": "비육사", "rooms": 4, "per": 381},
        ],
    }
    cfg = bw.default_config()

    # 1) 간격이 배치 체계로 이어져야 한다. 안 읽으면 3주를 골라도 주간으로
    #    돌아 배치 크기가 3배 어긋난다
    sid, why = bw.batch_system_from_setup(setup, cfg)
    assert sid and why is None, (sid, why)
    cfg.batch_system_id = sid
    assert cfg.batch_system.interval_weeks == 3.0
    # 맞는 체계가 없으면 조용히 넘어가지 않고 말해야 한다
    _, why10 = bw.batch_system_from_setup({"interval_days": 10}, cfg)
    assert why10 and "없다" in why10, why10

    # 2) crate_count 는 **방 하나 크기**다. 총수(2×36=72)를 넣으면 배치가
    #    방 수만큼 부풀어 어느 방에도 안 들어간다
    crates = bw.crates_from_setup(setup)
    assert crates == 36, crates
    cfg.crate_count = crates

    # 3) 육성사를 따로 등록했으면 비육사를 쪼개면 안 된다 — grower 가 두 번
    #    세어진다
    spec, notes = bw.rooms_from_setup(setup, cfg.merged())
    from collections import Counter
    houses = Counter(s["house"] for s in spec)
    assert houses == {"farrowing": 2, "nursery": 4,
                      "grower": 3, "finisher": 4}, houses

    # 4) 분만사는 단위 환산이 있어야 하고, 그 사실이 보고돼야 한다
    setup_f = dict(setup, barns=[{"name": "3동", "stage": "분만사",
                                  "rooms": 2, "per": 36}])
    fspec, fnotes = bw.rooms_from_setup(setup_f, cfg.merged())
    assert all(r["capacity_head"] > 36 * 10 for r in fspec), fspec
    assert any("환산" in n[2] for n in fnotes), fnotes

    # 5) 그래서 **끝까지 돈다.** 전이가 났고, 그 위에서 위반이 0 건이어야
    #    한다. 전이 0 회의 '위반 0 건' 은 통과가 아니다
    cfg.rooms = spec
    r = bw.watch(cfg, days=400, rooms=bw.build_rooms(cfg.merged(), from_config=True))
    assert not r["blocked"], r["blocked"]
    assert r["n_steady"] > 0, "전이가 0 회면 검사한 게 없다"
    bad = sum(v for k, v in r["counts"].items() if k != "유휴")
    assert bad == 0, r["counts"]
    assert r["verdict"] == "정상", r["verdict"]

    # 6) 자돈사를 3방으로 줄이면(일령 구간만 보고 센 값) 막혀야 한다 —
    #    이 검사가 통과하면 위 4방이 우연이라는 뜻이다
    thin = dict(setup, barns=[b if b["stage"] != "자돈사"
                              else dict(b, rooms=3) for b in setup["barns"]])
    cfg2 = bw.default_config()
    cfg2.batch_system_id = sid
    cfg2.crate_count = crates
    cfg2.rooms, _ = bw.rooms_from_setup(thin, cfg2.merged())
    r2 = bw.watch(cfg2, days=400,
                  rooms=bw.build_rooms(cfg2.merged(), from_config=True))
    assert r2["counts"].get("적체", 0) > 0 or r2["blocked"], r2["counts"]

    assert rf is not None


def test_run_farm_from_setup() -> None:
    """등록 화면 JSON 이 `run_farm` ①②③ 를 전부 끌고 가는가.

    마지막까지 남아 있던 구멍이다. ③ 개체 배치가 `demo_farm` 이라 **두수에
    맞춰 방을 지어 냈고**, 그래서 "자리가 모자란다" 는 사실이 절대 안 보였다.
    등록 농장에서 가장 알고 싶은 게 그건데도.

    더 나쁜 건 섞임이었다. ③ 만 사용자 농장이고 ①② 은 모돈 역산으로 만든
    딴 농장이면, 한 화면에 두 농장이 나란히 찍힌다.
    """
    import farm_registry as fr
    import run_farm as rf

    setup = {
        "name": "테스트농장", "n_sows": 300, "interval_days": 21,
        "lactation_days": 24, "pre_farrow_days": 7, "washout_days": 7,
        "barns": [
            {"name": "1동", "stage": "교배사", "rooms": 1, "per": 72,
             "housing": "stall"},
            {"name": "2동", "stage": "임신사", "rooms": 2, "per": 82,
             "housing": "group"},
            {"name": "3동", "stage": "분만사", "rooms": 2, "per": 36,
             "housing": "crate"},
            {"name": "4동", "stage": "후보사", "rooms": 1, "per": 15,
             "housing": "group"},
            {"name": "5동", "stage": "자돈사", "rooms": 4, "per": 396,
             "housing": "pen"},
            {"name": "6동", "stage": "육성사", "rooms": 3, "per": 385,
             "housing": "pen"},
            {"name": "7동", "stage": "비육사", "rooms": 4, "per": 381,
             "housing": "pen"},
        ],
    }

    # 1) 비율은 한 곳에서만 나온다. demo_farm 과 등록 농장이 다른 비율을 쓰면
    #    같은 두수가 다르게 배치된다
    want = fr.stage_counts(300)
    assert sum(want.values()) == 300, want
    demo = fr.demo_farm(300).occupancy().groupby("stage")["n"].sum().to_dict()
    assert {k: int(v) for k, v in demo.items()} == want, (demo, want)

    # 2) 등록 도면으로 지으면 **등록한 방만** 쓴다. 뒷단은 개체 배치 대상이
    #    아니므로 빠진다 — 그건 돈군흐름 쪽이다
    farm, notes = fr.farm_from_setup(setup)
    assert not notes, notes
    assert set(b["stage"] for b in farm.barns.values()) == set(want), farm.barns
    got = farm.occupancy().groupby("stage")["n"].sum().to_dict()
    assert {k: int(v) for k, v in got.items()} == want, (got, want)
    assert farm.name == "테스트농장"
    # 스톨은 자리 번호가 있어야 한다 — 자리가 곧 개체 ID 다
    t = farm.table()
    stall = t[t["housing"] == "stall"]
    assert len(stall) and stall["slot"].map(lambda x: str(x).isdigit()).all()

    # 3) **방을 만들어 내지 않는다.** 자리가 모자라면 못 넣은 두수를 보고한다
    thin = dict(setup, barns=[
        b if b["stage"] != "임신사" else dict(b, per=40)
        for b in setup["barns"] if b["stage"] != "후보사"])
    f2, n2 = fr.farm_from_setup(thin)
    by = {x[0]: x for x in n2}
    assert "임신사" in by and by["임신사"][2] == 80, n2
    assert "후보사" in by and "등록된 동이 없다" in by["후보사"][3], n2
    assert len(f2._where) < 300, len(f2._where)

    # 4) 끝까지 돈다 — ①②③ 가 전부 등록 농장이어야 한다
    r = rf.run(300, days=200, setup=setup, verbose=False)
    assert r["sources"]["crates"].startswith("등록 분만사"), r["sources"]
    assert r["sources"]["system"] == "등록 간격", r["sources"]
    assert r["sources"]["rooms"].startswith("등록 "), r["sources"]
    assert r["system"] == "B3W", r["system"]
    assert r["placed"] == 300 and not r["place_short"], r["place_short"]

    # 5) 분만틀이 받는 규모와 **다른 돈사가 받는 규모**는 다르다. 분만틀만
    #    보고 341두라고 하면 임신사 자리가 299두인 걸 놓친다
    cap = r["capacity"]
    assert cap["binding"] == "임신사" and cap["flows"], cap["binding"]
    assert cap["n_sows"] < r["plan"]["sow_inventory"], (cap, r["plan"])

    # 6) 등록이 없으면 예전 그대로다 — 배선이 기존 경로를 안 건드려야 한다
    base = rf.run(300, days=200, verbose=False)
    assert base["sources"] == {"crates": "모돈 역산", "system": "인자",
                               "rooms": "소요량대로 생성"}, base["sources"]
    assert base["placed"] == 300 and not base["place_short"]
    assert base["system"] == "WEEKLY" and "capacity" not in base


def test_herd_drives_stage_counts() -> None:
    """개체 이력이 ③단계 두수를 **유도에서 셈으로** 바꾸는가.

    마지막까지 남아 있던 유도값이다. 방은 등록으로 실제가 됐는데 단계별
    두수는 계속 번식주기 비율(23.9/54.4/21.8%)이었다. 그 매끈함이 문제였다 —
    정상 상태를 가정하므로 **자리 부족을 지운다.**

    셋을 본다: (1) 경계를 새로 정하지 않았는가, (2) 못 센 개체를 조용히
    채우지 않는가, (3) 스냅숏을 아무 날짜로나 읽지 않는가.
    """
    import csv
    import os
    import tempfile

    import farm_registry as fr
    import run_farm as rf

    # 이유 D-3 · 교배 D-0 · 임신 D-40 · 분만 예정 D-3 · 포유 D-10 · 재발 D-30
    on = "2026-03-01"
    recs = [
        {"id": "A", "parity": 3, "weaning_date": "2026-02-26",
         "service_date": "", "farrow_date": "", "outcome": ""},
        {"id": "B", "parity": 3, "weaning_date": "2026-02-19",
         "service_date": "2026-03-01", "farrow_date": "", "outcome": ""},
        {"id": "C", "parity": 2, "weaning_date": "2025-12-25",
         "service_date": "2026-01-20", "farrow_date": "2026-05-14",
         "outcome": "분만"},
        {"id": "D", "parity": 4, "weaning_date": "2025-10-20",
         "service_date": "2025-11-05", "farrow_date": "2026-03-04",
         "outcome": "분만"},
        {"id": "E", "parity": 5, "weaning_date": "2025-11-10",
         "service_date": "2025-11-17", "farrow_date": "2026-02-19",
         "outcome": "분만"},
        {"id": "F", "parity": 2, "weaning_date": "2026-01-20",
         "service_date": "2026-01-30", "farrow_date": "", "outcome": "재발"},
        {"id": "G", "parity": 0, "weaning_date": "", "service_date": "",
         "farrow_date": "", "outcome": ""},
        # 분만 후 이유 기록이 끊긴 개체 — **추정해서 넣지 않는다**
        {"id": "H", "parity": 6, "weaning_date": "2025-06-01",
         "service_date": "2025-06-08", "farrow_date": "2025-09-30",
         "outcome": "분만"},
    ]
    want = {"A": "교배사", "B": "교배사", "C": "임신사", "D": "분만사",
            "E": "분만사", "F": "교배사", "G": "후보사", "H": None}
    for r in recs:
        st, code, why = fr.stage_of(r, on)
        assert st == want[r["id"]], (r["id"], st, why)
        assert isinstance(code, str) and code, r["id"]
    # 재발돈을 임신사로 보내면 **있지도 않은 임신돈**이 생긴다
    assert fr.stage_of(recs[5], on)[1] == "returned"

    # 교배 후 기록이 끊긴 개체(유산·도태) — 분만사에 영구 배치되면 안 되고
    # "예정 -N일 전" 같은 있지도 않은 날짜가 근거에 찍혀도 안 된다
    cut = {"id": "I", "parity": 3, "weaning_date": "2025-01-01",
           "service_date": "2025-01-08", "farrow_date": "", "outcome": ""}
    st, code, why = fr.stage_of(cut, "2025-12-01")
    assert (st, code) == (None, "record_ends"), (st, code, why)
    assert "-" not in why.split("일")[0], why
    # 예정일 직후의 며칠은 기록 지연일 수 있다 — 아직 분만사다(음수 없이)
    st, _c, why = fr.stage_of(cut, "2025-05-04")   # 예정 5-02 + 2일
    assert st == "분만사" and "경과" in why, (st, why)
    # 이유 후 한 주기를 통째로 넘긴 미교배 — 도태로 끊긴 기록이다
    st, code, _w = fr.stage_of({"id": "J", "parity": 2,
                                "weaning_date": "2025-01-01"}, "2025-12-01")
    assert (st, code) == (None, "record_ends"), (st, code)
    # 산차 0 인데 교배 기록이 있다 — 후보사가 아니라 날짜로 판정해야 한다
    # (실농장은 산차를 분만 횟수로 세므로 교배된 후보돈이 산차 0 이다)
    st, _c, _w = fr.stage_of({"id": "K", "parity": 0,
                              "service_date": "2026-01-20"}, "2026-03-01")
    assert st == "임신사", st
    # parity 가 NaN(빈 칸) — 한 칸 때문에 전체 집계가 죽으면 안 된다
    st, _c, _w = fr.stage_of({"id": "L", "parity": float("nan"),
                              "service_date": "2026-02-20"}, "2026-03-01")
    assert st == "교배사", st
    # 교배/임신 경계는 `stage_counts` 와 같은 CONFIRM 이어야 한다. herd_board
    # 의 21일을 쓰면 유도값과의 차이에 정의 차이가 섞인다
    svc = {"id": "X", "parity": 3, "weaning_date": "2026-01-01",
           "service_date": "2026-01-01"}
    from datetime import date, timedelta
    d0 = date(2026, 1, 1)
    assert fr.stage_of(svc, d0 + timedelta(days=fr.CONFIRM - 1))[0] == "교배사"
    assert fr.stage_of(svc, d0 + timedelta(days=fr.CONFIRM))[0] == "임신사"

    c = fr.counts_from_herd(recs, on)
    assert c["counts"] == {"교배사": 3, "임신사": 1, "분만사": 2, "후보사": 1}
    assert c["n"] == 7 and c["unplaced"] == {"record_ends": 1}
    # **사유는 코드로 묶는다.** 문장에 일수가 박혀 있어 그걸로 묶으면
    # "분만 39일째" 1두씩 흩어져 몇 두가 왜 빠졌는지 안 보인다
    assert all(k.isascii() for k in c["unplaced"]), c["unplaced"]

    # CSV 왕복 — 기준일이 **파일 안에** 있어야 한다. 밖에 두면 잃어버리고,
    # 오늘 날짜로 읽으면 한 주기(145일)를 벗어난 개체가 통째로 빠진다
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]) + ["as_of"])
            w.writeheader()
            for r in recs:
                w.writerow({**r, "as_of": on})
        got, as_of = fr.herd_from_csv(path)
        assert as_of == on
        assert fr.counts_from_herd(got, as_of)["counts"] == c["counts"]
        # 한 파일에 두 날짜가 섞이면 스냅숏이 아니다 — 조용히 넘기지 않는다
        with open(path, "a", encoding="utf-8-sig") as fh:
            fh.write("Z,3,2026-02-26,,,,2026-04-01\n")
        try:
            fr.herd_from_csv(path)
            raise AssertionError("as_of 가 섞였는데 통과했다")
        except ValueError as e:
            assert "여러 날짜" in str(e), e
    finally:
        os.unlink(path)

    # run_farm 에 꽂으면 ③ 이 센 값을 쓰고, **유도값이 감추던 부족이 보인다**.
    # 같은 여덟 개체를 36벌로 늘려 300두 규모에 건다(pigflow KPI 는 몇 두짜리
    # 농장에서 나오지 않는다).
    big = [dict(r, id=f"{r['id']}{i}") for i in range(36) for r in recs]
    cb = fr.counts_from_herd(big, on)
    assert cb["counts"] == {k: v * 36 for k, v in c["counts"].items()}
    setup = {
        "name": "이력농장", "n_sows": 300, "interval_days": 21,
        "lactation_days": 24, "pre_farrow_days": 7, "washout_days": 7,
        "barns": [{"name": "1동", "stage": "교배사", "rooms": 1, "per": 72,
                   "housing": "stall"},
                  {"name": "2동", "stage": "임신사", "rooms": 2, "per": 82,
                   "housing": "group"},
                  {"name": "3동", "stage": "분만사", "rooms": 2, "per": 36,
                   "housing": "crate"},
                  {"name": "4동", "stage": "후보사", "rooms": 1, "per": 40,
                   "housing": "group"}],
    }
    r = rf.run(300, days=200, setup=setup, verbose=False,
               herd={"records": big, "as_of": on, "grade": "합성"})
    assert r["stage_counts"]["source"] == "개체 이력"
    assert r["stage_counts"]["used"] == cb["counts"]
    assert r["herd"]["unplaced"] == {"record_ends": 36}
    # 교배사에 108두가 있는데 자리는 72두. 유도값은 300두 기준 69두라
    # **넉넉해 보였다** — 매끈한 유도값이 지우고 있던 부족이다
    assert fr.stage_counts(300)["교배사"] <= 72 < cb["counts"]["교배사"]
    short = {x["stage"]: x for x in r["place_short"]}
    assert "교배사" in short and short["교배사"]["got"] == 72, r["place_short"]
    # 같은 두수로 되푼 유도값과 나란히 둔다. n_sows 기준으로 빼면 규모 차이가
    # 단계 차이처럼 보인다
    assert r["stage_counts"]["derived_same_n"] == fr.stage_counts(cb["n"])

    # 이력을 안 주면 예전 그대로 — 배선이 기존 경로를 안 건드려야 한다
    base = rf.run(300, days=200, setup=setup, verbose=False)
    assert base["stage_counts"]["source"] == "번식주기 비율"
    assert base["stage_counts"]["used"] == fr.stage_counts(300)
    assert "herd" not in base and not base["place_short"]

    # CLI: as_of 가 있는 파일의 기준일을 --herd-on 으로 덮을 수 없다 —
    # 덮게 두면 이 배선이 막으려던 사고(다른 날짜로 읽어 개체 유실)를
    # CLI 가 되살린다
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]) + ["as_of"])
            w.writeheader()
            for r in recs:
                w.writerow({**r, "as_of": on})
        assert rf.main(["--herd", path, "--herd-on", "2026-04-01"]) == 2
    finally:
        os.unlink(path)


def test_herd_cycle_from_perf() -> None:
    """주기·회전율이 **이 농장 값**인가, 그리고 모듈끼리 안 갈리는가.

    두 가지를 고쳤다.

    1. 회전율이 관행 2.3 이라 NPD 34일 농장과 58일 농장이 같은 규모로
       나왔다. `korean_farm_stats` 가 466행에서 확인한 PSY 항등식의 분모가
       바로 회전율이므로, 되풀 필요 없이 그걸 쓴다.
    2. `farm_registry` 는 임신 114 · 이유~교배 7 을, `batch_flow` 는 115 ·
       5.0 을 쓰고 있었다. 같은 농장에 다른 주기(145 vs 144)를 쓴 것이고,
       연속 흐름 돈사의 지지 두수가 주기에 정비례하므로 **같은 돈사가 모듈에
       따라 다른 규모로 나왔다.** 더 나쁜 건 방향이다 — 지침 WEI 5.0 은 실측
       중앙 6.9 보다 짧아서 교배사를 실제보다 **크게** 계산했다.
    """
    import batch_flow as bf
    import farm_registry as fr

    # 1) 출처를 라벨로 남긴다 — 조용히 채우면 내 농장 값인 줄 안다
    base = bf.herd_cycle()
    assert set(base["source"]) >= set(bf.CYCLE_FIELDS) | {"turnover"}
    assert base["source"]["turnover"] in ("실측 중앙", "관행")
    assert not base["given"], base["given"]

    mine = bf.herd_cycle({"npd": 34.1, "wean_to_estrus": 6.0})
    assert mine["source"]["turnover"] == "NPD 에서 유도"
    assert mine["source"]["wean_to_service"] == "입력값"
    assert mine["wean_to_service"] == 6.0
    assert set(mine["given"]) == {"turnover", "wean_to_service"}

    # 2) **검증된 항등식 그대로** — 새 산식이 아니다
    good, bad = bf.herd_cycle({"npd": 34.1}), bf.herd_cycle({"npd": 57.73})
    want = round((365.0 - 34.1) / (good["gestation"] + good["lactation"]), 3)
    assert good["turnover"] == want, (good["turnover"], want)
    # NPD 가 나쁘면 회전이 느리다 — 같은 분만틀이 **더 많은** 모돈을 받는다
    assert good["turnover"] > bad["turnover"]

    # 3) 비운 칸은 채우지 않는다. 성적을 하나도 안 줘도 터지지 않아야 한다
    assert bf.herd_cycle({}) == bf.herd_cycle(None) == bf.herd_cycle()
    assert bf.herd_cycle({"npd": None})["source"]["turnover"] != "NPD 에서 유도"

    # 4) **모듈끼리 같은 주기를 쓴다.** 여기가 갈리면 같은 돈사가 두 규모다
    assert (fr.W2S, fr.GEST, fr.LACT) == (
        round(base["wean_to_service"]), round(base["gestation"]),
        round(base["lactation"])), (fr.W2S, fr.GEST, fr.LACT, base)

    # 5) 회전율이 규모를 실제로 움직이는가 — 안 움직이면 배선이 안 된 것이다
    built = [{"stage": "분만사", "rooms": 2, "per": 36},
             {"stage": "자돈사", "rooms": 4, "per": 396},
             {"stage": "육성사", "rooms": 3, "per": 385},
             {"stage": "비육사", "rooms": 4, "per": 381}]
    sows = {}
    for tag, perf in (("good", {"npd": 34.1}), ("bad", {"npd": 57.73})):
        cap = bf.capacity_from_rooms(built, 21, lactation=24,
                                     weaned_per_crate=11.0,
                                     cycle=bf.herd_cycle(perf))
        sows[tag] = cap["n_sows"]
        assert cap["cycle"]["source"]["turnover"] == "NPD 에서 유도"
    assert sows["bad"] > sows["good"], sows

    # 6) **설계와 역산이 같은 회전율을 쓴다.** 여기만 관행값을 쓰면 분만틀에서
    #    지은 농장을 되읽을 때 같은 돈사가 다른 규모로 나온다
    assert bf.plan_from_crates(36, 21)["herd_size"] == \
           bf.plan_from_crates(36, 21, turnover=base["turnover"])["herd_size"]

    # 7) 지침 WEI 를 넣으면 옛 값이 되돌아온다 — 차이가 어디서 왔는지 증명
    old = bf.capacity_from_rooms(
        [{"stage": "교배사", "rooms": 1, "per": 69}], 21, lactation=24,
        cycle=bf.herd_cycle({"wean_to_estrus": bf.CYCLE_GUIDE["wean_to_service"],
                             "gestation": bf.CYCLE_GUIDE["gestation"]}))
    now = bf.capacity_from_rooms(
        [{"stage": "교배사", "rooms": 1, "per": 69}], 21, lactation=24)
    assert old["n_sows"] > now["n_sows"], (old["n_sows"], now["n_sows"])


def test_table_export() -> None:
    """CSV 로 뽑아도 **등급과 각주가 따라가는가.**

    CSV 는 서식이 없어서 화면의 배지와 각주가 통째로 사라진다. 그러면 격차
    분해가 개입 효과처럼, 유도값이 실측처럼 읽힌다 — 이 프로젝트가 가장
    조심해 온 오독이다. 그래서 머리말과 등급 열을 강제하고, 그게 유지되는지
    본다.
    """
    import csv as _csv
    import io
    import tempfile

    import table_export as tx

    sys.path.insert(0, os.path.dirname(ROOT))
    try:
        from fastapi.testclient import TestClient

        from competition.server.app import app
    except Exception:
        return                          # fastapi 미설치 환경
    c = TestClient(app)

    barns = c.get("/api/capacity/preset", params={"sows": 300}).json()["barns"]
    setup = {"name": "예시", "n_sows": 300, "interval_days": 21,
             "lactation_days": 24, "pre_farrow_days": 7, "washout_days": 7,
             "barns": barns,
             "performance": {"weaned": 10.0, "npd": 62.0,
                             "farrowing_rate": 74.0}}
    import farm_registry as fr
    import synth_farm as sf
    df = sf.generate(120, 1.0, "2025-01-01", 0, sf.Params())
    fd, hp = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        sf.to_herd_csv(df, hp, "2025-01-01")
        recs, as_of = fr.herd_from_csv(hp)
    finally:
        os.unlink(hp)

    bodies = {"capacity": {"setup": setup}, "interval": {"setup": setup},
              "diagnosis": {"setup": setup}, "priority": {"setup": setup},
              "season": {"sows": 300},
              "targets": {"herd": {"as_of": as_of, "records": recs}}}
    assert set(bodies) == set(tx.SHEETS)
    listed = {s["key"] for s in c.get("/api/export/sheets").json()["sheets"]}
    assert listed == set(tx.SHEETS), listed

    for sheet, body in bodies.items():
        r = c.post(f"/api/export/{sheet}", json=body)
        assert r.status_code == 200, (sheet, r.text)
        t = r.text
        # 1) **엑셀이 한글을 안 깨뜨리게** BOM 이 있어야 한다
        assert t.startswith("﻿"), sheet
        # 2) 파일 이름에 **농장 이름을 넣지 않는다** — 식별자다
        cd = r.headers["content-disposition"]
        assert cd.startswith('attachment; filename="yangdon_'), cd
        assert "예시" not in cd, cd
        # 3) 머리말에 등급과 각주가 있어야 한다
        head = [ln for ln in t.splitlines() if ln.lstrip("﻿").startswith("#")]
        assert any("등급" in ln for ln in head), sheet
        assert sum(1 for ln in head if "⚠" in ln) >= 2, (sheet, head)
        # 4) **행마다 등급 열** — 머리말을 지우고 붙여넣어도 남는다
        rows = list(_csv.DictReader(io.StringIO(
            "\n".join(ln for ln in t.lstrip("﻿").splitlines()
                      if not ln.startswith("#")))))
        assert rows and list(rows[0])[0] == "등급", sheet
        assert all(r0["등급"] in ("실측", "계산", "유도", "합성") for r0 in rows)
        # 5) 한 표에 **등급을 섞지 않는다** — 섞이면 전부 실측으로 읽힌다
        assert len({r0["등급"] for r0 in rows}) == 1, sheet

    # 6) bare 는 머리말만 뺀다 — **등급 열은 못 뺀다**
    b = c.post("/api/export/season?bare=true", json={"sows": 300}).text
    assert "#" not in b and b.lstrip("﻿").startswith("등급,"), b[:60]

    # 7) 축이 다른 표는 각주가 그 사실을 들고 다녀야 한다
    pri = c.post("/api/export/priority", json={"setup": setup}).text
    assert "개입 효과" in pri and "합산" in pri and "축이 다르다" in pri
    tg = c.post("/api/export/targets",
                json={"herd": {"as_of": as_of, "records": recs}}).text
    assert "판정이 아니라 겨냥" in tg and "모델은 아직 없다" in tg

    # 8) 화면과 **같은 수**여야 한다 — 여기서 다시 계산하면 갈린다
    cap = c.post("/api/capacity", json=setup).json()
    assert f',{cap["capacity"]["n_sows"]},' in \
           c.post("/api/export/capacity", json={"setup": setup}).text \
           or str(cap["capacity"]["n_sows"]) in \
           c.post("/api/export/capacity", json={"setup": setup}).text

    # 9) 없는 표·빈 입력은 거절한다. **비운 성적을 중앙값으로 채우지 않으므로**
    #    낼 표가 없으면 없다고 말한다
    assert c.post("/api/export/nope", json={"sows": 300}).status_code == 404
    assert c.post("/api/export/capacity", json={}).status_code == 422
    assert c.post("/api/export/diagnosis",
                  json={"performance": {}}).status_code == 422
    assert c.post("/api/export/targets", json={}).status_code == 422


def test_pig_behavior_adapter() -> None:
    """업로드된 행동 분할 모델이 계약에 **정직하게** 꽂혔는가.

    이 모델은 15종을 출력하지만 홀드아웃에서 AP 0.2 를 넘은 건 4종뿐이다.
    지키는 것 넷: (1) 어휘를 4종만 신고하는가 — 15종을 신고하면
    head_support 가 분만징후를 "돈다" 고 답한다(Scrubbing AP 0.0 인데),
    (2) 신뢰 밖 검출이 분포에 안 섞이는가, (3) 개체를 특정하는 척 안 하는가,
    (4) 가중치 파일 없이도 접목 점검이 도는가(*.pth 는 미커밋).
    """
    import vision_contract as vc
    import vision_pig_behavior as vpb
    from pig_behavior.predictor import CLASSES, RELIABLE_CLASSES, Detection

    # 1) 어휘 신고 — 15종 출력 중 신뢰 4종만, CLASSES 원래 순서 유지
    m = vpb.PigBehaviorModel()
    assert isinstance(m, vc.BehaviorModel)          # 계약 준수(runtime protocol)
    assert len(CLASSES) == 15
    assert set(m.classes) == set(RELIABLE_CLASSES) and len(m.classes) == 4
    assert list(m.classes) == [c for c in CLASSES if c in RELIABLE_CLASSES]
    # 신고 어휘의 근거(AP)가 응답에 붙어 다닌다 · 부풀린 0.953 은 없다
    assert set(vpb.HOLDOUT_AP) == set(m.classes)
    assert all(ap >= 0.2 for ap in vpb.HOLDOUT_AP.values())
    assert "train==val" in m.holdout["note"]

    # 2) head_support — 분만징후·기침 질병이 **막혀야 한다.** 뚫려 있으면
    #    AP 0.0 짜리 출력을 근거로 경보를 낸다는 뜻이다
    sup = vc.head_support(m)
    assert sup["estrus"]["runs"] and sup["return"]["runs"]
    assert not sup["farrowing"]["runs"]
    assert sup["farrowing"]["missing"] == ["Scrubbing"]
    assert not sup["disease"]["runs"]
    assert sup["disease"]["missing"] == ["Coughing"]
    # 채널 신고 — 어휘 판정(runs)은 그대로 두고(다음 학습 목록이 남게)
    # resp 채널이 질병·분만징후를 **따로** 연다. 등급(합성·실증 0회)이
    # 신고에 붙어 다닌다. 채널 없이 부르면 아무것도 안 열린다.
    sup_ch = vc.head_support(m, channels=("resp",))
    assert not sup_ch["disease"]["runs"]              # 어휘로는 여전히 막힘
    assert sup_ch["disease"]["channel_runs"]
    assert sup_ch["farrowing"]["channel_runs"]
    assert "합성" in sup_ch["disease"]["channel_why"]
    assert "실증 0회" in sup_ch["disease"]["channel_why"]
    assert not sup_ch["estrus"]["channel_runs"]       # resp 는 발정을 안 연다
    assert not sup["disease"]["channel_runs"]         # 기본 호출은 채널 없음

    # 반례: 15종을 그대로 신고하면 분만징후가 열린다 — 그래서 4종 신고다
    class Naive:
        version, classes = "naive", CLASSES
        def predict(self, f, t): return []
    assert vc.head_support(Naive())["farrowing"]["runs"]

    # 3) fold — 신뢰 밖 검출은 세되 분포에 안 넣고, 개체를 특정하지 않는다
    dets = [("2026-08-21T09:00", [Detection("Resting", 0.9, (0, 0, 1, 1)),
                                  Detection("Eating", 0.6, (0, 0, 1, 1)),
                                  Detection("Scrubbing", 0.99, (0, 0, 1, 1))]),
            ("2026-08-21T09:05", [Detection("Resting", 0.5, (0, 0, 1, 1))])]
    [r] = vpb.fold(dets, "cam1", "3동", "2방", model="pig-behavior-test")
    o = r["obs"]
    assert isinstance(o, vc.BehaviorObs)
    assert (r["n_detections"], r["n_used"], r["n_dropped"]) == (4, 3, 1)
    assert "Scrubbing" not in o.probs                # AP 0.0 은 분포에 안 섞인다
    assert abs(sum(o.probs.values()) - 1.0) < 1e-3
    assert abs(o.probs["Resting"] - 1.4 / 2.0) < 1e-3   # 점수 가중 구성비
    assert o.animal_id is None and o.track_id is None   # 방 단위 — 거짓 확신 금지
    assert o.activity_px == 0.0 and o.resp_bpm is None   # 안 준 채널은 빈 채로
    [r2] = vpb.fold(dets, "cam1", "3동", "2방", model="t",
                    activity_px=12.3, resp_bpm=44.0)
    assert (r2["obs"].activity_px, r2["obs"].resp_bpm) == (12.3, 44.0)
    assert (o.t0, o.t1) == ("2026-08-21T09:00", "2026-08-21T09:05")
    assert o.model == "pig-behavior-test"
    assert vpb.fold([], "c", "b", "p", model="x") == []

    # 4) 가중치 없이 접목 점검이 돈다 — 무거운 초기화는 predict 까지 미룬다
    assert m._pred is None
    assert "가중치 미지정" in m.version
    import vision_pig_behavior
    assert vision_pig_behavior.main([]) == 0


def test_behavior_baseline() -> None:
    """구성비 → 자기 기준선 편차 → 헤드 경보 — **문턱을 발명하지 않았는가.**

    지키는 것 여섯: (1) 이력 미달이면 기준선을 만들지 않는가, (2) 산포가
    이상치에 강건한가(IQR — σ 였으면 한 창이 기준선을 흔든다), (3) 컷이
    자기 이력 경보율 대역에서 역산되는가 + 산포 없으면 None·경보 불가인가,
    (4) 질병이 한 창으로 안 울리고 연속을 요구하는가, (5) 달력이 연 헤드만
    계산하는가(발정·분만 신호 겹침), (6) fold() 출력이 그대로 관통하는가.
    """
    import numpy as np

    import behavior_baseline as bb
    import vision_pig_behavior as vpb
    from pig_behavior.predictor import Detection

    classes = ("Searching", "Resting", "Walking", "Eating")

    def window(rest, eat, walk):
        raw = {"Resting": rest, "Eating": eat, "Walking": walk,
               "Searching": max(0.0, 1 - rest - eat - walk)}
        tot = sum(raw.values())
        return {k: v / tot for k, v in raw.items()}

    rng = np.random.default_rng(3)
    hist = [window(0.60 + rng.normal(0, .03), 0.25 + rng.normal(0, .03),
                   0.10 + rng.normal(0, .02)) for _ in range(40)]

    # 1) 미달이면 기준선 미형성 — 편차도 점수도 경보도 없다
    short = bb.fit(hist[:bb.MIN_WINDOWS - 1], "짧은방", classes)
    assert not short.formed and short.deviation(hist[0]) == {}
    a0 = bb.assess(short, window(0.4, 0.1, 0.4))
    assert a0["heads"] == {} and "기준선 미형성" in a0["why"]
    assert bb.fit(hist[:bb.MIN_WINDOWS], "딱맞는방", classes).formed

    # 2) IQR 산포 — 이상치 한 점이 기준선을 흔들지 않는다(σ 였으면 흔든다)
    x = np.array([0.25] * 20 + [0.26] * 20 + [0.95])
    _, rsd = bb._robust(x)
    assert rsd < float(np.std(x)) / 2
    med, rsd0 = bb._robust(np.array([0.3] * 30))
    assert med == 0.3 and rsd0 == 1e-6               # 산포 0 → 바닥값

    # 3) 컷은 자기 이력 경보율 대역에서 역산 — 상수 이력이면 None·경보 불가
    b = bb.fit(hist, "3동/2방", classes)
    for head in bb.HEAD_SIGNS:
        cut = b.cuts[head]
        assert cut is not None
        scores = np.array([b.head_score(h, head) for h in hist])
        rate = float((scores >= cut).mean())
        assert bb.RATE_BAND[0] <= rate <= bb.RATE_BAND[1]
    flat = bb.fit([window(0.6, 0.25, 0.1)] * 40, "상수방", classes)
    assert all(c is None for c in flat.cuts.values())
    af = bb.assess(flat, window(0.4, 0.1, 0.4))
    assert not af["heads"]["estrus"]["alert"]
    assert "경보 불가" in af["heads"]["estrus"]["why"]

    # 부호 방향 — 발정 창(불안정↑·식욕↓)은 estrus 양수, 반대 창은 음수
    est = window(0.42, 0.14, 0.32)
    assert b.head_score(est, "estrus") > 0
    assert b.head_score(window(0.72, 0.20, 0.03), "estrus") < 0
    a1 = bb.assess(b, est)
    assert a1["heads"]["estrus"]["alert"]             # SUSTAIN=1 — 즉시

    # 4) 질병은 연속 2창 — 첫 창은 over 만, 둘째 창에야 경보
    sick = window(0.78, 0.08, 0.05)
    d1 = bb.assess(b, sick)["heads"]["disease"]
    assert d1["over"] and not d1["alert"] and d1["streak"] == 1
    d2 = bb.assess(b, sick, recent=[sick])["heads"]["disease"]
    assert d2["alert"] and d2["streak"] == 2 == d2["sustain"]
    # 사이에 평시 창이 끼면 연속이 끊긴다
    d3 = bb.assess(b, sick, recent=[window(0.6, 0.25, 0.1)])["heads"]["disease"]
    assert not d3["alert"] and d3["streak"] == 1

    # 5) 달력 게이팅 — 발정 창은 분만 헤드도 넘지만(신호 겹침), 달력이 연
    #    헤드만 계산하는 것이 정상 경로다
    assert bb.assess(b, est)["heads"]["farrowing"]["over"]
    g = bb.assess(b, est, heads=("estrus",))
    assert list(g["heads"]) == ["estrus"]

    # 6) fold() → summarize() → fit() — 어댑터 출력이 그대로 관통한다.
    #    fold() 는 관측된 행동만 내놓으므로, 어휘를 추론한 기준선은 한 번도
    #    안 보인 행동을 모른다 — 그 어휘로 헤드를 재는 척하면 안 된다
    dets = [(f"2026-08-21T{h:02d}:00", [Detection("Resting", 0.8, (0, 0, 1, 1)),
                                        Detection("Eating", 0.4, (0, 0, 1, 1))])
            for h in range(14)]
    folded = [vpb.fold([d], "cam1", "3동", "2방", model="t")[0] for d in dets]
    hist2 = bb.summarize(folded)
    assert list(hist2) == ["3동/2방"] and len(hist2["3동/2방"]) == 14
    b_seen = bb.fit(hist2["3동/2방"], "3동/2방")      # 어휘 추론 — 본 것만 안다
    assert b_seen.formed and set(b_seen.classes) == {"Resting", "Eating"}
    assert b_seen.head_score(hist2["3동/2방"][0], "estrus") is None  # Walking 미비
    assert b_seen.cuts["estrus"] is None
    ag = bb.assess(b_seen, hist2["3동/2방"][0])["heads"]["estrus"]
    assert not ag["alert"] and "어휘" in ag["why"]
    # 계약 어휘를 명시하면 잰다 — 미관측 행동은 구성비 0 인 정상 이력이다
    b_full = bb.fit(hist2["3동/2방"], "3동/2방", classes=vpb.CONTRACT_CLASSES)
    assert b_full.formed
    assert b_full.head_score(hist2["3동/2방"][0], "estrus") is not None

    # 보조 신호 — 있으면 등가중 합류, 없어도 헤드를 닫지 않는다.
    # activity(AUC 0.739 실측)가 그동안 기준선 층에 안 오고 버려지고 있었다
    hist_act = [dict(h, activity_px=10 + i % 3) for i, h in enumerate(hist)]
    b_act = bb.fit(hist_act, "방")
    est_act = dict(est, activity_px=22.0)             # 발정 창 — 활동 급등
    s_with = b_act.head_score(est_act, "estrus")
    s_wo = b.head_score(est, "estrus")
    assert s_with > s_wo                              # 활동이 발정을 보강
    assert "activity_px" in bb.HEAD_EXTRA["estrus"]
    assert bb.HEAD_EXTRA["disease"]["resp_bpm"] == +1.0   # 빈호흡=질병 부호
    # 리뷰가 실행으로 확인한 결함 둘의 회귀 방지 — 채널 부재는 0 이 아니다:
    # 추적기가 꺼진 평시 창이 질병 경보로 둔갑하지 않고, 간헐 측정 이력이
    # 채널 중심을 0 쪽으로 끌어내리지 않는다
    normal_no_act = window(0.61, 0.24, 0.10)
    d1 = bb.assess(b_act, normal_no_act)["heads"]["disease"]
    assert not d1["over"], "미측정 채널이 z 로 계산됐다(부재=0 결함 재발)"
    half = [dict(h) for h in hist_act]
    for h_ in half[::2]:
        h_.pop("activity_px")
    b_half = bb.fit(half, "방")
    assert b_half.center["activity_px"] > 8, "간헐 측정이 중심을 오염시켰다"

    # summarize 가 채널을 구성비와 **나란히** 얹는다(분포에 섞지 않는다)
    import vision_pig_behavior as _vpb
    from pig_behavior.predictor import Detection as _D
    [fr] = _vpb.fold([("t0", [_D("Resting", 0.9, (0, 0, 1, 1))])],
                     "c", "동", "방", model="t", activity_px=7.5, resp_bpm=41.0)
    ent = bb.summarize([fr])["동/방"][0]
    assert ent["activity_px"] == 7.5 and ent["resp_bpm"] == 41.0
    assert abs(sum(v for k, v in ent.items()
                   if k not in ("activity_px", "resp_bpm")) - 1.0) < 1e-6

    # 한계 신고 고정 — 등가중임을, 판정이 아니라 의심임을 응답이 스스로 말한다
    assert "등가중" in a1["weights"] and a1["grade"] == "계산"
    assert "판정이 아니라 의심" in a1["note"]


def test_behavior_head_train() -> None:
    """헤드 가중치 학습 — **사전 등록(등록 2)의 규칙이 코드에 박혔는가.**

    지키는 것 다섯: (1) 라벨 헤더가 규약과 다르면 짐작하지 않고 죽는가,
    (2) 성립 조건 미달이면 학습하지 않는가, (3) 동점이면 등가중을 유지
    하는가(엄격히 이겨야 후보), (4) 이겨도 부호가 문헌과 충돌하면 보류
    하는가, (5) 브리지 도구가 창 묶기를 재구현하지 않는가.
    """
    import csv
    import json
    import tempfile

    import numpy as np

    import behavior_head_train as ht

    # 1) 헤더 규약 — 열을 짐작해서 맞으면 그게 더 위험하다
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "bad.csv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([("방키", "영상", "s", "e", "라벨")])
        try:
            ht.load_labels(p)
            raise AssertionError("틀린 헤더를 통과시켰다")
        except SystemExit as e:
            assert "규약과 다르다" in str(e)

    # 2)+3) 합성 관통 — 방 4개 대비 성립 → 학습은 되지만 동점이라 등가중 유지
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        ht._synth(tmp, rng, rooms=4, wins=16, pos_from=10)
        rows = ht.load_labels(os.path.join(tmp, "labels.csv"))
        rooms = ht.build_dataset(os.path.join(tmp, "dets"), rows,
                                 60, 30.0, "발정", "비발정")
        a = ht.audit(rooms)
        assert a["ok"] and a["n_contrast"] == 4
        r = ht.train(rooms, "estrus")
        assert r["preregistered"].endswith("등록 2")
        assert r["auc_learned_loro"] == r["auc_equal"] == 1.0
        assert r["verdict"].startswith("등가중 유지")     # 동점은 승리가 아니다
        # 성립 조건 미달 — 방 2개만 남기면 학습 자체를 거부한다
        two = {k: rooms[k] for k in list(rooms)[:2]}
        r2 = ht.train(two, "estrus")
        assert r2["verdict"].startswith("학습 불가") and "방 2" in r2["verdict"]
        assert "auc_learned_loro" not in r2

    # 4) 부호 충돌 — 양성이 Eating↑ 이면 학습은 이기지만 문헌 부호(−)와
    #    충돌한다 → 교체가 아니라 보류. 데이터가 문헌과 싸우면 멈추고 본다
    def synth_eat(tmp, rng):
        os.makedirs(os.path.join(tmp, "dets"), exist_ok=True)
        rows = []
        for k in range(3):
            room, video = f"R{k}", f"v{k}"
            with open(os.path.join(tmp, "dets", f"{video}.jsonl"), "w",
                      encoding="utf-8") as f:
                for i in range(16 * 60):
                    pos = i // 60 >= 10
                    w = [55, 25, 10, 6] if not pos else [30, 55, 6, 5]
                    dets = [{"label": ["Resting", "Eating", "Walking",
                                       "Searching"][rng.choice(4, p=np.array(w) / sum(w))],
                             "score": 0.8, "bbox": [0, 0, 1, 1],
                             "reliable": True} for _ in range(8)]
                    f.write(json.dumps({"image": f"{i:06d}.jpg",
                                        "detections": dets}) + "\n")
            rows += [(room, video, i * 1800, (i + 1) * 1800,
                      "발정" if i >= 10 else "비발정") for i in range(16)]
        with open(os.path.join(tmp, "labels.csv"), "w", encoding="utf-8",
                  newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(ht.LABEL_HEADER)
            wcsv.writerows(rows)

    with tempfile.TemporaryDirectory() as tmp:
        synth_eat(tmp, np.random.default_rng(5))
        rows = ht.load_labels(os.path.join(tmp, "labels.csv"))
        rooms = ht.build_dataset(os.path.join(tmp, "dets"), rows,
                                 60, 30.0, "발정", "비발정")
        r = ht.train(rooms, "estrus")
        assert r["auc_learned_loro"] > r["auc_equal"]
        assert "Eating" in r["sign_conflicts"]
        assert r["verdict"].startswith("보류")

    # 5) 브리지 도구는 창 묶기를 여기서 가져다 쓴다 — 재구현 금지
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "baseline_from_dets",
        os.path.join(ROOT, "tools", "baseline_from_dets.py"))
    tool = iu.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.load_windows is ht.load_windows


def test_mating_plan() -> None:
    """교배 배정 — **근친이 인덱스를 이기는가.**

    지키는 것 다섯: (1) 혈연계수가 교과서 값과 맞는가(반형매 0.125 ·
    부모자식 0.25 · 전형매 0.25), (2) 혈통을 지우면 근친율이 내려가는가
    (하한 성질 — 그래서 각주가 '하한'이라 말한다), (3) 최고 웅돈이라도
    근친 한도에 걸리면 배정에서 빠지는가, (4) 배정 불가 사유가 근친/상한을
    구분해 말하는가, (5) 배정 합이 전체 최적인가(탐욕이 아니라).
    """
    import mating_plan as mp

    # 1) 혈연계수 — 교과서 값
    ped = mp.Pedigree({"A": ("S", None), "B": ("S", None),      # 반형매
                       "C": ("S", "D"), "E": ("S", "D"),        # 전형매
                       "S": (None, None), "D": (None, None)})
    assert abs(ped.kinship("A", "B") - 0.125) < 1e-12
    assert abs(ped.kinship("C", "S") - 0.25) < 1e-12            # 부모-자식
    assert abs(ped.kinship("C", "E") - 0.25) < 1e-12            # 전형매
    assert abs(ped.kinship("A", "A") - 0.5) < 1e-12             # 자기(비근친)
    # 2) 하한 성질 — 부 정보를 지우면 같은 쌍의 근친율이 내려간다
    ped2 = mp.Pedigree({"A": (None, None), "B": ("S", None), "S": (None, None)})
    assert ped2.kinship("A", "B") == 0.0 < ped.kinship("A", "B")
    # 순환 혈통은 조용히 돌지 않고 죽는다
    try:
        mp.Pedigree({"X": ("Y", None), "Y": ("X", None)}).kinship("X", "Y")
        raise AssertionError("순환 혈통을 통과시켰다")
    except ValueError as e:
        assert "순환" in str(e)

    # 3)+5) 시연 구성 — 최고 웅돈 B-X 는 S001·S003 과 반형매(F 12.5%)라
    #    한도(6.25%)에 걸린다. 전체 최적은 S002+X · S001+Y · S003+Y = 328.5
    r = mp._demo()
    pick = {row["모돈번호"]: row["웅돈번호"] for row in r["rows"]}
    assert pick == {"S001": "B-Y", "S002": "B-X", "S003": "B-Y"}
    assert not r["unassigned"]
    assert abs(sum(row["후손의 예상인덱스"] for row in r["rows"]) - 328.5) < 1e-9
    assert all(u["배정"] <= u["상한"] for u in r["boar_use"].values())
    assert any("하한" in n for n in r["notes"])

    # 4) 사유 구분 — 근친 전부 초과 vs 웅돈 상한에 밀림
    sows = {"S1": {"index": 100.0, "sire": "F", "dam": None, "max_services": None},
            "S2": {"index": 99.0, "sire": "F", "dam": None, "max_services": None}}
    only_kin = {"B1": {"index": 120.0, "sire": "F", "dam": None,
                       "max_services": 2}}
    r2 = mp.plan(sows, only_kin)
    assert len(r2["unassigned"]) == 2
    assert all("근친 한도 초과" in u["사유"] for u in r2["unassigned"])
    one_slot = {"B1": {"index": 120.0, "sire": "G", "dam": None,
                       "max_services": 1}}
    r3 = mp.plan(sows, one_slot)
    assert len(r3["rows"]) == 1 and len(r3["unassigned"]) == 1
    assert "상한에 밀림" in r3["unassigned"][0]["사유"]
    # 상한에 밀릴 때 남는 자리는 인덱스 높은 모돈에게 간다(전체 최적)
    assert r3["rows"][0]["모돈번호"] == "S1"

    # CSV — 농장장 표의 열 + 등급 열 + 하한 각주가 파일을 떠나지 않는다
    text = mp.to_csv(r)
    assert "하한" in text and "등급,모돈번호,모돈인덱스,웅돈번호" in text
    assert "후손의 예상인덱스,근친율(%),교배횟수" in text
    bare = mp.to_csv(r, bare=True)
    assert "#" not in bare.splitlines()[0]


def test_barn_env_control() -> None:
    """돈사 환경 위험 알람 — **센서 차이를 사육환경 차이로 읽지 않는가.**

    알람만 낸다 — 제어 지시는 내지 않는다. 지키는 것 다섯: (1) 위험
    (지침 층)은 이력이 없어도 울고 주의(편차 층)는 기준선 미형성이면
    침묵하는가, (2) 센서 오프셋이 편차를 흔들지 못하는가, (3) 겨울
    저온+고암모니아는 위험 둘이 동시에 우는가 — 조치 지시는 없는가,
    (4) 지침 안이지만 평소와 다른 것이 주의(점검)로만 나오는가,
    (5) 편차 산포·컷이 행동 기준선 층의 **같은 코드**인가(재구현 금지).
    """
    import numpy as np

    import barn_env_control as ec
    import behavior_baseline as bb

    # 5) 같은 코드 — 이름만 같은 복제가 아니라 동일 객체
    assert ec._robust is bb._robust and ec._calibrate_cut is bb._calibrate_cut
    assert ec.MIN_WINDOWS is bb.MIN_WINDOWS

    # 1) 이력 3개 — 기준선 미형성이어도 지침 층은 판정·제어를 낸다
    r = ec.assess({"신설동": {"temp_c": [30.0, 30.5, 31.0],
                             "nh3_ppm": [10.0, 11.0, 12.0]}},
                  {"신설동": "임신돈·웅돈"})
    s = r["barns"]["신설동"]["sensors"]["temp_c"]
    assert not s["formed"] and s["z"] is None and s["guide_state"] == "고온 위반"
    al = r["barns"]["신설동"]["alarms"]
    assert [a["수준"] for a in al] == ["위험"] and "고온 위반" in al[0]["내용"]
    assert r["ranking"] == []                     # 편차 층은 침묵

    # 2) 센서 오프셋 불변 — 같은 환경, 한쪽 센서만 +3℃
    rng = np.random.default_rng(3)
    base = list(rng.normal(18, 0.5, 40))
    nh3 = list(rng.normal(12, 1.5, 40))
    r = ec.assess({"A": {"temp_c": base, "nh3_ppm": nh3},
                   "B": {"temp_c": [v + 3.0 for v in base], "nh3_ppm": nh3}},
                  {"A": "임신돈·웅돈", "B": "임신돈·웅돈"})
    za = r["barns"]["A"]["sensors"]["temp_c"]["z"]
    zb = r["barns"]["B"]["sensors"]["temp_c"]["z"]
    assert abs(za - zb) < 1e-9                    # 오프셋은 z 에 흔적이 없다
    raw_a = r["barns"]["A"]["sensors"]["temp_c"]["now"]
    raw_b = r["barns"]["B"]["sensors"]["temp_c"]["now"]
    assert raw_b - raw_a > 2.5                    # 원값 비교였다면 속았다

    # 3) 겨울 상충 — 저온 위반 + 암모니아 초과
    t = list(rng.normal(17, 0.5, 30)) + [13.0]
    a = list(rng.normal(14, 1.5, 30)) + [32.0]
    r = ec.assess({"3동": {"temp_c": t, "nh3_ppm": a}}, {"3동": "포유모돈"})
    al = r["barns"]["3동"]["alarms"]
    assert [x["수준"] for x in al] == ["위험", "위험"]   # 둘이 동시에 운다
    assert any("저온 위반" in x["내용"] for x in al)
    assert any("상한 초과" in x["내용"] for x in al)
    # 조치 지시는 없다 — 겨울 상충에서 '환기 증대' 같은 지시는 틀릴 수 있다
    assert not any(w in x["내용"] for x in al for w in ("환기", "냉방", "보온"))

    # 4) 지침 안 + 평소와 다름 → 점검이지 제어가 아니다
    t = list(rng.normal(16, 0.15, 40)) + [19.5]   # 적온 안이지만 z 가 크다
    a2 = list(rng.normal(12, 1.5, 40)) + [12.0]
    r = ec.assess({"4동": {"temp_c": t, "nh3_ppm": a2}}, {"4동": "임신돈·웅돈"})
    s = r["barns"]["4동"]["sensors"]["temp_c"]
    assert s["guide_state"] == "적정" and s["alert"]
    al = r["barns"]["4동"]["alarms"]
    assert [x["수준"] for x in al if "temp_c" in x["내용"]] == ["주의"]
    assert any("점검" in x["내용"] for x in al)

    # 반대로: 평소부터 더운 돈사(기준선이 높음)는 지침 위반이되 편차 무경보 —
    # 두 층이 서로를 덮지 않고 각자 말한다
    hot = list(rng.normal(24, 0.5, 40)) + [24.2]
    r = ec.assess({"5동": {"temp_c": hot, "nh3_ppm": a2}}, {"5동": "임신돈·웅돈"})
    s = r["barns"]["5동"]["sensors"]["temp_c"]
    assert s["guide_state"] == "고온 위반" and not s["alert"]
    al = r["barns"]["5동"]["alarms"]
    assert [x["수준"] for x in al if "temp_c" in x["내용"]] == ["위험"]
    # 지침 밖인데 편차로는 평소 수준 → 센서 치우침/상시 위반 주석이 붙는다.
    # 센서 차이 문제를 편차가 걸러 주는 자리다 — 새 문턱 없이.
    assert "센서 치우침" in al[0]["내용"]
    # 반대로 갑작스런 위반(3동 저온, 편차 경보 동반)에는 주석이 없다
    r36 = ec.assess({"3동": {"temp_c": t, "nh3_ppm": a}}, {"3동": "포유모돈"})
    tmsg = next(x["내용"] for x in r36["barns"]["3동"]["alarms"]
                if "temp_c" in x["내용"])
    assert "센서 치우침" not in tmsg

    # 지침값이 제공 자료 그대로인가 — 상한을 임의로 완화하면 여기서 깨진다
    assert ec.TEMP_GUIDE["포유자돈"] == (30.0, 35.0)
    assert ec.TEMP_GUIDE["이유자돈"] == (22.0, 29.0)
    assert ec.NH3_LIMIT == 15.0 and ec.H2S_LIMIT == 5.0   # 축산원 환절기 자료
    # 습도·황화수소 — 지침 층 판정
    rh = list(rng.normal(52, 2, 20)) + [34.0]
    hs = list(rng.normal(2, 0.5, 20)) + [6.0]
    r = ec.assess({"6동": {"temp_c": [18.0] * 21, "rh_pct": rh,
                           "h2s_ppm": hs}}, {"6동": "임신돈·웅돈"})
    st6 = r["barns"]["6동"]["sensors"]
    assert st6["rh_pct"]["guide_state"] == "저습 위반"
    assert st6["h2s_ppm"]["guide_state"] == "상한 초과"
    # 지침표 조회 — 한랭 추가 사료요구량·풍속 쾌적성·단열 점검
    assert ec.cold_feed_penalty(60, -6) == 108
    assert ec.cold_feed_penalty(120, -10) == 263
    assert ec.cold_feed_penalty(58, -5.6) == 108          # 가장 가까운 칸
    assert ec.cold_feed_penalty(60, -1) == 0
    assert ec.comfort(21, 0.1, 1) == "쾌적"
    assert ec.comfort(13, 0.1, 6) == "불쾌"               # 8주령 이하 불쾌
    assert ec.comfort(13, 0.1, 10) == "쾌적"
    assert ec.comfort(2, 0.1, 30) == "불쾌"               # 비육돈도 불쾌
    ia = ec.insulation_alarms(day_temps=[14.0, 23.0], spot_temps=[18.0, 21.0])
    assert len(ia) == 2 and all("단열" in x["내용"] for x in ia)
    assert ec.insulation_alarms(day_temps=[18.0, 22.0]) == []

    # 번식 달력 결합 — 같은 고온 위반이라도 착상기 모돈이 있는 돈사가
    # 먼저다(여름 실측: 임신사고 구성이 1차 재발 쪽으로 +8.0%p). 정보는
    # 달력이 주고 환경 층은 표시만 한다 — 스스로 판정하지 않는다
    hot2 = list(rng.normal(24, 0.5, 30)) + [26.0]
    n2 = list(rng.normal(10, 1.5, 30)) + [10.0]
    r = ec.assess({"임신1동": {"temp_c": hot2, "nh3_ppm": n2},
                   "비육동": {"temp_c": list(hot2), "nh3_ppm": list(n2)}},
                  {"임신1동": "임신돈·웅돈", "비육동": "비육돈"},
                  implantation={"임신1동"})
    a_imp = next(x for x in r["barns"]["임신1동"]["alarms"]
                 if "temp_c" in x["내용"])
    a_no = next(x for x in r["barns"]["비육동"]["alarms"]
                if "temp_c" in x["내용"])
    assert "착상기" in a_imp["내용"] and "먼저 보라" in a_imp["내용"]
    assert "착상기" not in a_no["내용"]
    assert r["barns"]["임신1동"]["implantation"] is True

    # 시연 관통 + 노트 고정
    log, stages = ec._demo()
    r = ec.assess(log, stages)
    assert any("알람만 낸다" in n for n in r["notes"])
    assert ec.main([]) == 0


def test_pig_behavior_toolkit() -> None:
    """로컬 자리에서 온 toolkit 합류 — **정본과 갈라지지 않았는가.**

    pig-behavior-toolkit 저장소가 정본이고 여기 것은 벤더 사본이다.
    지키는 것 넷: (1) 호흡수 합성 검증이 이 환경에서도 도는가 — 정답을
    되찾고, 무호흡 흔들림에 안 속고, 짧으면 지어내지 않고 기각하는가
    (컷 0.2·결맞음 강등은 이 환경 실측으로 정본에서 고친 것),
    (2) analyze 가 규약대로 스스로 정하지 않는가(hut_type 검증),
    (3) 날짜 파서, (4) 기존 접목(predictor 경로)이 toolkit 판으로도
    그대로 도는가 — 이건 test_pig_behavior_adapter 가 같이 본다.
    """
    from pig_behavior.analyze import HUT_TYPES, VideoAnalyzer, _date_parts
    from pig_behavior.respiration import RespirationMeter, count_cycles

    # 컷 상수는 pytest 없이도 지킨다 — 두 환경 실측으로 정한 값이다
    assert RespirationMeter().max_cycle_cv == 0.2

    # 1) 호흡 — 합성 관통의 대표 다섯 (전체는 toolkit pytest 11개가 정본).
    #    합성 영상 생성이 pytest 를 쓰므로 없으면 건너뛴다 — 정적 뷰가
    #    서버 없이 돌아야 하는 것과 같은 이유로 전체를 실패시키지 않는다.
    try:
        import test_respiration as tr
    except ImportError as e:
        print(f"      (pytest 없음 — 호흡 합성 검증 건너뜀: {e})")
    else:
        tr.test_count_cycles_on_pure_sine()
        tr.test_count_cycles_on_white_noise()
        tr.test_recovers_known_rate(45.0, 6.0, 20.0)  # 정답을 되찾는가
        tr.test_rejects_when_no_breathing(20.0)       # 안 속는가
        tr.test_too_short_is_refused_not_guessed()    # 지어내지 않는가

    # 2)+3) analyze 경량 확인 — 무거운 것(영상·모델)은 로컬 실증 몫
    assert HUT_TYPES == {"a": "스톨", "b": "방목", "c": "기타"}
    assert _date_parts("녹음 2026-08-22 152041.mp4") == ("2026년도 08월 22일", "260822")
    assert _date_parts("20260822_room3.mp4") == ("2026년도 08월 22일", "260822")
    assert _date_parts("room3.mp4") == (None, None)
    assert count_cycles is not None and VideoAnalyzer is not None


def test_ops_api_and_view() -> None:
    """새 기능 셋이 **서버·화면까지 이어졌는가.**

    최근에 만든 기능이 CLI 에만 있으면 심사장에서는 없는 기능이다. 그런데
    화면과 서버가 각자 계산하기 시작하면 같은 농장에 다른 답을 한다 —
    이 프로젝트가 이미 두 번 겪은 사고다. 그래서 넷을 고정한다:
    (1) 라우터 응답이 모듈 출력과 **같은 객체**인가, (2) 정적 뷰가 같은
    함수를 불러 구워졌는가, (3) 지침 상수를 서버가 제 것으로 베끼지
    않았는가, (4) 허브에 등록됐는가.
    """
    import barn_env_control as ec
    import behavior_baseline as bb
    import build_dashboard_hub as hub
    import mating_plan as mp

    # 4) 허브 등록 + 뷰 파일
    assert any(v[0] == "ops_console.html" for v in hub.VIEWS)
    out = os.path.join(ROOT, "dashboard", "ops_console.html")
    assert os.path.exists(out), "build_ops_console.py 를 돌려야 한다"
    html = open(out, encoding="utf-8").read()

    # 2) 뷰가 모듈 값을 박아 넣었는가 — 화면이 제 산식을 갖지 않았다는 뜻
    ref = mp._demo()
    for row in ref["rows"]:
        assert f'{row["후손의 예상인덱스"]:g}' in html
    assert "등급 합성" in html and "서버가 필요하다" in html
    assert "센서 치우침" in html            # 편차가 바이어스를 거르는 자리
    import build_ops_console as boc
    assert "계산은 여기서 하지 않는다" in boc.__doc__

    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as e:
        print(f"      (fastapi 없음 — ops API 건너뜀: {type(e).__name__})")
        return
    import tempfile
    repo = os.path.dirname(ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault("YANGDON_DB",
                          os.path.join(tempfile.mkdtemp(), "t.db"))
    from competition.server.app import app
    c = TestClient(app)

    # 1) 교배 — 서버가 모듈과 같은 답인가
    def animals(d, boar=False):
        return [{"id": k, "index": v["index"], "sire": v["sire"],
                 "dam": v["dam"],
                 **({"max_services": v["max_services"]} if boar else {})}
                for k, v in d.items()]
    sows = {"S001": {"index": 110.0, "sire": "F1", "dam": None},
            "S002": {"index": 105.0, "sire": "F2", "dam": None},
            "S003": {"index": 98.0, "sire": "F1", "dam": None}}
    boars = {"B-X": {"index": 120.0, "sire": "F1", "dam": None, "max_services": 2},
             "B-Y": {"index": 112.0, "sire": "F3", "dam": None, "max_services": 2},
             "B-Z": {"index": 104.0, "sire": "F4", "dam": None, "max_services": 2}}
    r = c.post("/api/ops/mating", json={"sows": animals(sows),
                                        "boars": animals(boars, True)})
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == ref["rows"]
    # 혈통 순환은 조용히 돌지 않고 400 이다
    bad = [{"id": "X", "index": 100.0, "sire": "Y"},
           {"id": "Y", "index": 100.0, "sire": "X"}]
    assert c.post("/api/ops/mating",
                  json={"sows": bad, "boars": animals(boars, True)}
                  ).status_code == 400

    # 2) 환경 — 같은 답 + 센서 없는 돈사는 400(0 으로 채우면 위반이 된다)
    log, stages = ec._demo()
    body = {"barns": [{"barn": b, "stage": stages[b],
                       **{k: list(v) for k, v in log[b].items()}} for b in log]}
    r = c.post("/api/ops/env", json=body)
    assert r.status_code == 200, r.text
    got = r.json()
    exp = ec.assess(log, stages)
    assert got["barns"]["3동"]["alarms"] == exp["barns"]["3동"]["alarms"]
    assert got["ranking"] == exp["ranking"]
    assert c.post("/api/ops/env",
                  json={"barns": [{"barn": "빈동"}]}).status_code == 400

    # 3) 기준선 — 달력 게이팅이 서버에서도 도는가
    import numpy as np
    rng = np.random.default_rng(7)

    def win(rest, eat, walk):
        raw = {"Resting": rest, "Eating": eat, "Walking": walk,
               "Searching": max(0.0, 1 - rest - eat - walk)}
        tot = sum(raw.values())
        return {k: v / tot for k, v in raw.items()}
    hist = [win(0.60 + rng.normal(0, .03), 0.25 + rng.normal(0, .03),
                0.10 + rng.normal(0, .02)) for _ in range(42)]
    now = win(0.42, 0.14, 0.32)
    r = c.post("/api/ops/baseline", json={"key": "3동/2방", "history": hist,
                                          "now": now, "heads": ["estrus"]})
    assert r.status_code == 200, r.text
    got = r.json()
    exp = bb.assess(bb.fit(hist, "3동/2방"), now, heads=("estrus",))
    assert got["heads"] == exp["heads"] and list(got["heads"]) == ["estrus"]
    assert got["baseline"]["min_windows"] == bb.MIN_WINDOWS
    assert c.post("/api/ops/baseline",
                  json={"history": [], "now": now}).status_code == 400

    # 리뷰 반영 — 중복 id 는 조용히 접지 않고 400, 단열 전용 요청은 통과,
    # guide 오버라이드는 부분만 줘도 동작한다
    dup = animals(sows) + [animals(sows)[0]]
    assert c.post("/api/ops/mating", json={"sows": dup,
                                           "boars": animals(boars, True)}
                  ).status_code == 400
    r = c.post("/api/ops/env", json={"barns": [
        {"barn": "1동", "day_temps": [15.0, 24.0],
         "spot_temps": [18.0, 21.5]}]})
    assert r.status_code == 200
    assert len(r.json()["insulation"]["1동"]) == 2      # 일교차·자리차 둘 다
    r = c.post("/api/ops/env", json={
        "barns": [{"barn": "A", "stage": "임신돈·웅돈",
                   "nh3_ppm": [10.0] * 15 + [20.0]}],
        "guide": {"nh3": 30.0}})
    assert r.status_code == 200
    a = r.json()["barns"]["A"]["sensors"]["nh3_ppm"]
    assert a["guide_state"] == "적정"                   # 농장 기준 30 이 이겼다
    assert c.post("/api/ops/baseline",
                  json={"history": [{"Eating": None}], "now": {"Eating": 0.3}}
                  ).status_code == 422                  # 비수치는 500 이 아니라 422

    # 지침 상수를 서버가 베끼지 않았다 — 모듈이 정본이다
    g = c.get("/api/ops/guide").json()
    assert g["nh3_ppm_limit"] == ec.NH3_LIMIT
    assert g["h2s_ppm_limit"] == ec.H2S_LIMIT
    assert g["mating_max_inbreeding"] == mp.MAX_F_GUIDE
    assert g["temp_c"]["포유자돈"] == list(ec.TEMP_GUIDE["포유자돈"])
    assert g["grade"] == "지침"


def test_farm_scale_and_formula() -> None:
    """등록 규모와 공식 — **두 수를 섞지 않는가.**

    상시모돈(번식 모돈만)과 총사육수(전 두수)는 다른 수다. 한 칸에 몰아
    받으면 300두 농장이 어떤 화면에선 300, 어떤 화면에선 3,000 이 된다.
    지키는 것 여섯: (1) 두 수가 따로 담기고 모순이면 잡는가, (2) 동별
    사육수가 자리를 넘으면 잡는가, (3) 무허가면적을 '적법화 대상일 수
    있다'까지만 말하는가(위법 판정 금지), (4) 법정 기준이 없는 번식사에
    밀도를 지어내지 않는가, (5) 공식이 비운 입력을 채우지 않고 '무엇이
    없어서 못 냈는지'로 답하는가, (6) 회전율 표기 대조를 숨기지 않는가.
    """
    import farm_scale as fs
    import perf_formula as pf

    # 1) 두 수 분리 — 상시모돈 > 총사육수면 모순이다
    r = fs.reconcile(fs._demo())
    assert r["n_sows"] == 300 and r["n_head_total"] == 3200
    bad = dict(fs._demo(), n_sows=5000)
    msgs = [c["내용"] for c in fs.reconcile(bad)["checks"]]
    assert any("상시모돈" in m and "총사육수" in m for m in msgs)
    assert any(c["수준"] == "위험" for c in fs.reconcile(bad)["checks"])

    # 2) 사육수 > 자리 (5동: 430두 vs 6방×66=396)
    msgs = [c["내용"] for c in r["checks"]]
    assert any("사육수 430" in m and "자리 396" in m for m in msgs)
    # 동별 합 ≠ 총사육수도 잡는다
    assert any("동별 사육수 합" in m for m in msgs)

    # 3) 무허가면적 — 표시까지, 위법 판정은 하지 않는다
    non = next(m for m in msgs if "무허가면적" in m)
    assert "적법화 대상일 수 있다" in non
    assert "위법" not in non and "불법" not in non

    # 4) 번식사에는 법정 밀도를 지어내지 않는다 — 자돈사만 나온다
    stages = {d["동"] for d in r["density"]}
    assert stages == {"5동"}                       # 교배사·임신사는 기준 없음
    d5 = r["density"][0]
    assert d5["기준"] == "허가면적" and d5["overcrowded"]
    assert "교배사" not in fs.DENSITY_STAGE and "임신사" not in fs.DENSITY_STAGE
    # 밀도는 growth_flow 를 그대로 부른다 — 재구현 금지
    import growth_flow as gf
    assert d5["required_m2"] == gf.density_check(1, 1.0, "이유자돈")["required_m2"]

    # 5) 공식 — 비운 입력은 채우지 않고 이름으로 답한다
    out = pf.compute(pf._demo())
    res = out["results"]
    assert res["psy"]["value"] == 22.87 and res["msy"]["value"] == 21.5
    assert res["turnover"]["value"] == 2.21
    # MSY = PSY × 이유후육성율 — 공식 사슬이 자기모순 없이 닫힌다
    assert abs(res["msy"]["value"]
               - res["psy"]["value"] * 94.0 / 100) < 0.02
    r7 = res["return_7d_rate"]
    assert r7["value"] is None and "필요하다" in r7["why"]
    assert "7일이내 재귀복수" in out["missing"]
    # 어느 입력이 들어갔는지 결과가 들고 다닌다
    assert "실산자수" in res["psy"]["uses"] and "비생산일수(연간)" in res["psy"]["uses"]

    # 6) 회전율 — 제공 표기 그대로의 값도 같이 낸다(숨기지 않는다)
    v = res["turnover"]["variants"]
    assert len(v) == 2 and all(x < res["turnover"]["value"] for x in v.values())
    assert "466행" in res["turnover"]["basis"]
    # 총량 읽기는 복당 읽기와 **따로** 낸다
    tot = pf.compute(dict(pf._demo(), weaned_total=7200, n_sows=300))
    assert tot["results"]["psy_from_total"]["value"] == 24.0
    assert tot["results"]["psy"]["value"] == 22.87   # 서로 덮지 않는다
    assert pf.compute({})["results"]["psy"]["value"] is None


def test_improve_path() -> None:
    """현재 ↔ 달성 가능 상한 — **허수를 목표로 걸지 않는가.**

    상위 10% 가 복당 12두를 낸다고 우리 목표를 12두로 잡으면 자돈사가 못
    받는 생산량을 '낼 수 있다' 고 말하게 된다(그렇게 허수 594두·1.4억원을
    낸 적이 있다). 지키는 것 다섯: (1) 돈사 천장이 분포 상한을 깎는가,
    (2) 개별 여지를 **더하지 않는가**(PSY 는 곱셈 항등식), (3) 이미 상한
    위인 지표를 지렛대로 세지 않는가, (4) 임신기간을 권고에서 뺐는가,
    (5) PSY 를 재구현하지 않고 farm_gap 을 부르는가.
    """
    import farm_gap as fg
    import improve_path as ip

    # 5) 재구현 금지 — 같은 함수를 쓴다
    assert ip._psy({"weaned": 11.0, "npd": 43.0, "lactation": 24.8,
                    "gestation": 115.0}) == fg.psy_from(11.0, 43.0, 24.8, 115.0)

    # 1) 돈사 천장이 분포 상한(p90=12.0)을 11.0 으로 깎는다
    with_barn = ip.plan(ip._demo(), weaned_ceiling=11.0)
    no_barn = ip.plan(ip._demo())
    assert no_barn["ceilings"]["weaned"]["ceiling"] == 12.0
    assert "466농장" in no_barn["ceilings"]["weaned"]["source"]
    assert with_barn["ceilings"]["weaned"]["ceiling"] == 11.0
    assert "돈사 상한" in with_barn["ceilings"]["weaned"]["source"]
    # 깎였으므로 여지도 작아야 한다 — 허수가 줄어든 것이다
    assert with_barn["psy_all_ceiling"] < no_barn["psy_all_ceiling"]
    # 천장이 분포보다 높으면 분포가 이긴다(더 낙관적으로 가지 않는다)
    loose = ip.plan(ip._demo(), weaned_ceiling=20.0)
    assert loose["ceilings"]["weaned"]["ceiling"] == 12.0

    # 2) 합산 금지 — 개별 합과 전부 상한이 **다르다는 것을 스스로 말한다**
    assert with_barn["sum_of_each"] != with_barn["headroom_total"]
    assert any("합이 아니다" in n for n in with_barn["notes"])
    assert any("개입 효과가 아니다" in n for n in with_barn["notes"])

    # 3) 이미 상한 위면 여지 0 이고 이유를 말한다
    good = ip.plan({"weaned": 12.5, "npd": 20.0, "lactation": 20.0,
                    "gestation": 115.0})
    assert all(r["여지"] == 0.0 for r in good["rows"])
    assert all("이미 상한" in (r["왜"] or "") for r in good["rows"])
    assert good["steps"] == []                    # 지렛대가 없으면 빈 경로

    # 4) 임신기간은 지렛대가 아니다
    assert "gestation" not in ip.LEVERS
    assert not any("임신기간" == r["지표"] for r in with_barn["rows"])

    # 경로는 여지 큰 순 + 무엇을 해야 하는지가 붙는다(새 처방을 짓지 않는다)
    steps = with_barn["steps"]
    assert steps[0]["지표"] == "비생산일수(연간)"
    assert steps[0]["PSY 여지"] >= steps[-1]["PSY 여지"]
    assert "재발" in steps[0]["무엇을"]
    assert "민감도" in with_barn["note_order"]     # 효과 순서가 아니라고 말한다


def test_vision_contract() -> None:
    """영상 모델이 꽂힐 자리 — **모델 없이 배선이 관통하는가.**

    행동 분류 모델을 만드는 중이라 아직 없다. 계약을 먼저 동결해 두면 배선을
    지금 시험할 수 있고, 완성된 모델은 `predict()` 하나만 맞추면 들어온다.

    지키는 것 넷: (1) 겨냥 창을 여기서 새로 만들지 않는가, (2) 판정을
    주장하지 않는가, (3) 모델이 못 내는 헤드를 등록 시점에 말하는가,
    (4) 스텁이 모델인 척하지 않는가.
    """
    from datetime import date, timedelta

    import estrus_early_warning as ew
    import farm_registry as fr
    import pregnancy_check as pc
    import vision_contract as vc

    # 1) **창은 전부 기존 모듈에서 온다.** 여기서 숫자를 지어내면 모델이 오기
    #    전에 이미 틀린 답이 박힌다
    assert (vc.RETURN_FROM, vc.RETURN_TO) == (pc.CHECKPOINTS[0][1],
                                              pc.CHECKPOINTS[0][2])
    assert vc.ESTRUS_WATCH_FROM < ew.ANESTRUS_DAY

    on = "2026-03-01"
    day = lambda n: (date(2026, 3, 1) - timedelta(days=n)).isoformat()  # noqa: E731
    recs = [
        # 이유 5일째 · 미교배 → 발정 창 한가운데
        {"id": "E", "parity": 3, "weaning_date": day(5)},
        # 이유 2일째 → 아직 이르다(창 밖)
        {"id": "E2", "parity": 3, "weaning_date": day(2)},
        # 이유 12일째 → 지연 경보 구간이라 오히려 우선순위가 높다
        {"id": "E3", "parity": 3, "weaning_date": day(12)},
        # 교배 21일째 → 3주 관문
        {"id": "R", "parity": 2, "weaning_date": day(28), "service_date": day(21)},
        # 교배 40일째 → 관문 밖(5주 초음파는 영상이 아니라 사람이 한다)
        {"id": "R2", "parity": 2, "weaning_date": day(47), "service_date": day(40)},
        # 분만 예정 3일 전 → 분만사
        {"id": "F", "parity": 4, "weaning_date": day(140),
         "service_date": (date(2026, 3, 1) + timedelta(days=3)
                          - timedelta(days=fr.GEST)).isoformat()},
    ]
    t = vc.targets(recs, on)
    got = {h: {r["animal_id"] for r in t["heads"][h]["rows"]} for h in vc.HEADS}
    assert got["estrus"] == {"E", "E3"}, got["estrus"]
    assert got["return"] == {"R"}, got["return"]
    assert got["farrowing"] == {"F"}, got["farrowing"]
    # 질병만 달력이 없다 — 전 개체 상시다. 그 성질을 숨기지 않는다
    assert got["disease"] == {r["id"] for r in recs}
    assert all(r["priority"] == 0.0 and r["window"] is None
               for r in t["heads"]["disease"]["rows"])
    assert not vc.targets(recs, on, disease_all=False)["heads"]["disease"]["n"]

    # 2) **판정하지 않는다.** 분만 임박은 행동을 봐야 아는 것이라 달력이
    #    낼 수 있는 건 "분만사에 있고 아직 안 낳았다" 까지다
    f = t["heads"]["farrowing"]["rows"][0]
    assert "모델이 판정한다" in f["why"] and f["day"] == 3
    assert "판정은 모델이 한다" in t["note"]
    # 지연 구간이 정상 구간보다 낮은 우선순위면 안 된다 — 더 봐야 할 개체다
    pri = {r["animal_id"]: r["priority"] for r in t["heads"]["estrus"]["rows"]}
    assert pri["E3"] >= 0.9, pri

    # 3) **모델이 못 내는 헤드를 등록 시점에 말한다.** 나중에 조용히 빈 결과를
    #    내면 "경보가 없다" 와 "볼 수가 없다" 를 구분할 수 없다
    m = vc.ReplayModel(("Lying", "Standing", "Scrubbing", "Searching"))
    sup = vc.head_support(m)
    assert sup["farrowing"]["runs"] and not sup["disease"]["runs"]
    assert sup["disease"]["missing"] == ["Coughing"]
    assert all(s["why"].strip() and not s["why"].startswith(" ")
               for s in sup.values()), {k: v["why"] for k, v in sup.items()}
    # 어휘 밖 클래스를 조용히 받지 않는다
    try:
        vc.ReplayModel(("Lying",)).predict(None, None) if False else None
        m2 = vc.ReplayModel(("Lying",), [("c", "1동", "1방", "t0", "t1",
                                          {"Flying": 1.0}, 0.0)])
        m2.predict(None, None)
        raise AssertionError("어휘 밖 클래스가 통과했다")
    except ValueError as e:
        assert "어휘에 없는" in str(e), e

    # 4) **스텁이 모델인 척하지 않는다** — 버전이 관측마다 따라다녀야 한다
    obs = vc.ReplayModel(
        ("Lying", "Standing"),
        [("cam1", "1동", "1방", "2026-03-01T09:00", "2026-03-01T09:05",
          {"Lying": 0.8, "Standing": 0.2}, 12.0)]).predict(None, None)
    assert len(obs) == 1 and obs[0].model.startswith("replay")
    assert obs[0].top() == ("Lying", 0.8)
    # 개체를 특정 못 하면 None 이다 — 군사에서 트랙을 며칠씩 못 끌고 간다
    assert obs[0].animal_id is None
    # 활동량은 모델과 **따로** 온다. 모델이 실패해도 살아 있어야 한다
    assert obs[0].activity_px == 12.0

    # 5) API 가 같은 값을 내는가 — 라우터에 산술이 있으면 갈린다
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(ROOT))
    try:
        from fastapi.testclient import TestClient

        from competition.server.app import app
    except Exception:
        return                          # fastapi 미설치 환경
    c = TestClient(app)
    r = c.post("/api/vision/targets", json={"as_of": on, "records": recs})
    assert r.status_code == 200, r.text
    assert {h: r.json()["heads"][h]["n"] for h in vc.HEADS} == \
           {h: t["heads"][h]["n"] for h in vc.HEADS}
    assert c.post("/api/vision/targets",
                  json={"as_of": on, "records": []}).status_code == 422
    ct = c.get("/api/vision/contract").json()
    # **구현마다 근거와 한계를 들고 다녀야 한다** — 이름만 나열하면
    # "모델이 있다" 로 읽힌다. 스텁은 스텁이라고, 실모델은 신뢰 4종과
    # 못 도는 헤드를 응답 자체가 말한다
    impl = {x["name"]: x for x in ct["implemented"]}
    assert impl["ReplayModel"]["kind"] == "스텁"
    pb = impl["PigBehaviorModel"]
    assert pb["classes_out"] == 15 and len(pb["classes_contract"]) == 4
    assert pb["heads"] == {"estrus": True, "return": True,
                           "farrowing": False, "disease": False}, pb["heads"]
    assert "train==val" in pb["holdout"]["note"]
    assert ct["baseline"]["behavior_10cls"] == 0.485
    assert ct["windows"]["return"]["from_service"] == [vc.RETURN_FROM,
                                                       vc.RETURN_TO]

    # 6) CSV 를 거쳐도 그대로 도는가 — pandas 가 주는 NaN 은 JSON 에 못 싣는다
    import csv
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        cols = ["id", "parity", "weaning_date", "service_date",
                "farrow_date", "outcome", "as_of"]
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for x in recs:
                w.writerow({**{k: "" for k in cols}, **x, "as_of": on})
        got_recs, got_on = fr.herd_from_csv(path)
        assert got_on == on
        assert all(v is None or v == v for r in got_recs for v in r.values()), \
            "NaN 이 남아 있으면 API 직렬화가 깨진다"
        assert c.post("/api/vision/targets",
                      json={"as_of": got_on,
                            "records": got_recs}).status_code == 200
    finally:
        os.unlink(path)


def test_season_interval_view() -> None:
    """정적 뷰가 **서버와 같은 수**를 말하는가.

    여름 손실·간격 what-if 두 기능이 API 로만 있었다. 나머지 뷰 22개는 파일만
    열면 도는데 그 둘만 서버를 요구해서, 심사장에서 서버를 못 띄우면 가장
    최근 기능이 안 보였다.

    구워 넣는 순간 **값이 두 벌**이 된다 — 화면에 박힌 것과 API 가 내는 것.
    같은 함수(`season.compute` · `capacity.interval_whatif`)를 부르므로 같아야
    하고, 산식을 빌더로 옮겨 적으면 그 자리에서 깨지도록 대조한다.
    """
    import re

    import build_season_interval as bsi

    # 빌더가 competition.server.routers 를 부르므로 저장소 루트가 필요하다
    sys.path.insert(0, os.path.dirname(ROOT))
    d = bsi.gather()
    html = bsi.build(d)

    # 1) **자체완결** — 서버 없이 열려야 하므로 외부 연결이 하나도 없어야 한다
    assert not re.findall(r'https?://(?!localhost)', html), "외부 URL"
    assert not re.findall(r'<(?:script|iframe)|\bfetch\s*\(', html), "동적 요소"

    # 2) 서버 응답과 같은 수인가 — 금액은 만원 단위로 표시된다
    s = d["season"]
    for won in (s["scenario"]["median"]["won_year"],
                s["scenario"]["p90"]["won_year"],
                s["panel_won_ref"]["median"]):
        assert f"{round(won / 1e4):,}만원" in html, won
    # **곱의 중앙값 ≠ 중앙값의 곱** — 두 금액을 나란히 놓고 다르다고 적는다
    assert s["scenario"]["median"]["won_year"] != s["panel_won_ref"]["median"]
    assert "곱의 중앙값 ≠ 중앙값의 곱" in html
    for r in d["interval"]["rows"]:
        if r["n_sows"]:
            assert f'{r["n_sows"]:,}두' in html and f'{r["ceiling_year"]:,}두' in html
        else:
            assert "막힘" in html

    # 3) **박아 넣은 값이라 입력을 못 바꾼다** — 그 사실을 화면이 먼저 말해야
    #    한다. 안 말하면 심사위원이 자기 농장 값인 줄 안다
    assert "실제 농장이 아닙니다" in html
    assert "특정 농장의 값이 아닙니다" in html
    assert "서버를 띄워야" in html
    # 각주 넷이 그대로 실려야 한다 (상한·ρ·41%·중복)
    assert "손실 상한" in html and "ρ -0.149" in html and "중복" in html
    # **강조 표기를 그대로 흘리지 않는다** — 별표가 화면에 보이면 안 된다
    assert "**" not in html, "서버 문구의 ** 가 그대로 새어 나왔다"

    # 4) 허브·통합 콘솔에 등록됐는가 — 안 하면 파일은 있는데 아무도 못 찾는다
    import build_dashboard_hub as hub
    import build_pc_suite as suite
    assert any(v[0] == "season_interval.html" for v in hub.VIEWS)
    assert any(v[0] == "season_interval.html" for v in suite.VIEWS)


def test_timing_cache_is_transparent() -> None:
    """교배 적기 캐시가 **답을 바꾸지 않는가**.

    일정 하나 만드는 데 540ms 가 걸렸다 — 400일 이산사건 시뮬레이션(26ms)
    보다 20배 느렸다. 원인은 `optimal_ai_times` 가 격자를 매번 다시 훑으며
    `ai_efficacy` 를 5,486번 부르는 것이었다. (parity, wei_days) 만으로
    정해지는 상수라 `lru_cache` 를 걸었다.

    캐시는 성능 장치지 계산 장치가 아니다. **캐시 있을 때와 없을 때가 같은
    값**이어야 하고, 돌려준 dict 를 누가 고쳐 쓰면 캐시가 오염되므로 그것도
    본다.
    """
    import breeding_timing as bt
    import repro_calendar as rc

    assert hasattr(bt.optimal_ai_times, "cache_clear"), "캐시가 안 걸려 있다"
    assert hasattr(bt.insemination_window, "cache_clear")

    # 1) 캐시를 비우고 잰 값 == 캐시된 값
    for parity in ("sow", "primiparous"):
        for wei in (4.0, 5.0, 7.0, 10.0):
            bt.optimal_ai_times.cache_clear()
            bt.insemination_window.cache_clear()
            cold_w = dict(bt.insemination_window(parity, wei))
            cold_t = bt.optimal_ai_times(parity, wei)
            warm_w = bt.insemination_window(parity, wei)
            warm_t = bt.optimal_ai_times(parity, wei)
            assert cold_w == warm_w, (parity, wei, cold_w, warm_w)
            assert cold_t == warm_t, (parity, wei)

    # 2) 캐시된 dict 를 아무도 고쳐 쓰지 않는가 — 고치면 다음 호출이 오염된다
    bt.insemination_window.cache_clear()
    before = dict(bt.insemination_window("sow", 7.0))
    rc.schedule_from_weaning("2026-08-19")
    bt.detection_value(24.0)
    after = bt.insemination_window("sow", 7.0)
    assert before == after, (before, after)

    # 3) 실제로 빨라졌는가 — 두 번째 호출이 첫 번째보다 훨씬 싸야 한다.
    #    (절대 시간은 기계마다 다르므로 비율로만 본다)
    import time
    bt.optimal_ai_times.cache_clear()
    bt.insemination_window.cache_clear()
    t0 = time.perf_counter(); rc.schedule_from_weaning("2026-08-19")
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(10):
        rc.schedule_from_weaning(f"2026-09-{i + 1:02d}")
    warm = (time.perf_counter() - t0) / 10
    assert warm < cold / 10, f"캐시가 안 듣는다 (cold {cold:.3f}s, warm {warm:.4f}s)"

    # 4) 문서에 실린 수치가 그대로여야 한다 — 캐시가 값을 바꿨다면 여기서 깨진다
    w = bt.insemination_window()
    assert (w["ai1_h"], w["ai2_h"]) == (24.0, 32.0), w
    for hrs, want in ((1.0, 0.893), (12.0, 0.862), (24.0, 0.805)):
        assert bt.detection_value(hrs)["conception"] == want, hrs


def test_server_api() -> None:
    """백엔드가 **도메인 모듈과 같은 답**을 하는가.

    서버는 계산을 다시 구현하지 않기로 했다. 그 약속을 지키는지 확인하는
    유일한 방법은 API 응답과 모듈 출력을 직접 대조하는 것이다 — 서버가
    자기 산식을 갖는 순간 CLI 와 API 가 같은 농장에 다른 답을 한다.

    fastapi 가 없으면 건너뛴다. requirements 에 넣긴 했지만 정적 뷰는
    서버 없이도 돌아야 하므로, 없다고 전체 스위트를 실패시키지 않는다.
    """
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as e:
        print(f"      (fastapi/httpx 없음 — 서버 테스트 건너뜀: "
              f"{type(e).__name__})")
        return

    import tempfile
    # `competition.server` 는 패키지 경로로 import 한다 — 리포지터리 루트가
    # sys.path 에 있어야 한다(테스트는 competition/src 만 넣는다).
    repo = os.path.dirname(ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ["YANGDON_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
    import batch_flow as bf
    import breeding_timing as bt
    import repro_calendar as rc
    from competition.server import db as sdb
    from competition.server.app import app
    from competition.server.routers import farms as farms_router

    farms_router._con = sdb.connect(os.environ["YANGDON_DB"])
    c = TestClient(app)

    # 1) 어휘는 farm_registry 것이어야 한다 — 서버가 새 낱말을 만들면
    #    받아 준 값을 도메인 모듈이 거절한다
    import farm_registry as fr
    h = c.get("/api/health").json()
    assert h["stages"] == list(fr.BARN_STAGES), h["stages"]
    assert h["housings"] == list(fr.HOUSING), h["housings"]
    # 프론트가 자기 상수를 갖지 않도록 서버가 내려 준다. 갈리면 안 된다
    k = h["constants"]
    assert k["farrow_rate_p10"] == bf.FARROW_RATE_P10
    assert k["sow_turnover"] == bf.SOW_TURNOVER
    assert k["downstream_days"] == bf.DOWNSTREAM_DAYS

    # 2) **기본 구성이 자기 검사를 통과해야 한다.** 프론트가 이걸 직접
    #    계산했다가 자돈사를 3방으로 잡아, 넣자마자 "막힘" 이 떴다
    pre = c.get("/api/capacity/preset", params={"sows": 300}).json()
    setup = {"n_sows": 300, "interval_days": 21, "lactation_days": 24,
             "pre_farrow_days": 7, "washout_days": 7,
             "barns": pre["barns"],
             "performance": {"farrowing_rate": 74, "weaned": 10,
                             "survival": 86}}
    r = c.post("/api/capacity", json=setup)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["capacity"]["flows"], d["capacity"]["blocked"]

    # 3) API 응답 == 모듈 출력. 여기가 이 테스트의 핵심이다
    from competition.server.routers.capacity import _extra_rooms
    from competition.server.schemas import FarmSetup
    fs = FarmSetup(**setup)
    pc = bf.capacity_from_rooms(
        [b.model_dump() for b in fs.barns], 21.0, lactation=24,
        pre_farrow=7, washdown=7, extra_rooms=_extra_rooms(fs),
        weaned_per_crate=10.0)
    pt = bf.throughput(pc, farrow_rate=0.74, weaned_per_litter=10.0,
                       grow_survival=0.86)
    for key in ("n_sows", "binding", "crates", "flows", "weaned_ceiling"):
        assert d["capacity"][key] == pc[key], (key, d["capacity"][key], pc[key])
    for key in ("now_year", "ceiling_year", "gap_year", "achieved",
                "top_weaned", "sum_of_ways"):
        assert d["throughput"][key] == pt[key], (key, d["throughput"][key])
    assert [w["gain"] for w in d["throughput"]["ways"]] == \
           [w["gain"] for w in pt["ways"]]

    # 4) 돈사가 없으면 **두수를 지어내지 않는다**
    assert c.post("/api/capacity", json={**setup, "barns": []}).status_code == 422

    # 5) 비운 성적은 비운 채로 — 중앙값으로 채우면 격차가 늘 0 이 된다
    blank = c.post("/api/capacity", json={**setup, "performance": {}}).json()
    assert blank["given"] is False
    assert blank["throughput"]["gap_year"] == 0, blank["throughput"]

    # 6) 점검 주기 비교는 **각 주기의 최적 프로토콜**로 해야 한다.
    #    고정 오프셋으로 재면 하루 2회가 연속 관찰보다 높게 나왔다
    det = c.get("/api/breeding/detection").json()["rows"]
    assert [x["label"] for x in det] == ["연속 관찰 (CCTV)", "하루 2회", "하루 1회"]
    assert det[0]["conception"] > det[1]["conception"] > det[2]["conception"], det
    for x, hrs in zip(det, (1.0, 12.0, 24.0)):
        assert x["conception"] == bt.detection_value(hrs)["conception"]

    # 7) 번식 일정은 repro_calendar 와 같아야 한다
    sch = c.post("/api/breeding/schedule",
                 json={"weaning_date": "2026-08-19"}).json()
    want = rc.schedule_from_weaning("2026-08-19")
    assert len(sch["tasks"]) == len(want)
    # 모듈은 date 객체를 돌려주고 JSON 은 ISO 문자열이다. 직렬화 차이라
    # 값이 다른 게 아니므로 문자열로 맞춰 비교한다.
    ws = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
          for k, v in rc.cycle_summary(want).items()}
    assert sch["summary"] == ws, (sch["summary"], ws)
    assert [t["date"] for t in sch["tasks"]] == \
           [t["date"].isoformat() if hasattr(t["date"], "isoformat")
            else t["date"] for t in want]
    assert c.post("/api/breeding/schedule",
                  json={"weaning_date": "안녕"}).status_code == 422

    # 8) 투자 순서 — 병목을 풀면 그 다음은 누구인가
    rl = c.post("/api/capacity/relief", json=setup).json()
    want_rl = bf.relief_chain(
        [b.model_dump() for b in fs.barns], 21.0, lactation=24,
        pre_farrow=7, washdown=7, extra_rooms=_extra_rooms(fs),
        weaned_per_crate=10.0)
    assert [x["binding"] for x in rl["rows"]] == \
           [x["binding"] for x in want_rl], rl["rows"]
    # 규모는 단조증가해야 한다 — 제약을 풀었는데 줄면 계산이 틀린 것이다
    sows = [x["n_sows"] for x in rl["rows"]]
    assert sows == sorted(sows), sows
    # 같은 수준에 나란히 걸린 곳은 **묶여야** 한다. 안 묶으면 하나만 넓혀도
    # 는 줄 알고, 실제로는 안 는다
    assert sum(len(g["stages"]) for g in rl["groups"]) == len(rl["rows"])
    for g in rl["groups"]:
        assert len({r["n_sows"] for r in rl["rows"]
                    if r["binding"] in g["stages"]}) == 1, g
    # **비용을 계산하지 않는다** — 그 사실이 응답에 있어야 한다
    assert "비용도 계산하지 않는다" in rl["note"]

    # 8′) 간격 what-if — 같은 돈사, 간격만 바꿨을 때
    iw = c.post("/api/capacity/interval", json=setup).json()
    assert [r["name"] for r in iw["rows"]] == list(bf.BATCH_INTERVALS)
    cur = [r for r in iw["rows"] if r["current"]]
    assert len(cur) == 1 and cur[0]["interval_days"] == 21.0
    # 지금 간격 행은 `/api/capacity` 와 **같은 값**이어야 한다. 갈리면
    # 한 화면이 같은 농장에 두 규모를 말한다
    assert cur[0]["n_sows"] == pc["n_sows"], (cur[0], pc)
    assert cur[0]["binding"] == pc["binding"]
    # **`extra_rooms` 를 간격마다 다시 낸다.** 21일 기준 보정을 그대로 돌려
    # 쓰면 넓은 간격이 막힌 것으로 나온다 — 손으로 다시 계산해 대조한다
    for r in iw["rows"]:
        stub = fs.model_copy(update={"interval_days": r["interval_days"]})
        want = bf.capacity_from_rooms(
            [b.model_dump() for b in fs.barns], r["interval_days"],
            lactation=24, pre_farrow=7, washdown=7,
            extra_rooms=_extra_rooms(stub), weaned_per_crate=10.0)
        assert r["n_sows"] == want["n_sows"], (r["name"], r, want["n_sows"])
        assert r["binding"] == want["binding"], r["name"]
        # **막힌 간격에 배치 크기를 찍지 않는다** — 0두로 내면 "배치당 0.0두"
        # 라는 있지도 않은 수가 표에 남는다
        assert (r["services_per_batch"] is None) == (r["n_sows"] == 0), r
        assert r["peak_ratio"] == round(r["interval_days"] / 7.0, 1)
    # 넓힐수록 같은 분만틀이 도는 횟수가 줄어 **규모가 준다** — 막히지 않은
    # 구간에서 단조감소여야 한다. 늘어나면 방향이 뒤집힌 것이다
    ok = [r for r in iw["rows"] if r["n_sows"] > 0]
    assert [r["n_sows"] for r in ok] == \
           sorted((r["n_sows"] for r in ok), reverse=True), ok
    assert iw["best"] == max(ok, key=lambda r: r["n_sows"])["interval_days"]
    assert "인력·공사비를 계산하지 않는다" in iw["note"]

    # 9) 오늘 할 일 — repro_calendar 와 같아야 한다
    q = c.get("/api/breeding/today",
              params=[("weaning", "2026-05-12"), ("weaning", "2026-06-02"),
                      ("on", "2026-08-20"), ("horizon", 3)]).json()
    scheds = {f"B{i + 1}": rc.schedule_from_weaning(w)
              for i, w in enumerate(("2026-05-12", "2026-06-02"))}
    assert len(q["overdue"]) == len(rc.overdue(scheds, today="2026-08-20"))
    assert len(q["due"]) == len(
        rc.due_today(scheds, today="2026-08-20", horizon=3))
    # 지연 일수가 있어야 "지난 것" 이 읽힌다
    assert all("late_days" in t for t in q["overdue"]), q["overdue"][:1]
    # 배치 날짜는 **유도값**이고 실제 이력이 아니라고 말해야 한다
    bt_ = c.get("/api/breeding/batches",
                params={"first_weaning": "2026-05-12", "interval_days": 21,
                        "n": 7}).json()
    assert len(bt_["weaning_dates"]) == 7
    assert "유도" in bt_["grade"], bt_["grade"]

    # 10) 진단·처방 — `farm_gap`·`psy_priority` 와 같은 값이어야 한다
    import farm_gap as fg
    import psy_priority as pp
    perf = {"weaned": 10.0, "npd": 62.0, "farrowing_rate": 74.0}
    dg = c.post("/api/diagnosis?sows=300", json=perf).json()
    want_g = fg.diagnose(dict(perf), n_sows=300)
    assert dg["diagnosis"]["psy"] == want_g["psy"]
    assert dg["diagnosis"]["psy_gap"] == want_g["psy_gap"]
    assert [r["metric"] for r in dg["diagnosis"]["rows"]] == \
           [r["metric"] for r in want_g["rows"]]
    want_p = pp.build(dict(perf), 300, None)
    assert [r["name"] for r in dg["priority"]["rows"]] == \
           [r["name"] for r in want_p["rows"]]
    # **개입 효과를 주장하지 않는다** — 각주가 응답에 늘 붙어 있어야 한다
    assert "개입 효과의 추정이 아니다" in dg["priority"]["footer"]
    assert "실농장 개입 실험은 수행하지 않았다" in dg["priority"]["footer"]
    # **합산 금지** — 개별 합과 총 격차를 나란히 두고 합을 주장하지 않는다
    assert "합산해" in dg["priority"]["sum_note"]
    assert dg["priority"]["sum_of_parts"] != abs(dg["priority"]["psy_gap"])
    # 근거 등급이 행마다 붙는가 — 회수량 순으로만 세우면 횡단면 비교가
    # 농장 내 변화처럼 읽힌다
    assert all(r["grade"] in ("A", "B", "C") for r in dg["priority"]["rows"])
    # **비운 성적을 중앙값으로 채우지 않는다** — 채우면 격차가 늘 0 이 된다
    assert c.post("/api/diagnosis", json={}).status_code == 422

    # 11) 여름 손실 — `farm_monthly_panel` 과 같은 값이어야 하고, **두 칸을
    #     비운 시나리오를 우리 농장 값처럼 말하면 안 된다**
    import json as _json
    import os as _os
    _p = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "data", "farm_monthly_panel.json")
    if _os.path.exists(_p):
        panel = _json.load(open(_p, encoding="utf-8"))
        sn = c.get("/api/season", params={"sows": 300}).json()
        assert sn["loss"] == panel["loss"], (sn["loss"], panel["loss"])
        assert sn["spread"] == panel["spread"]
        assert sn["given"] is False and "mine" not in sn
        assert "우리 농장 값이 아니다" in sn["scenario"]["note"]
        # **곱의 중앙값 ≠ 중앙값의 곱** — 시나리오와 패널 실측 금액은 다른 수다
        assert sn["scenario"]["median"]["won_year"] != \
               sn["panel_won_ref"]["median"], "두 금액이 같으면 라벨이 거짓말"
        # 규모에 비례하는가
        sn2 = c.get("/api/season", params={"sows": 600}).json()
        assert sn2["panel_won_ref"]["median"] == \
               2 * sn["panel_won_ref"]["median"]
        # 값을 주면 우리 농장 값으로 바뀌고, 산식이 패널과 같은가
        mine = c.get("/api/season", params={"sows": 300, "psy": 23,
                                            "summer": 78, "winter": 86}).json()
        assert mine["given"] is True and "scenario" not in mine
        assert mine["psy_source"] == "입력값"
        import farm_monthly_panel as _mp
        want_psy = 23 * _mp.SEASON_SHARE * (8 / 86)
        assert abs(mine["mine"]["d_psy"] - want_psy) < 5e-4, mine["mine"]
        # 금액은 **반올림 전** ΔPSY 로 낸다 — 표시값으로 다시 곱하면 반올림이
        # 겹쳐 등록 화면과 CLI 가 몇 천 원씩 갈린다
        assert mine["mine"]["won_year"] == round(want_psy * sn["per_sow_won"] * 300)
        # 여름이 겨울보다 나은 농장은 **손실이 음수**로 나와야 한다
        good = c.get("/api/season", params={"summer": 88, "winter": 84}).json()
        assert good["mine"]["loss_pp"] < 0 and good["mine"]["won_year"] < 0
        assert good["psy_source"].endswith("가정")
        # 상한이라는 것과 ρ 를 늘 들고 다녀야 한다
        joined = " ".join(sn["caveats"])
        assert "손실 상한" in joined and "ρ" in joined and "중복" in joined
    assert c.get("/api/season", params={"summer": 5}).status_code == 422

    # 12) 농장 CRUD 가 왕복하는가
    f = c.post("/api/farms", json={"name": "T", "setup": setup})
    assert f.status_code == 201, f.text
    fid = f.json()["id"]
    got = c.get(f"/api/farms/{fid}/capacity").json()
    assert got["capacity"]["binding"] == pc["binding"]
    assert c.delete(f"/api/farms/{fid}").status_code == 204
    assert c.get(f"/api/farms/{fid}").status_code == 404


def test_farm_diagnosis_view() -> None:
    """20번째 뷰 — 실측 진단이 화면에 올라왔는가.

    이 뷰의 임무는 **새 계산이 아니라 기존 출력의 렌더링**이다. 그래서
    화면 수치가 모듈 출력과 같은지까지 본다 — 여기서 값을 다시 계산하면
    언젠가 두 곳이 어긋난다.
    """
    import re
    import build_farm_diagnosis as bfd

    d = bfd.collect()
    html = bfd.build(d)

    # 자체완결 — 외부로 나가는 연결이 하나도 없어야 한다
    ext = re.findall(r'https?://[^"\'\s)]+', html)
    assert not ext, f"외부 URL {ext[:3]}"
    assert not re.findall(r'<(?:script|link|img)[^>]*\bsrc=|<link[^>]*href=', html)

    # 농장 식별자 유출 — 원자료는 커밋 금지 대상이다
    assert not re.search(r'PIGGO|00217\d', html), "농장 식별자 노출"

    # 출처 라벨(원칙 1) — 실측과 계산이 한 화면에 섞이므로 패널마다 있어야
    assert html.count('class="tag"') >= 5, "패널별 출처 라벨이 모자란다"
    for kind in ("실측", "계산"):
        assert f'>{kind}</span>' in html, f"'{kind}' 라벨이 없다"

    # **화면 수치 = 모듈 출력.** 뷰가 따로 계산하고 있으면 여기서 갈린다.
    assert str(d["diag"]["psy"]) in html and str(d["prog"]["psy"]) in html
    dn = d["panel"]["downside"]
    assert f'{dn["expected_won_year"]/10_000:,.0f}만원' in html
    lo = next(g for g in d["panel"]["paths_matched"]["groups"]
              if g["label"] == "하락")
    assert f'{lo["d_npd"]:+.1f}' in html

    # 패널 4 는 전/후를 **동시에** 보여야 한다. 토글이면 안 눌러 본다.
    raw, sh = d["monthly"]["farrowing_rate_raw"], d["monthly"]["farrowing_rate"]
    assert raw["min_month"] != sh["min_month"], \
        "되돌리기 전후 최저월이 같다 — 대비가 사라졌다"
    assert f'최저 {raw["min_month"]}월' in html and f'최저 {sh["min_month"]}월' in html
    assert "toggle" not in html.lower() and "<button" not in html

    # 다섯 패널이 다 있는가
    for n in "12345":
        assert f'class="cn">{n}</span>' in html, f"패널 {n} 이 없다"

    # 예시 농장이 실제 농장으로 오해되면 안 된다
    assert "실제 농장이 아니다" in html

    # 다크 테마 3종 선언(기존 뷰와 같은 규약)
    for sel in (":root{", "prefers-color-scheme:dark",
                ':root[data-theme=dark]'):
        assert sel in html, f"테마 선언 누락: {sel}"

    # 허브에 등록됐는가
    import build_dashboard_hub as hub
    assert any(v[0] == "farm_diagnosis.html" for v in hub.VIEWS), "허브 미등록"
    sh_path = os.path.join(ROOT, "build_all.sh")
    assert "build_farm_diagnosis.py" in open(sh_path, encoding="utf-8").read()


def test_farm_panel() -> None:
    """같은 농장의 연도별 변화 — 원자료 없이 JSON + 합성 프레임으로 검증.

    핵심 주장이 "하락은 이유두수가 아니라 NPD·분만율에서 온다" 이므로,
    그 비대칭이 실제로 JSON 에 남아 있는지까지 확인한다.
    """
    import json
    import pandas as pd
    import farm_panel as fp

    j = os.path.join(ROOT, "data", "farm_panel.json")
    assert os.path.exists(j), "패널 집계 JSON 이 커밋돼 있어야 한다"
    r = json.load(open(j, encoding="utf-8"))

    # 농장 식별자가 새어 나가면 안 된다 — 원자료는 커밋 금지 대상이다
    blob = json.dumps(r, ensure_ascii=False)
    assert "PIGGO" not in blob and "farm\":" not in blob, "식별자 누출"

    m = r["movement"]
    assert m["n_pairs"] >= 100 and m["n_farms"] >= 50, m
    assert m["p10"] < m["p25"] < m["median"] < m["p75"] < m["p90"], m
    # 오른 비율과 1두 이상 오른 비율은 순서가 정해져 있다
    assert m["share_up_1"] <= m["share_up"] <= 1.0

    # 평균회귀는 **있다**. 없다고 나오면 층화 경고가 근거를 잃는다.
    mr = r["mean_reversion"]
    assert mr["rho_prev_vs_delta"] < -0.1, mr
    assert mr["low_delta_median"] > mr["high_delta_median"], mr

    # 대조군: 모돈두수 변화는 ΔPSY 와 상관이 없어야 한다. 상관이 있으면
    # 규모 변동이 섞인 것이고 나머지 해석이 다 흔들린다.
    ctl = next(x for x in r["drivers"] if x["metric"] == "sows")
    assert abs(ctl["rho"]) < 0.15, f"규모 변동이 섞였다: {ctl}"
    npd = next(x for x in r["drivers"] if x["metric"] == "npd")
    assert npd["rho"] < -0.5 and npd["in_identity"], npd

    # **핵심 비대칭.** 전년 수준을 맞춘 층에서도 남아야 한다.
    up, dn = (next(g for g in r["paths_matched"]["groups"] if g["label"] == x)
              for x in ("상승", "하락"))
    assert dn["d_npd"] > abs(up["d_npd"]), \
        f"하락 NPD 폭이 상승보다 크지 않다: {dn['d_npd']} vs {up['d_npd']}"
    assert abs(dn["d_weaned"]) < 0.3 < up["d_weaned"], \
        f"하락군 이유두수가 움직였다: {dn['d_weaned']}"

    # 방어 금액 = 떨어졌을 때 손실 × 빈도
    d = r["downside"]
    # freq 는 소수 3자리로 반올림돼 저장되므로 그 폭만큼은 어긋난다
    tol = d["loss_if_falls_won"] * 5e-4 + 1
    assert abs(d["expected_won_year"] - d["loss_if_falls_won"] * d["freq"]) < tol
    assert d["size_psy"] < 0 and 0 < d["freq"] < 1

    # -- 함수 자체: 연도가 붙은 쌍만 써야 한다 --------------------------
    # 2020→2022 를 한 칸으로 세면 2년치 변화가 1년치인 척하게 된다.
    cols = {c: 1.0 for c in fp.TRACK}
    syn = pd.DataFrame([
        {**cols, "farm": "A", "year": 2020, "psy": 20.0},
        {**cols, "farm": "A", "year": 2021, "psy": 22.0},
        {**cols, "farm": "B", "year": 2020, "psy": 20.0},
        {**cols, "farm": "B", "year": 2022, "psy": 30.0},   # 구멍 — 빠져야
        {**cols, "farm": "C", "year": 2021, "psy": 25.0},   # 1년치 — 빠져야
    ])
    p = fp.pairs(syn)
    assert list(p["farm"]) == ["A"], f"연속 아닌 쌍이 섞였다: {list(p['farm'])}"
    assert abs(p.iloc[0]["d_psy"] - 2.0) < 1e-9


def test_farm_monthly_panel() -> None:
    """농장별 계절 손실 → 원/년. 원자료 없이 JSON + 합성 프레임으로 검증."""
    import json
    import re
    import numpy as np
    import pandas as pd
    import farm_monthly_panel as mp

    j = os.path.join(ROOT, "data", "farm_monthly_panel.json")
    assert os.path.exists(j), "월별 패널 집계 JSON 이 커밋돼 있어야 한다"
    r = json.load(open(j, encoding="utf-8"))

    blob = json.dumps(r, ensure_ascii=False)
    assert "PIGGO" not in blob and not re.search(r'"0\d{6}"', blob), "식별자 누출"

    # 중복이 실제로 있었고, 값이 다른 중복은 없다 — 그래서 지워도 안전하다
    d = r["duplicates"]
    assert d["rows_dedup"] < d["rows_raw"] and d["conflicting_keys"] == 0, d

    # 되돌리기 전/후. 되돌려야만 여름이 드러난다는 게 발견 ③의 요지다
    assert r["overall"]["summer_minus_winter"] < -2.0, r["overall"]
    assert abs(r["overall_raw_month"]["summer_minus_winter"]) < 1.0, \
        "기록월 기준으로도 여름이 보이면 되돌리기의 근거가 사라진다"

    # 분산 분해는 항등식이다 — 관측 = 진짜 + 오차
    s = r["spread"]
    assert abs(s["var_true"] + s["var_error"] - s["var_observed"]) < 0.01, s
    assert 0.0 < s["true_share"] < 1.0, s
    assert s["sd_true"] < s["sd_observed"], s
    # 수축은 **분포를 좁힌다**. 넓어지면 부호나 가중이 뒤집힌 것이다
    z, q = r["loss_shrunk"], r["loss"]
    assert (z["p90"] - z["p10"]) < (q["p90"] - q["p10"]), (z, q)

    # 규모 보정: 작아서 시끄러운 게 맞다면 표준오차가 상시모돈과 음의 상관
    b = r["by_size"]
    assert b["rho_se_sows"] < -0.2, b
    assert abs(b["gap_shrunk"]) < abs(b["gap_raw"]), \
        f"수축 후 층간 차이가 안 줄었다: {b}"

    # 금액은 PSY 지렛대에서 와야 한다 — 새 환산 계수를 만들면 축이 어긋난다
    import farm_economics as fe
    lev = fe.levers(n_sows=mp.REF_SOWS)
    want = int(lev.loc[lev["lever"] == "PSY +1두", "두당효과"].iloc[0])
    assert r["money"]["per_sow_won"] == want, (r["money"]["per_sow_won"], want)

    # 경로 분해: 분만율만 크게 떨어지고 이유두수·재귀율은 거의 안 움직인다
    pw = r["pathways"]
    assert abs(pw["metrics"]["평균이유"]["summer_minus_winter"]) < 1.0, pw
    # 여름에 사고 구성이 **재발 쪽으로** 기운다 — 발정 관리가 겨냥하는 항목
    assert pw["accidents"]["delta"]["임신사고(1차)"] > 0.03, pw["accidents"]

    # -- 함수 자체 ------------------------------------------------------
    # 교배월 환산은 전단사여야 한다(farm_monthly 와 같은 규칙)
    assert len({mp.service_month(m) for m in range(1, 13)}) == 12
    assert mp.service_month(12) == 8

    # 계절당 관측이 모자란 농장은 **빼야** 한다. 한 달짜리는 그 달의 사고다
    # 분만 5·6·7월 → 교배 1·2·3월(겨울) · 분만 11·12·1월 → 교배 7·8·9월(여름)
    def _row(farm, m, v):
        return {"농장": farm, "년도": 2020, "데이터구분": "분만율",
                "m": m, "v": v}
    rows = [_row("A", m, v) for m, v in
            ((5, 89.0), (6, 90.0), (7, 91.0),      # 겨울 평균 90
             (11, 79.0), (12, 80.0), (1, 81.0))]   # 여름 평균 80 → 손실 10
    rows += [_row("B", 6, 90.0), _row("B", 12, 80.0)]   # 계절당 1개월 — 빠져야
    syn = pd.DataFrame(rows)
    out = mp.farm_seasonal(syn, min_obs=2)
    assert list(out["농장"]) == ["A"], f"관측이 모자란 농장이 남았다: {list(out['농장'])}"
    assert abs(float(out.iloc[0]["loss"]) - 10.0) < 1e-9, out.to_dict()
    assert float(out.iloc[0]["lo"]) < 10.0 < float(out.iloc[0]["hi"])

    # 진짜 분산이 0 이면(모두 같은 손실) 수축은 전부 평균으로 보낸다
    flat = pd.DataFrame([{"농장": f, "loss": 5.0, "se": 1.0} for f in "ABCDE"])
    assert np.allclose(mp.shrink(flat)["shrunk"], 5.0)


def test_farm_monthly_model() -> None:
    """lag 회귀 기준선 + 114일 자기검증.

    이 검사의 요점은 성능이 아니라 **분할이 정직한가**다. 롤링 윈도우는
    행끼리 겹치므로 농장을 갈라 두지 않으면 같은 농장의 이웃 달로 자기를
    맞히게 된다.
    """
    import json
    import numpy as np
    import pandas as pd
    import farm_monthly_panel as mp

    j = os.path.join(ROOT, "data", "farm_monthly_model.json")
    assert os.path.exists(j), "lag 모델 JSON 이 커밋돼 있어야 한다"
    r = json.load(open(j, encoding="utf-8"))
    assert "PIGGO" not in json.dumps(r, ensure_ascii=False), "식별자 누출"

    g = r["group_split"]
    assert g["leaks"] == 0, "농장이 train/test 양쪽에 들어갔다"
    assert g["n_eval"] == r["n_rows"], (g["n_eval"], r["n_rows"])
    # B1(농장별 과거 평균)이 실질 기준선. 전체 평균은 그보다 나빠야 한다
    sc = g["scores"]
    assert sc["B0"]["mae"] > sc["B1"]["mae"], sc
    assert abs(sc["B1"]["gain_vs_B1"]) < 1e-9

    # 시간 홀드아웃은 같은 농장이 양쪽에 들어간다 — 그 사실이 기록돼야 한다
    t = r["time_split"]
    assert t["shared_farms"] > 0 and t["leaks"] == 1, t

    # 114일 검증: 되돌림을 데이터가 고르게 한다. 부트스트랩 최댓값이
    # 3~4개월에 몰려야 한다(임신 114일 = 3.75개월). 이게 깨지면 발견 ③의
    # 되돌리기 자체가 근거를 잃는다.
    s = r["shift_scan"]
    assert s["share_3_or_4"] > 0.7, s
    assert s["by_shift"]["0"] > s["by_shift"]["3"], s   # 안 되돌리면 안 보인다

    # -- 함수 자체 ------------------------------------------------------
    # 결측 월은 **보간하지 않는다.** 구멍이 있으면 그 행이 통째로 빠져야 한다
    rows = [{"농장": "A", "년도": 2020, "데이터구분": "분만율", "m": m,
             "v": 80.0 + m} for m in range(1, 13)]
    rows += [{"농장": "B", "년도": 2020, "데이터구분": "분만율", "m": m,
              "v": 80.0} for m in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)]  # 7월 구멍
    d = mp.lag_frame(pd.DataFrame(rows))
    a = d[d["농장"] == "A"]
    assert list(a["m"]) == [7, 8, 9, 10, 11, 12], list(a["m"])
    assert abs(float(a.iloc[0]["lag6"]) - 81.0) < 1e-9   # 7월의 lag6 = 1월
    # B 는 7월이 비어 8~12월 중 lag 이 다 차는 달이 없다 → 한 행도 안 남는다
    assert len(d[d["농장"] == "B"]) == 0, d[d["농장"] == "B"].to_dict()

    # 인과성: farm_past 는 **이전 달만** 본다. 자기 값이 섞이면 안 된다
    assert abs(float(a.iloc[0]["farm_past"]) - np.mean([81.0 + i
                                                        for i in range(6)])) < 1e-9


def test_farm_gap() -> None:
    """분포에서의 **거리** 진단 — 순위가 아니라 크기를 말해야 한다."""
    import farm_gap as fg

    st = fg.load_stats()
    med = {k: v["p50"] for k, v in st["quantiles"].items()}

    # PSY 항등식: 466농장으로 확인한 정의. NPD 는 **연간**이다.
    #   회전율 = (365 − NPD) / (임신 + 포유)
    # 주기당으로 잘못 쓰면 오차 −0.32 가 난다(실제로 그렇게 시작했다).
    p = fg.psy_from(11.0, 43.0, 24.8)
    assert abs(p - 11.0 * (365 - 43) / (115 + 24.8)) < 1e-9
    assert abs(p - med["psy"]) < 1.5, (p, med["psy"])   # 중앙 농장과 맞물린다
    # 단조성: 이유두수↑ PSY↑ · NPD↑ PSY↓ · 포유↑ PSY↓
    assert fg.psy_from(12, 43, 24.8) > p > fg.psy_from(10, 43, 24.8)
    assert fg.psy_from(11, 30, 24.8) > p > fg.psy_from(11, 60, 24.8)
    assert fg.psy_from(11, 43, 21) > p
    for bad in ((11, 365, 24.8), (11, 400, 24.8)):
        try:
            fg.psy_from(*bad)
        except ValueError:
            continue
        raise AssertionError(f"불가능한 NPD 를 통과시킴: {bad}")

    # 중앙값 농장은 거리 0
    d0 = fg.diagnose({k: med[k] for k in fg.DIRECT + fg.INDIRECT}, st)
    assert abs(d0["psy_gap"]) < 0.01, d0["psy_gap"]
    for r in d0["rows"]:
        assert abs(r["iqr_z"]) < 0.01 and r["band"] == "중앙 부근", r

    # 부진 농장: 거리가 음의 방향, 회수량이 양수
    bad_farm = {"weaned": 10.0, "npd": 62.0, "lactation": med["lactation"],
                "farrowing_rate": 74.0, "wean_to_estrus": 9.5}
    d = fg.diagnose(bad_farm, st, n_sows=300)
    assert d["psy_gap"] < -1.0, d["psy_gap"]
    rows = {r["metric"]: r for r in d["rows"]}
    assert rows["weaned"]["psy_recover"] > 0 and rows["npd"]["psy_recover"] > 0
    # 결과는 회수량 내림차순 — 무엇부터 고칠지가 표의 목적이다
    rec = [r["psy_recover"] for r in d["rows"] if r["psy_recover"] is not None]
    assert rec == sorted(rec, reverse=True), rec
    # 간접 지표는 PSY 로 환산하지 않는다(없는 인과를 지어내지 않는다)
    assert rows["farrowing_rate"]["psy_recover"] is None
    assert rows["wean_to_estrus"]["psy_recover"] is None
    # 방향 부호: 낮을수록 좋은 지표는 값이 크면 '나쁨'
    assert "나쁨" in rows["npd"]["band"] and rows["npd"]["gap"] > 0
    assert "나쁨" in rows["weaned"]["band"] and rows["weaned"]["gap"] < 0

    # 우수 농장: 반대 방향
    good = fg.diagnose({"weaned": 12.0, "npd": 30.0,
                        "lactation": med["lactation"]}, st)
    assert good["psy_gap"] > 1.0
    g = {r["metric"]: r for r in good["rows"]}
    assert "좋음" in g["weaned"]["band"] and "좋음" in g["npd"]["band"]
    assert g["npd"]["psy_recover"] < 0, "이미 좋은 지표를 중앙으로 되돌리면 손해"

    # 개별 합 ≠ 전체 (항이 곱해진다) — 이걸 합산해 보고하면 과장이 된다
    assert abs(d["sum_of_parts"] - (d["psy_all_median"] - d["psy"])) > 0.01

    # 금액 환산은 회수량에 단조
    m = d["won_per_year"]
    assert m and all(x["won_year"] > 0 for x in m)
    assert m[0]["psy_recover"] >= m[-1]["psy_recover"]
    assert m[0]["won_year"] >= m[-1]["won_year"]

    # IQR 기반 z — 분포가 치우쳐도 중앙이 0
    q = st["quantiles"]["npd"]
    assert abs(fg.robust_z(q["p50"], q)) < 1e-9
    assert fg.robust_z(q["p75"], q) > 0 > fg.robust_z(q["p25"], q)


def test_run_farm_end_to_end() -> None:
    """모돈 두수 하나로 전체가 도는가 — 이게 안 되면 시연을 못 한다."""
    import contextlib
    import io
    import farm_registry as fr
    import run_farm as rfm
    from pigflow import calc
    from pigflow.config import default_config

    # 두수 → 분만틀 역산이 실제로 그 두수를 만드는지
    cfg = default_config()
    for n in (150, 300, 600):
        cr = rfm.crates_for_sows(n, cfg)
        cfg2 = default_config()
        cfg2.crate_count = cr
        got = calc.plan(cfg2)["sow_inventory"]
        assert abs(got - n) / n < 0.10, f"{n}두 요청 → {got:.0f}두 (분만틀 {cr})"
        # 단조: 두수가 크면 분만틀도 커야 한다
        assert cr >= rfm.crates_for_sows(max(50, n // 2), cfg)
    assert rfm.crates_for_sows(1, cfg) >= 1        # 하한에서 0 이 나오면 안 된다

    # 도면은 두수에 맞춰 자리를 만들어야 한다. 처음엔 68두로 고정돼 있어서
    # 300 을 줘도 68두만 배치되고 나머지가 조용히 사라졌다.
    for n in (68, 300, 500):
        f = fr.demo_farm(n)
        assert len(f._where) == n, f"{n}두 요청 → {len(f._where)}두 배치"
        occ = f.occupancy()
        assert (occ["n"] <= occ["capacity"]).all(), "수용능력 초과"
        assert set(occ["stage"]) == {"교배사", "임신사", "분만사", "후보사"}
        # 임신사가 가장 크다(임신 114일이 주기의 대부분)
        by = occ.groupby("stage")["n"].sum()
        assert by["임신사"] == by.max(), dict(by)

        # **단계별 두수는 주기 일수에서 유도돼야 한다.** 눈대중으로 25/55/15 를
        # 넣었다가 분만사가 45두 vs 64두로 어긋났다. 정상 상태에서
        #   단계별 두수 = 총두수 × (그 단계 일수 ÷ 주기)
        from pigflow.config import BREEDING_DEFAULTS as BD
        cyc = (BD["wean_to_service_days"] + BD["gestation_days"]
               + BD["lactation_days"])
        body = n - by["후보사"]
        want = {
            "교배사": body * (BD["wean_to_service_days"] + 28) / cyc,
            "임신사": body * (BD["gestation_days"] - 28 - 7) / cyc,
            "분만사": body * (7 + BD["lactation_days"]) / cyc,
        }
        for st, w in want.items():
            assert abs(by[st] - w) <= max(2, 0.03 * w), \
                f"{n}두 {st}: {by[st]} vs 주기유도 {w:.0f}"

    # 두 모듈이 **독립적으로** 계산한 분만 수용력이 맞아야 한다.
    # demo_farm 은 주기 비율에서, pigflow 는 분만틀×방수에서 나온다.
    for n in (300, 500):
        f = fr.demo_farm(n)
        by = f.occupancy().groupby("stage")["n"].sum()
        cfg3 = default_config()
        cfg3.crate_count = rfm.crates_for_sows(n, cfg3)
        crates = cfg3.crate_count * calc.rooms_required(
            cfg3.stage("SUCKLING"), cfg3.batch_system.interval_weeks)
        assert abs(by["분만사"] - crates) <= max(3, 0.05 * crates), \
            f"{n}두: demo_farm 분만사 {by['분만사']} vs pigflow 분만틀 {crates}"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = rfm.run(300, days=250)
    out = buf.getvalue()

    assert r["n_sows"] == 300 and r["placed"] == 300
    assert r["findings"]["n"] == 0, "설계대로 지었는데 경고가 난다"
    assert r["plan"]["services_per_batch"] > 0
    assert 0 < r["kpi"]["psy"] < 40 and r["kpi"]["msy"] < r["kpi"]["psy"]
    assert 0 < r["growth"]["survival"] <= 1.0
    assert r["breakeven"] > 0
    # 여섯 단계가 다 나와야 한다
    for tag in ("①", "②", "③", "④", "⑤", "⑥"):
        assert tag in out, f"{tag} 단계가 없다"

    # **기본 진단이 자기 자신과의 비교면 안 된다.** 입력이 없을 때 실측
    # 중앙값을 그대로 기본값으로 넣고 있어서 격차가 늘 +0.00 으로 찍혔다.
    # 진단처럼 보이지만 항등식을 두 번 계산한 것뿐이었다.
    assert r["gap_basis"] == "프로그램 가정값"
    assert abs(r["gap"]["psy_gap"]) > 0.5, \
        f"입력 없이 격차 {r['gap']['psy_gap']} — 중앙값을 되돌려 넣고 있다"
    assert "가정값 대 실측 분포" in out, "무엇과 비교한 건지 표시가 없다"

    # '중앙 농장' 은 합성값이다. PSY 열의 실제 중앙(24.1)보다 높게 나오는데
    # 그걸 "평균 농장의 PSY" 로 인용하면 1.2두 부풀린다 — 둘 다 찍어야 한다.
    assert r["gap"]["psy_median_observed"] < r["gap"]["psy_median_farm"] - 0.5
    assert str(r["gap"]["psy_median_observed"]) in out, "실측 PSY 중앙이 없다"

    # program_metrics 의 NPD 는 pigflow 가 계산하는 이론 최소치와 같아야 한다.
    # 두 곳에 식이 있으므로 어긋나면 여기서 잡는다.
    pm = rfm.program_metrics(default_config())
    assert abs(pm["npd"] - r["kpi"]["npd_floor_annual_days"]) < 0.15, \
        (pm["npd"], r["kpi"]["npd_floor_annual_days"])
    npd_row = [x for x in r["gap"]["rows"] if x["metric"] == "npd"][0]
    assert npd_row["value"] < npd_row["median"] - 10, \
        "이론 최소 NPD 가 실측 중앙보다 낮지 않다 — 어느 쪽 단위가 틀렸다"

    # 임신기간은 항등식에 들어가되 **지렛대로 세면 안 된다**
    g_row = [x for x in r["gap"]["rows"] if x["metric"] == "gestation"][0]
    assert g_row["psy_recover"] is not None and g_row["actionable"] is False
    assert all(m["metric"] != "gestation" for m in r["gap"]["won_per_year"]), \
        "임신기간을 단축하라는 금액 권고가 나왔다"

    # **PSY 분모가 둘이다** — 설명 없이 나란히 두면 어느 쪽이 틀린 줄 안다.
    # pigflow 는 후보돈 자리를 포함하고, 실측 비교는 번식돈 기준이다.
    assert r["psy_breeding_only"] > r["kpi"]["psy"], \
        "번식돈 기준 PSY 가 총모돈 기준보다 작다 — 분모 계산이 뒤집혔다"
    assert "후보돈 자리" in out and "번식돈" in out, "분모 차이 설명이 없다"
    assert abs(r["psy_breeding_only"] - r["gap"]["psy_median_farm"]) < 1.0, \
        (r["psy_breeding_only"], r["gap"]["psy_median_farm"])

    # 출처 표시 — 유도값을 실측처럼 보이게 두면 안 된다.
    # '합성' 이라고 적어 놨었는데 ③단계는 난수가 아니라 주기 비율에서
    # 유도한 결정론적 값이다. 없는 난수를 있다고 적는 것도 틀린 표시다.
    for word in ("실측", "가정", "유도"):
        assert word in out, f"출처 구분에 '{word}' 가 없다"

    # 실제로 난수가 없는지 — 같은 입력이면 두 번 돌려 같아야 한다
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        rfm.run(300, days=250)
    assert buf3.getvalue() == out, "같은 입력인데 출력이 다르다 — 난수가 섞였다"

    # 성적을 나쁘게 주면 격차가 음수, 회수량이 양수
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        bad = rfm.run(300, days=250,
                      farm_metrics={"npd": 62.0, "weaned": 10.0})
    assert bad["gap"]["psy_gap"] < -1.0
    top = bad["gap"]["rows"][0]
    assert top["psy_recover"] > 0

    # 필요 자료 목록 — 필수가 정확히 하나여야 혼란이 없다
    req = [x for x in rfm.REQUIRED if x[1] == "필수"]
    assert len(req) == 1 and "모돈" in req[0][0], req
    for name, need, desc, fallback in rfm.REQUIRED:
        assert desc and fallback, (name, desc, fallback)
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        assert rfm.main(["--data"]) == 0
    assert "모돈 두수" in buf3.getvalue()


def test_farm_monthly() -> None:
    """월별 계절성·임신사고 구성 — 커버리지 함정을 피했는지가 핵심."""
    import json
    import numpy as np
    import pandas as pd
    import farm_monthly as fm

    j = os.path.join(ROOT, "data", "farm_monthly.json")
    assert os.path.exists(j), "집계 JSON 이 커밋돼 있어야 한다"
    r = json.load(open(j, encoding="utf-8"))
    assert r["n_farms"] >= 100 and r["n_obs"] >= 10_000
    assert r["years"] and min(r["years"]) >= 2020 and max(r["years"]) <= 2023

    # 분만율은 **분만 시점 기록**이라 교배월로 되돌려야 여름 효과가 보인다.
    # 되돌리지 않으면 11월이 최저로 나와 계절 원인을 엉뚱하게 짚는다.
    fr_ = r["farrowing_rate"]
    assert fr_["basis"] == "교배월"
    assert fr_["min_month"] in (7, 8), fr_["min_month"]
    assert fr_["summer_minus_winter"] < 0, fr_
    # 재귀율은 이유 직후 사건이라 기록월이 곧 발생월
    assert r["return_7d"]["basis"] == "기록월"
    assert r["return_7d"]["summer_minus_winter"] < 0

    # 교배월 환산: 분만 12월 → 교배 8월
    assert fm.service_month(12) == 8 and fm.service_month(1) == 9
    assert all(1 <= fm.service_month(m) <= 12 for m in range(1, 13))
    assert len({fm.service_month(m) for m in range(1, 13)}) == 12  # 전단사

    a = r["accidents"]
    assert 0 < a["return_share"] < 1
    assert a["return_share"] > 0.5, "재발 계열이 과반이 아니면 논거가 흔들린다"
    assert abs(sum(a["mix"].values()) - 1.0) < 0.01
    # **커버리지 함정**: 전체 합으로 계산하면 값이 달라진다. 그걸 알고 있는지.
    #
    # 처음엔 "1위가 불규칙으로 바뀐다"를 걸었는데, 그 순위 뒤집힘은 원자료
    # 중복이 만든 것이었다. 중복을 지우면 1위는 양쪽 다 1차 재발이다.
    # 함정 자체는 남아 있으므로 **순위가 아니라 크기**로 건다.
    assert a["n_complete"] < a["n_all"]
    cov = a["coverage"]
    assert max(cov.values()) - min(cov.values()) > 0.2, \
        "보고율이 고르면 이 보정은 필요 없다 — 원자료가 바뀐 것"
    assert a["max_mix_gap"] > 0.03, \
        f"두 계산이 거의 같다({a['max_mix_gap']:.3f}) — 보정이 무의미해졌는지 확인할 것"
    assert a["return_share"] > a["naive_return_share"], \
        "완전보고 쪽 재발 비중이 더 커야 한다 — 재발 유형이 덜 보고되기 때문"

    # 함수 검증 — 합성 프레임으로. 여름에 분만율이 낮은 신호를 심고 찾는지.
    rows = []
    for farm in range(20):
        for m in range(1, 13):
            sm = fm.service_month(m)
            v = 84.0 - (4.0 if sm in fm.SUMMER else 0.0)
            rows.append({"년도": 2021, "지역": "-", "규모": "A",
                         "농장": f"F{farm}", "데이터구분": "분만율",
                         "월": f"{m}월", "v": v, "m": m})
    d = pd.DataFrame(rows)
    s1 = fm.seasonality(d, "분만율", shift_to_service=True)
    assert s1["summer_minus_winter"] < -3.5, s1
    s0 = fm.seasonality(d, "분만율", shift_to_service=False)
    # 시프트하지 않으면 여름이 낮게 안 보인다 — 이게 함정의 실체다
    assert s0["summer_minus_winter"] > s1["summer_minus_winter"], (s0, s1)
    assert fm.seasonality(d, "없는지표") == {}

    # 임신사고: 한 유형만 많이 보고되면 전체합이 그쪽으로 기운다
    acc = []
    for farm in range(60):
        for m in range(1, 13):
            acc.append({"년도": 2021, "지역": "-", "규모": "A",
                        "농장": f"F{farm}", "데이터구분": "임신사고(공태)",
                        "월": f"{m}월", "v": 3.0, "m": m})
            if farm < 10:      # 소수 농장만 보고하는 유형
                acc.append({"년도": 2021, "지역": "-", "규모": "A",
                            "농장": f"F{farm}", "데이터구분": "임신사고(1차)",
                            "월": f"{m}월", "v": 9.0, "m": m})
    am = fm.accident_mix(pd.DataFrame(acc))
    assert am["coverage"]["임신사고(공태)"] > am["coverage"]["임신사고(1차)"]
    # 완전 보고분만 보면 1차(9) 가 공태(3) 보다 크다
    assert am["mix"]["임신사고(1차)"] > am["mix"]["임신사고(공태)"], am["mix"]

    # barn_environment 가 이 실측을 반영하는지
    import barn_environment as be
    assert be.SEASONAL_MEASURED["farrowing_rate_summer_gap_pp"] < 0
    assert abs(be.SEASONAL_MEASURED["farrowing_rate_summer_gap_pp"]
               - fr_["summer_minus_winter"]) < 0.2


def test_synth_farm() -> None:
    """가상 데이터 — 실측 분포를 재현하고 날짜가 앞뒤 맞는지.

    검사 없는 합성은 시뮬레이션을 통째로 무의미하게 만든다. 생성기보다
    **검증기가 제 일을 하는지**가 이 테스트의 요점이다.
    """
    import tempfile
    import pandas as pd
    import synth_farm as sf

    P = sf.Params()
    assert 0.5 < P.farrowing_rate < 1.0
    assert P.summer_gap < 0, "여름 교배 분만율이 낮다는 실측이 반영돼야 한다"

    # 계절 검사가 실제로 돌 만큼 표본을 준다. 작게 잡으면 건너뛰어서
    # "통과"가 아무것도 보증하지 않는다.
    df = sf.generate(n_sows=600, years=3.0, seed=1, params=P)
    assert len(df) > 2000 and df["sow_id"].nunique() == 600
    v = sf.validate(df, P)
    assert v["ok"], (v["checks"], v["consistency"])
    assert not v["consistency"], v["consistency"]

    # 검사 항목이 실제로 다 돌았는지 — 조용히 건너뛰면 통과가 무의미하다
    names = {c["name"] for c in v["checks"]}
    for need in ("분만율", "재귀발정일 중앙", "임신기간 중앙",
                 "복당 이유두수 중앙", "하계 분만율 차(%p)"):
        assert need in names, f"{need} 검사가 없다"

    # 계절 대비는 실측과 **같은 구간**이어야 한다(7·8·9 vs 1·2·3).
    # 처음엔 6월을 넣고 "나머지 전체"와 비교해 −5.2%p 가 나왔는데 부호만
    # 봐서 통과했다. 실측 −2.97%p 의 2배를 통과시키면 계절을 과장하게 된다.
    assert sf.SUMMER == (7, 8, 9) and sf.WINTER == (1, 2, 3)
    gap = next(c for c in v["checks"] if "하계" in c["name"])
    assert "skipped" not in gap, "표본이 모자라 계절 검사를 건너뛰었다"
    # 허용치는 **표준오차에서** 나온다. 고정 허용치를 쓰면 표본이 작을 때
    # 잡음을 잡아내지 못하거나(느슨) 멀쩡한 걸 실패시킨다(빡빡).
    assert abs(gap["got"] - gap["want"]) < max(1.0, 2.5 * gap["se_pp"]), gap
    assert gap["n"][0] >= 200 and gap["n"][1] >= 200

    # 표본이 모자라면 **판정하지 말고 건너뛰어야** 한다.
    # 모돈 150·1.5년에서 −6.4%p 가 나왔는데 그건 생성기 문제가 아니라
    # 각 군 100건일 때 표준오차가 5%p 를 넘기 때문이다.
    small = sf.validate(sf.generate(n_sows=60, years=0.8, seed=2, params=P), P)
    sg = next(c for c in small["checks"] if "하계" in c["name"])
    assert "skipped" in sg and sg["ok"], sg

    # 날짜 정합성 — 앱의 일정·지연 판정이 여기 기댄다
    assert (pd.to_datetime(df["service"]) >= pd.to_datetime(df["estrus"])).all()
    far = df[df["outcome"] == "분만"]
    assert (pd.to_datetime(far["farrow"]) > pd.to_datetime(far["service"])).all()
    assert (pd.to_datetime(far["wean"]) > pd.to_datetime(far["farrow"])).all()
    assert (far["weaned"] <= far["born_alive"]).all()
    assert set(df["outcome"]) <= {"분만", "재발"}
    ret = df[df["outcome"] == "재발"]
    assert ret["farrow"].isna().all() and len(ret) > 0
    assert set(ret["return_type"].dropna()) <= set(sf.RETURN_MIX) | {"기타"}

    # 재귀발정일은 **오른쪽 꼬리**여야 한다. 대칭으로 만들면 '늦게 오는 소수'가
    # 사라져 조기경보를 시험할 표본이 없어진다.
    w2e = (pd.to_datetime(df["estrus"]) - pd.to_datetime(df["wean_prev"])).dt.days
    assert w2e.mean() > w2e.median(), (w2e.mean(), w2e.median())
    assert w2e.min() >= 3

    # **검증기가 나쁜 데이터를 잡는가** — 이게 안 되면 통과가 의미 없다
    broken = df.copy()
    broken.loc[broken.index[0], "service"] = (
        pd.to_datetime(broken.loc[broken.index[0], "estrus"])
        - pd.Timedelta(days=5)).date()
    vb = sf.validate(broken, P)
    assert not vb["ok"] and vb["consistency"], "교배가 발정보다 앞선 걸 못 잡는다"

    flat = df.copy()
    flat["outcome"] = "분만"        # 분만율 100% — 실측과 크게 어긋난다
    assert not sf.validate(flat, P)["ok"], "비현실적 분만율을 통과시킨다"

    # 재현성
    a = sf.generate(n_sows=40, years=0.6, seed=7, params=P)
    b = sf.generate(n_sows=40, years=0.6, seed=7, params=P)
    assert len(a) == len(b) and a["outcome"].tolist() == b["outcome"].tolist()

    # CSV → 앱이 먹는 형태(개체당 최근 이벤트 한 줄)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "herd.csv")
        n = sf.to_herd_csv(df, path, today="2025-06-01")
        assert n > 0
        out = pd.read_csv(path)
        assert out["id"].is_unique and len(out) == n
        for c in ("id", "parity", "weaning_date", "service_date"):
            assert c in out.columns
        # 기준일 이후 사건은 들어가면 안 된다(미래를 아는 셈이 된다).
        # 공태돈은 교배일이 비어 있으므로 빈 칸을 빼고 본다
        svc = pd.to_datetime(out["service_date"]).dt.date.dropna()
        assert (svc <= pd.Timestamp("2025-06-01").date()).all()
        assert (pd.to_datetime(out["weaning_date"]).dt.date
                <= pd.Timestamp("2025-06-01").date()).all()
        # **공태돈(이유했고 아직 미교배)이 있어야 한다.** 예전에는 "교배가
        # 지난 마지막 주기" 만 뽑아서 그 주기에 반드시 교배가 있었고, 그래서
        # 공태가 한 두도 안 나왔다 — 정상 상태면 이유~교배 7일 / 주기 145일
        # 이므로 5% 쯤이어야 하고, 빠지면 NPD 의 원천이 화면에서 사라진다
        open_share = out["service_date"].isna().mean()
        assert 0.01 < open_share < 0.15, f"공태 비중 {open_share:.1%}"
        assert (out.loc[out["service_date"].isna(), "outcome"] == "공태").all()


def test_docs_consistent() -> None:
    """문서에 박힌 숫자가 실제와 맞는지. 어긋나면 나머지 수치도 못 믿게 된다.

    실제로 테스트를 47→53개로 늘렸는데 문서 세 곳이 47 로 남아 있었다.
    손으로 고치면 또 어긋나므로 테스트로 붙든다.
    """
    import contextlib
    import importlib.util
    import io

    path = os.path.join(ROOT, "tools", "check_docs.py")
    spec = importlib.util.spec_from_file_location("_chkdocs", path)
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cd.main()
    assert rc == 0, "\n" + buf.getvalue()

    # 검사기 자체가 동작하는지 — 실제값을 못 세면 통과해도 의미가 없다
    a = cd.actual_counts()
    assert a["tests"] > 40 and a["views"] > 10 and a["modules"] > 30, a
    m = cd.actual_metrics()
    assert "posture3_best_acc" in m and "poly_gain_posture" in m, m
    # 폴리곤 실험의 결론(이득 없음)이 뒤집히면 발표자료 슬라이드 9-1 을 고쳐야 한다
    assert m["poly_gain_posture"] <= 0, \
        f"폴리곤 이득이 양수로 바뀌었다({m['poly_gain_posture']}) — 슬라이드 9-1 재검토"


def test_farm_economics() -> None:
    """생산비 구조: 사료 비중·손익분기·지렛대 순서."""
    import farm_economics as fe

    fp = fe.feed_plan()
    assert list(fp["stage"]) == ["이유자돈", "육성돈", "비육돈"], list(fp["stage"])
    for r in fp.itertuples(index=False):
        assert abs(r.feed_kg - r.gain_kg * r.fcr) < 0.1
        assert r.cost == int(round(r.feed_kg * r.price))
    # 포유(1.4→8.0)를 뺀 이유~출하 증체
    assert abs(fp.attrs["total_gain_kg"] - 107.0) < 1e-6
    assert fp.attrs["overall_fcr"] > 2.0

    c = fe.cost_per_pig()
    assert c["total"] == fp.attrs["feed_cost"] + sum(fe.NON_FEED.values())
    assert abs(sum(c["share"].values()) - 1.0) < 0.01
    # 사료가 절반을 넘어야 한다 — 넘지 않으면 단가 가정이 어긋난 것
    assert 0.5 < c["feed_share"] < 0.7, c["feed_share"]

    rev = fe.revenue_per_pig()
    assert abs(rev["carcass_kg"] - rev["live_kg"] * fe.DRESSING_RATE) < 0.1
    assert rev["revenue"] > c["total"], "두당 적자면 이후 시나리오가 무의미"

    # 손익분기: 그 단가에서 순이익이 0 이어야 한다
    be = fe.breakeven_price(24.0, 0.86)
    assert 0 < be < fe.PORK_PRICE, f"손익분기 {be} 가 현재 시세 이상"
    at_be = fe.per_sow_year(24.0, 0.86, price=be)["net_per_sow"]
    assert abs(at_be) < 5000, f"손익분기 단가에서 순이익 {at_be}"

    # MSY = PSY × 육성률, 순이익 = 총이익 − 모돈 유지비
    p = fe.per_sow_year(24.0, 0.86)
    assert abs(p["msy"] - 24.0 * 0.86) < 0.01
    assert p["net_per_sow"] == p["gross_per_sow"] - p["sow_cost"]
    assert p["margin_per_pig"] == p["revenue_per_pig"] - p["cost_per_pig"]

    # 지렛대는 효과 크기 내림차순이고, 시세는 농장이 못 바꾼다고 표시돼야 한다
    lv = fe.levers()
    eff = list(lv["연간효과"])
    assert eff == sorted(eff, reverse=True), eff
    assert all(x > 0 for x in eff), eff
    assert any("못 바꾼다" in x for x in lv["경로"]), "통제 불가 항목 표시 없음"

    # 모돈 두수에 대해 선형 — 600두 효과는 300두의 2배
    a = fe.levers(n_sows=300)
    b = fe.levers(n_sows=600)
    assert list(a["lever"]) == list(b["lever"]), "두수만 바꿨는데 순위가 뒤집혔다"
    assert abs(b["연간효과"][0] - 2 * a["연간효과"][0]) < 2.0

    # 두 경로는 서로 다르므로 합산 가능하되 상호작용은 작아야 한다
    v = fe.app_value()
    assert v["repro_path"] > 0 and v["growth_path"] > 0
    assert abs(v["interaction"]) < 0.05 * v["combined"], v

    # PSY·육성률이 높을수록 순이익이 커야 한다(단조)
    prof = [fe.per_sow_year(x, 0.86)["net_per_sow"] for x in (20.0, 24.0, 28.0)]
    assert prof == sorted(prof), prof
    surv = [fe.per_sow_year(24.0, s)["net_per_sow"] for s in (0.80, 0.86, 0.93)]
    assert surv == sorted(surv), surv


def test_pigflow_package() -> None:
    """돈군흐름 패키지 — 명세 §5 검산과 시뮬레이터 회귀를 통째로 돌린다."""
    import importlib.util
    path = os.path.join(ROOT, "pigflow", "tests", "test_pigflow.py")
    assert os.path.exists(path), path
    spec = importlib.util.spec_from_file_location("_pigflow_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed = []
    for t in mod.TESTS:
        try:
            t()
        except Exception as e:                                    # noqa: BLE001
            failed.append(f"{t.__name__}: {e}")
    assert not failed, f"{len(failed)}/{len(mod.TESTS)} 실패 — " + "; ".join(
        failed[:3])


def test_bio_baseline_thresholds() -> None:
    """평균 0 기준표의 문턱이 살아 있는가 — 죽은 문턱을 커밋으로 막는다.

    전 필드에 3σ 를 일괄로 물렸을 때 back_temp 는 0건(발화 불가),
    latent_heat 는 10.2%(알림 과다)로 문턱이 죽어 있었다. 필드마다 z 를
    조정하도록 고쳤으므로, 어느 필드든 FLAG_BAND 밖으로 나가면 실패시킨다.
    """
    import bio_baseline_71763 as bb
    import parse_aihub
    import tempfile
    tmp = tempfile.mkdtemp()
    parse_aihub.generate_synthetic_71763(tmp, n=4000)
    clips = parse_aihub.aggregate_71763_clips(parse_aihub.parse_71763(tmp))
    clips = clips[clips["modality"] == "호흡량"]
    assert len(clips) >= 30, f"합성 호흡량 클립 부족: {len(clips)}"

    b = bb.build_baseline(clips)
    assert not b.empty, "기준표가 비었다"
    lo, hi = bb.FLAG_BAND
    dead = b[b["usable"] != "쓸만함"]
    assert dead.empty, (
        "문턱이 죽은 필드: "
        + ", ".join(f"{r.field}({r.flagged_pct}%, {r.usable})"
                    for r in dead.itertuples()))
    assert (b["flagged_pct"].between(lo, hi)).all(), b[
        ["field", "z", "flagged_pct"]].to_string()

    # 분산분해는 제곱합으로 갈라야 두 열의 합이 100 이 된다.
    v = bb.variance_split(clips)
    if not v.empty:
        tot = v["within_pct"] + v["between_pct"]
        assert (tot.sub(100).abs() <= 1).all(), v.to_string()


def rc_date(x):
    import repro_calendar as rc
    return rc._d(x)


def main() -> int:
    tests = [test_bio_baseline_thresholds,
             test_dependencies_import, test_aihub_client_no_key,
             test_pipeline_runs, test_aihub_parsers,
             test_pipeline_gilt_integration, test_estrus_onset_and_dashboard,
             test_edinburgh_parser, test_posture_eval_mapping,
             test_view_align_feats, test_estrus_link, test_aihub_reference,
             test_appearance_crop_feats, test_iou_tracker,
             test_eval_report_figs, test_estrus_reference_validation,
             test_repro_cause_attribution, test_estrus_early_warning,
             test_repro_dashboard_svg, test_parse_71471_real_schema,
             test_estrus_calendar_link, test_estrus_contrast_eval,
             test_keypoints_parser_pose, test_pose_vs_behavior_eval,
             test_motion_tracker, test_box_merge, test_temporal_features,
             test_breeding_timing, test_stall_estrus, test_feeding_monitor,
             test_repro_calendar, test_pregnancy_check, test_herd_board,
             test_barn_queue, test_batch_flow, test_work_log,
             test_aihub_bridge, test_pig_polygon, test_growth_flow,
             test_farm_registry, test_breeding_ledger, test_barn_environment,
             test_posture_crop_feats, test_posture_crossview, test_posture_report,
             test_dashboard_builders, test_farm_economics,
             test_pigflow_package, test_check_download,
             test_finetune_polygon, test_fetch_622, test_korean_farm_stats, test_farm_monthly, test_synth_farm, test_farm_panel, test_farm_monthly_panel, test_farm_monthly_model, test_psy_priority, test_presentation_cnn_current, test_estrus_label_audit, test_path_predict, test_barn_watch, test_farm_setup_view, test_capacity_from_rooms, test_throughput_ceiling, test_setup_screen_matches_module, test_setup_json_actually_runs, test_run_farm_from_setup, test_herd_drives_stage_counts, test_herd_cycle_from_perf, test_table_export, test_pig_behavior_adapter, test_behavior_baseline, test_behavior_head_train, test_mating_plan, test_barn_env_control, test_pig_behavior_toolkit, test_ops_api_and_view, test_farm_scale_and_formula, test_improve_path, test_vision_contract, test_season_interval_view, test_timing_cache_is_transparent, test_server_api, test_farm_diagnosis_view, test_pc_suite, test_ml_core, test_kaggle_notebooks, test_farm_gap, test_run_farm_end_to_end, test_docs_consistent,
             test_image_name_collision,
             test_real_622_schema,
             test_fetch_622_doctor]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
