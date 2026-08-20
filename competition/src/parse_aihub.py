"""AI Hub 라벨 → 분석용 정형 테이블(train.csv) 파서.

실제 데이터 확보 전, AI Hub 공개 문서(데이터 view 페이지의 '어노테이션 포맷 및
데이터 구조')에서 확인한 스키마에 맞춰 미리 작성했다. 각 파서는 실제 라벨
디렉터리를 받아 pandas DataFrame 을 돌려준다.

세 데이터셋:
  71763  양돈 생체 에너지 데이터(2023)  — JSON, 정형 회귀에 최적
  71471  소·돼지 발정행동 데이터        — JSON, 행동분류/발정탐지
  622    지능형 스마트축사(양돈)         — XML(CVAT), 탐지/자세

각 파서에는 '문서 스키마대로' 합성 샘플을 만드는 generate_synthetic_* 가 딸려
있어, 실데이터 없이도 파싱 로직을 검증할 수 있다(--selftest).

주의: 실제 필드명이 문서와 미세하게 다를 수 있다. 파서는 방어적으로 접근하며
(get/기본값), 실데이터가 오면 이 스키마와 대조해 필요한 곳만 수정한다.
"""
from __future__ import annotations

import json
import os
import random
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# 71763 양돈 생체 에너지 데이터 (JSON) — 실라벨로 스키마 확정 (2026-08-19)
#
# 문서(SCHEMA.md)는 TextInfo/SensorData/TemperatureData/FeedingAndManagementData
# 가 annotations **안에** 있다고 적어 두었으나, 실제로는 annotations 의
# **형제**다. breath-rate·sensibleHeat·latentHeat·pig-manure 는 어느 블록에도
# 속하지 않는 최상위 필드다. 문서의 IatentHeat(대문자 I)는 latentHeat 이고
# feedstuff-volume 은 feedstuff_volume(밑줄)이다.
#
#   ImageInfo{video-category, videoid, chamber-number, pig-classification,
#             pig-number, breathing-type, date, time, timestamp, Floormaterial}
#   annotations{keypoint-top:[[x,y],[x,y]], pointcount, distance,
#               available-area-bbox:[x,y,x,y], bbox:[x,y,w,h]}
#   TextInfo{chamber-number, pig-number, weight, pig-classification,
#            measure-date, measure-time}
#   SensorData{T, RH, CO2, NH3}
#   TemperatureData{rectal-, back-, neck-, head-temperature}
#   FeedingAndManagementData{ventilation-rate, feedstuff_volume, watersupply}
#   breath-rate | evaporation | pig-manure | sensibleHeat | latentHeat  ← 최상위
#
# 모달리티가 둘이고, 문서에는 앞의 하나만 적혀 있었다.
#   호흡량 video-category="pig"   : 개체 있음(pig-number). 타깃 breath-rate.
#   증발량 video-category="floor" : 개체 **없음**. 타깃 evaporation. bbox 계열.
#   둘 다 sensibleHeat/latentHeat 를 갖는다.
#
# 분석 단위는 프레임이 아니라 **클립**(폴더 하나 = 측정 세션 하나)이다.
# 한 클립 안에서 타깃·환경·개체 값이 전부 상수이고 프레임마다 변하는 것은
# distance 와 breathing-type 뿐이다. 프레임 행으로 회귀를 돌리면 같은 답을
# 수십 번 세는 셈이라 성적이 부풀려진다 → aggregate_71763_clips() 로 접는다.
# ---------------------------------------------------------------------------
PIG_CLASSES_71763 = ["weaningpig", "piglet", "growing-pig", "porker"]
SPLITS_71763 = ("TL", "VL_A", "VL_B", "VL")   # VL 보다 VL_A/VL_B 를 먼저 본다

#: 클립 안에서 상수인 필드(= 클립 단위 정보). 나머지는 프레임 단위다.
CLIP_CONST_71763 = [
    "video_category", "videoid", "pig_class", "pig_number", "floor_material",
    "date", "time", "weight_kg", "measure_date", "measure_time",
    "temp_c", "humidity_pct", "co2_ppm", "nh3_ppm",
    "rectal_temp", "back_temp", "neck_temp", "head_temp",
    "ventilation", "feed_volume", "water_supply", "manure",
    "breath_rate", "evaporation", "sensible_heat", "latent_heat",
]
#: 프레임마다 변하는 필드 — 클립 단위로 접을 때 요약 통계를 만든다.
CLIP_VARY_71763 = ["keypoint_distance", "keypoint_distance_calc",
                   "bbox_area", "area_ratio"]

TARGETS_71763 = ["breath_rate", "sensible_heat", "latent_heat", "evaporation"]


def _num(d: dict, key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return v if v is not None else default


def _split_of(path: str) -> Any:
    parts = set(path.replace("\\", "/").split("/"))
    for s in SPLITS_71763:
        if s in parts:
            return s
    return None


def _kp_distance(kp: Any) -> Any:
    """keypoint-top 두 점 사이 거리를 직접 계산한다(라벨 distance 검증용)."""
    if not isinstance(kp, list) or len(kp) < 2:
        return None
    try:
        x1, y1 = float(kp[0][0]), float(kp[0][1])
        x2, y2 = float(kp[1][0]), float(kp[1][1])
    except (TypeError, ValueError, IndexError):
        return None
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def parse_71763(label_dir: str) -> pd.DataFrame:
    """71763 라벨 JSON 디렉터리 → **프레임** 단위 정형 테이블.

    회귀에 그대로 넣으면 안 된다. 클립 안에서 타깃이 상수이므로
    aggregate_71763_clips() 로 접은 뒤 모델에 넣는다.
    """
    rows: list[dict] = []
    for path in _iter_json(label_dir):
        try:
            obj = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        # 예전 스키마(모든 블록이 annotations 안)도 계속 받아 준다.
        ann = obj.get("annotations") or {}
        src = obj if ("TextInfo" in obj or "ImageInfo" in obj) else ann
        if not isinstance(ann, dict):
            ann = {}
        ii = src.get("ImageInfo", {}) or {}
        ti = src.get("TextInfo", {}) or {}
        sd = src.get("SensorData", {}) or {}
        td = src.get("TemperatureData", {}) or {}
        fm = src.get("FeedingAndManagementData", {}) or {}

        cat = ii.get("video-category")
        bbox = ann.get("bbox") or []
        avail = ann.get("available-area-bbox") or []
        bw = _to_float(bbox[2]) if len(bbox) >= 4 else None
        bh = _to_float(bbox[3]) if len(bbox) >= 4 else None
        b_area = bw * bh if bw and bh else None
        a_area = None
        if len(avail) >= 4:
            ax, ay, bx, by = (_to_float(v) for v in avail[:4])
            if None not in (ax, ay, bx, by):
                a_area = abs(bx - ax) * abs(by - ay)

        rows.append({
            # ── 식별자 (클립 = 파일이 들어 있는 폴더) ─────────────────────
            "split": _split_of(path),
            "modality": {"pig": "호흡량", "floor": "증발량"}.get(cat, cat),
            "clip_id": os.path.basename(os.path.dirname(path)),
            "frame_file": os.path.basename(path),
            "timestamp": ii.get("timestamp"),
            # ── ImageInfo ────────────────────────────────────────────────
            "video_category": cat,
            "videoid": _to_int(ii.get("videoid")),
            "breathing_type": ii.get("breathing-type"),
            "floor_material": ii.get("Floormaterial"),
            "date": ii.get("date") or ti.get("measure-date"),
            "time": ii.get("time") or ti.get("measure-time"),
            # ── TextInfo (chamber/개체는 ImageInfo 를 보조로 쓴다) ────────
            "chamber": _to_int(_num(ti, "chamber-number", ii.get("chamber-number"))),
            "pig_number": _to_int(_num(ti, "pig-number", ii.get("pig-number"))),
            "pig_class": ti.get("pig-classification") or ii.get("pig-classification"),
            "weight_kg": _to_float(ti.get("weight")),
            "measure_date": ti.get("measure-date"),
            "measure_time": ti.get("measure-time"),
            # ── 환경 센서 ────────────────────────────────────────────────
            "temp_c": _to_float(sd.get("T")),
            "humidity_pct": _to_float(sd.get("RH")),
            "co2_ppm": _to_float(sd.get("CO2")),
            "nh3_ppm": _to_float(sd.get("NH3")),
            # ── 개체 부위 온도 ───────────────────────────────────────────
            "rectal_temp": _to_float(td.get("rectal-temperature")),
            "back_temp": _to_float(td.get("back-temperature")),
            "neck_temp": _to_float(td.get("neck-temperature")),
            "head_temp": _to_float(td.get("head-temperature")),
            # ── 사양관리 ─────────────────────────────────────────────────
            "ventilation": _to_float(fm.get("ventilation-rate")),
            "feed_volume": _to_float(_num(fm, "feedstuff_volume",
                                          fm.get("feedstuff-volume"))),
            "water_supply": _to_float(fm.get("watersupply")),
            # ── 최상위 (문서가 블록 안이라고 적어 둔 것들) ───────────────
            "manure": _to_float(_num(src, "pig-manure", fm.get("pig-manure"))),
            "breath_rate": _to_float(_num(src, "breath-rate",
                                          sd.get("breath-rate"))),
            "evaporation": _to_float(src.get("evaporation")),
            "sensible_heat": _to_float(_num(src, "sensibleHeat",
                                            fm.get("sensibleHeat"))),
            "latent_heat": _to_float(_num(src, "latentHeat",
                                          _num(fm, "latentHeat",
                                               fm.get("IatentHeat")))),
            # ── annotations (프레임 단위) ────────────────────────────────
            "pointcount": _to_int(ann.get("pointcount")),
            "keypoint_distance": _to_float(ann.get("distance")),
            "keypoint_distance_calc": _kp_distance(ann.get("keypoint-top")),
            "bbox_w": bw,
            "bbox_h": bh,
            "bbox_area": b_area,
            "area_ratio": (b_area / a_area) if b_area and a_area else None,
        })
    return pd.DataFrame(rows)


def aggregate_71763_clips(df: pd.DataFrame) -> pd.DataFrame:
    """프레임 테이블 → **클립** 단위 테이블(회귀에 쓸 정직한 단위).

    클립 내 상수 필드는 첫 값을 쓰되 실제로 상수였는지 `<필드>_nuniq` 로
    남긴다(2 이상이면 상수 가정이 깨진 것이므로 반드시 확인해야 한다).
    프레임마다 변하는 필드는 평균·표준편차·최소·최대로 접는다.
    """
    if df.empty:
        return df
    keys = [k for k in ("split", "modality", "chamber", "clip_id")
            if k in df.columns]
    out: list[dict] = []
    for key, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n_frames"] = len(g)
        for c in CLIP_CONST_71763:
            if c in g.columns and c not in keys:
                v = g[c].dropna()
                row[c] = v.iloc[0] if len(v) else None
                row[c + "_nuniq"] = int(v.nunique())
        for c in CLIP_VARY_71763:
            if c in g.columns:
                v = pd.to_numeric(g[c], errors="coerce").dropna()
                row[c + "_mean"] = v.mean() if len(v) else None
                row[c + "_std"] = v.std() if len(v) > 1 else None
                row[c + "_min"] = v.min() if len(v) else None
                row[c + "_max"] = v.max() if len(v) else None
        if "breathing_type" in g.columns:
            bt = g["breathing_type"].dropna()
            row["insp_ratio"] = (bt == "inspiratory").mean() if len(bt) else None
        out.append(row)
    return pd.DataFrame(out)


def group_key_71763(df: pd.DataFrame) -> pd.Series:
    """누수 방지용 그룹 키.

    호흡량은 개체가 있지만 pig-number 가 **챔버마다 다시 매겨진다**(고유
    pig-number 71개 중 13개가 둘 이상의 챔버에 나타난다). 따라서 개체는
    (chamber, pig_number) 조합이어야 한다. pig_class 는 **키에 넣지 않는다** —
    한 개체가 자라면서 weaningpig→piglet 으로 바뀌는 사례가 있어(chamber4
    pig5), 넣으면 같은 개체가 두 그룹으로 쪼개져 오히려 누수가 된다.

    증발량에는 개체가 없으므로 (chamber, date) 로 묶는다 — 같은 날 같은
    챔버의 측정은 환경이 사실상 같아서 서로 다른 폴드로 흩어지면 누수가 된다.
    """
    def key(r):
        if r.get("modality") == "증발량" or pd.isna(r.get("pig_number")):
            return "floor|%s|%s" % (r.get("chamber"), r.get("date"))
        return "pig|%s|%s" % (r.get("chamber"), r.get("pig_number"))
    return df.apply(key, axis=1)


def generate_synthetic_71763(out_dir: str, n: int = 200) -> None:
    """**실라벨 스키마대로** 71763 합성 라벨을 만든다(파서 검증용).

    실데이터처럼 클립 폴더 구조를 만들고, 클립 안에서 타깃·환경은 상수로 두고
    distance·breathing-type 만 프레임마다 바꾼다. 두 모달리티를 모두 낸다.
    """
    rng = random.Random(71763)
    n_clips = max(2, n // 20)
    for c in range(n_clips):
        floor = (c % 2 == 1)
        chamber = rng.randint(1, 4)
        T = round(rng.uniform(15, 32), 1)
        RH = round(rng.uniform(20, 90), 1)
        weight = round(rng.uniform(8, 120), 1)
        breath = round(20 + (T - 20) * 2.2 + (RH - 60) * 0.1 + rng.gauss(0, 3))
        sensible = round(weight * 1.1 + (T - 20) * 0.5 + rng.gauss(0, 5), 2)
        latent = round(weight * 0.7 + (RH - 60) * 0.3 + rng.gauss(0, 4), 2)
        pig_class = rng.choice(PIG_CLASSES_71763)
        date = "22%02d%02d" % (rng.randint(9, 12), rng.randint(1, 28))
        tm = "%02d%02d" % (rng.randint(0, 23), rng.randint(0, 59))
        cat = "floor" if floor else "pig"
        pig_no = None if floor else rng.randint(1, 99)
        clip = ("chamber%d_%s_%s_%s%s_%s00_%03d"
                % (chamber, cat, pig_class,
                   "" if floor else ("pig%d_" % pig_no), date, tm, c))
        d = os.path.join(out_dir, "증발량 이미지" if floor else "호흡량 이미지",
                         "chamber%d" % chamber, clip)
        os.makedirs(d, exist_ok=True)
        for f in range(20):
            ii = {"video-category": cat, "videoid": c,
                  "chamber-number": chamber, "pig-classification": pig_class,
                  "date": date, "time": tm, "timestamp": "%05d" % f}
            if floor:
                ii["Floormaterial"] = rng.choice(["concrete", "diatomite"])
                ann = {"available-area-bbox": [460, 210, 1090, 870],
                       "bbox": [rng.randint(600, 700), rng.randint(200, 240),
                                rng.randint(500, 600), rng.randint(800, 900)]}
            else:
                ii["pig-number"] = pig_no
                ii["breathing-type"] = rng.choice(["inspiratory", "expiratory"])
                x1, y1 = rng.randint(400, 550), rng.randint(600, 720)
                x2, y2 = rng.randint(900, 1000), rng.randint(280, 330)
                ann = {"keypoint-top": [[x1, y1], [x2, y2]], "pointcount": 2,
                       "distance": round(((x1 - x2) ** 2
                                          + (y1 - y2) ** 2) ** 0.5, 3)}
            ti = {"chamber-number": chamber, "pig-classification": pig_class,
                  "measure-date": date, "measure-time": tm}
            if not floor:
                ti["pig-number"] = pig_no
                ti["weight"] = weight
            obj = {
                "ImageInfo": ii,
                "annotations": ann,
                "TextInfo": ti,
                "SensorData": {"T": T, "RH": RH,
                               "CO2": round(rng.uniform(400, 3000), 1),
                               "NH3": round(rng.uniform(1, 40), 1)},
                "FeedingAndManagementData": {
                    "ventilation-rate": round(rng.uniform(1, 3), 2),
                    "feedstuff_volume": round(rng.uniform(0.3, 3.5), 3),
                    "watersupply": round(rng.uniform(1, 10), 1)},
                "pig-manure": round(rng.uniform(400, 1400), 1),
                "sensibleHeat": sensible,
                "latentHeat": latent,
            }
            if floor:
                obj["evaporation"] = round(rng.uniform(0.5, 40), 2)
            else:
                obj["breath-rate"] = breath
                obj["TemperatureData"] = {
                    "rectal-temperature": round(rng.uniform(35.5, 40.0), 1),
                    "back-temperature": round(rng.uniform(30, 38), 1),
                    "neck-temperature": round(rng.uniform(30, 38), 1),
                    "head-temperature": round(rng.uniform(30, 38), 1)}
            json.dump(obj, open(os.path.join(d, "%s_%05d.json" % (clip, f)),
                                "w", encoding="utf-8"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 71471 소·돼지 발정행동 데이터 (JSON)
#   행동분류: standing/lying/eating/head shaking/tailing/sitting ...
#   발정여부(estrus): 발정체크장비 + 전문가 검수
#   Bounding Box / Keypoints / Polygon 어노테이션
# ---------------------------------------------------------------------------
BEHAVIORS_71471 = ["standing", "lying", "eating", "head shaking",
                   "tailing", "sitting"]
SPECIES_71471 = ["pig", "blackpig"]


def parse_71471(label_dir: str) -> pd.DataFrame:
    """71471 라벨 JSON → 개체·프레임(키프레임) 단위 테이블.

    발정 판단은 시간에 걸친 행동/활동 변화가 핵심이므로, 프레임 단위 행을
    개체(individual_id)·프레임(frame_idx) 식별자와 함께 반환한다. keypoint 는
    프레임 간 이동(활동량) 계산을 위해 중심좌표(centroid)와 산포로 축약한다.
    상위 피처 집계는 build_estrus_features() (model_71471_estrus.py) 가 담당.
    """
    rows: list[dict] = []
    for path in _iter_json(label_dir):
        try:
            obj = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        meta = obj.get("metadata", obj.get("info", {}))
        species = obj.get("species") or meta.get("species")
        indiv = (obj.get("individual_id") or meta.get("individual_id")
                 or _id_from_name(os.path.basename(path)))
        frame = (obj.get("frame_idx") if obj.get("frame_idx") is not None
                 else meta.get("frame_idx"))
        estrus = _to_int(obj.get("estrus", meta.get("estrus")))
        for a in _annotations_list(obj):
            bbox = a.get("bbox") or a.get("bounding_box") or []
            kps = a.get("keypoints") or a.get("keypoint") or []
            w = h = None
            if len(bbox) == 4:
                x1, y1, x3, y4 = bbox
                w = abs(x3) if (x3 < x1 or x3 < 5) else abs(x3 - x1)
                h = abs(y4) if (y4 < y1 or y4 < 5) else abs(y4 - y1)
            cx, cy, spread = _keypoint_summary(kps)
            rows.append({
                "file": os.path.basename(path),
                "individual_id": indiv,
                "frame_idx": frame,
                "species": species,
                "behavior": a.get("behavior") or a.get("action")
                            or a.get("category"),
                "estrus": estrus,
                "bbox_w": w, "bbox_h": h,
                "aspect_ratio": (w / h) if (w and h) else None,
                "centroid_x": cx, "centroid_y": cy, "kp_spread": spread,
            })
    return pd.DataFrame(rows)


def _keypoint_summary(kps: list) -> tuple:
    """[x1,y1,x2,y2,...] 평면 리스트 → (중심x, 중심y, 산포)."""
    if not kps:
        return (None, None, None)
    flat = kps
    if isinstance(kps[0], (list, tuple)):  # [[x,y],...] 형태
        flat = [v for xy in kps for v in xy[:2]]
    xs = flat[0::2]
    ys = flat[1::2]
    if not xs or not ys:
        return (None, None, None)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    spread = (sum((x - cx) ** 2 for x in xs) / len(xs)) ** 0.5
    return (round(cx, 2), round(cy, 2), round(spread, 2))


def _id_from_name(name: str):
    """파일명에서 개체 식별자 추출(발정시간·채널·프레임 인코딩 규칙 대응).

    실데이터 파일명 규칙 확인 후 이 함수만 조정하면 된다. 기본은 마지막
    '_f### / frame' 앞부분을 개체 키로 본다.
    """
    base = name.rsplit(".", 1)[0]
    for tok in ("_f", "_frame", "_F"):
        if tok in base:
            return base.split(tok)[0]
    return base


def generate_synthetic_71471(out_dir: str, n_individuals: int = 80,
                             frames: int = 20) -> None:
    """개체×프레임 구조의 합성 라벨 생성.

    발정 개체는 (a) 활동량↑(프레임 간 중심 이동 큼), (b) standing/tailing 잦음,
    (c) lying 적음 — 이라는 도메인 신호를 심어, 시계열 집계 피처가 학습되도록 한다.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(71471)
    farms = ["pigfarmA", "blackpigB"]
    for ind in range(n_individuals):
        estrus = 1 if rng.random() < 0.4 else 0
        species = rng.choice(SPECIES_71471)
        farm = rng.choice(farms)
        base_x, base_y = rng.uniform(200, 800), rng.uniform(200, 600)
        # 활동량: 발정이 평균적으로 높지만 개체차로 범위가 겹치게(현실적)
        move = (rng.uniform(10, 28) if estrus else rng.uniform(5, 20)) \
            + rng.gauss(0, 3)
        move = max(2.0, move)
        # 행동 분포도 경향만 다르게(완전 분리 아님)
        if estrus:
            probs = {"standing": .24, "tailing": .15, "eating": .15,
                     "head shaking": .10, "sitting": .11, "lying": .25}
        else:
            probs = {"lying": .34, "eating": .20, "standing": .15,
                     "sitting": .13, "head shaking": .08, "tailing": .10}
        behs = list(probs); wts = list(probs.values())
        for f in range(frames):
            behavior = rng.choices(behs, weights=wts)[0]
            cx = base_x + rng.gauss(0, move)
            cy = base_y + rng.gauss(0, move)
            kps = []
            for _ in range(17):  # 17 keypoint
                kps += [round(cx + rng.gauss(0, 25), 1),
                        round(cy + rng.gauss(0, 25), 1)]
            w = rng.uniform(120, 300); h = rng.uniform(70, 180)
            obj = {
                "metadata": {"species": species, "farm": farm,
                             "individual_id": f"{farm}_{ind:03d}",
                             "frame_idx": f},
                "estrus": estrus,
                "annotations": [{
                    "behavior": behavior,
                    "bbox": [round(cx - w / 2, 1), round(cy - h / 2, 1),
                             round(w, 1), round(h, 1)],
                    "keypoints": kps}],
            }
            fn = f"{farm}_{ind:03d}_ch1_f{f:03d}.json"
            json.dump(obj, open(os.path.join(out_dir, fn), "w",
                                encoding="utf-8"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 622 지능형 스마트축사(양돈) — CVAT XML (annotations.xml)
#   <image name width height>
#     <box label xtl ytl xbr ybr/>
#     <polygon label points="x,y;x,y;..."/>
#     <points label points="x,y;..."/>
#   라벨 = 월령/상태 (이유자돈 전기/후기, 육성돈, 비육돈, 분만돈, 임신돈, 환돈)
# ---------------------------------------------------------------------------
STAGES_622 = ["이유자돈전기", "이유자돈후기", "육성돈전기", "육성돈후기",
              "비육돈전기", "비육돈후기", "분만돈", "임신돈", "환돈"]


def parse_622(label_dir: str) -> pd.DataFrame:
    """622 CVAT XML → 객체 단위 테이블(이미지·라벨·형상)."""
    rows: list[dict] = []
    for path in _iter_files(label_dir, ".xml"):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for img in root.iter("image"):
            iw = _to_float(img.get("width"))
            ih = _to_float(img.get("height"))
            name = img.get("name")
            for box in img.findall("box"):
                xtl, ytl = _to_float(box.get("xtl")), _to_float(box.get("ytl"))
                xbr, ybr = _to_float(box.get("xbr")), _to_float(box.get("ybr"))
                rows.append({
                    "file": os.path.basename(path), "image": name,
                    "img_w": iw, "img_h": ih, "shape": "box",
                    "label": box.get("label"),
                    "obj_w": (xbr - xtl) if (xbr and xtl) else None,
                    "obj_h": (ybr - ytl) if (ybr and ytl) else None,
                })
            for poly in list(img.findall("polygon")) + list(img.findall("points")):
                pts = _parse_points(poly.get("points", ""))
                rows.append({
                    "file": os.path.basename(path), "image": name,
                    "img_w": iw, "img_h": ih,
                    "shape": poly.tag, "label": poly.get("label"),
                    "num_points": len(pts),
                })
    return pd.DataFrame(rows)


def generate_synthetic_622(out_dir: str, n_images: int = 60) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(622)
    root = ET.Element("annotations")
    for i in range(n_images):
        img = ET.SubElement(root, "image", id=str(i), name=f"img_{i:04d}.jpg",
                            width="1920", height="1080")
        for _ in range(rng.randint(1, 6)):
            label = rng.choice(STAGES_622)
            xtl, ytl = rng.uniform(0, 1500), rng.uniform(0, 800)
            ET.SubElement(img, "box", label=label,
                          xtl=f"{xtl:.1f}", ytl=f"{ytl:.1f}",
                          xbr=f"{xtl + rng.uniform(50, 300):.1f}",
                          ybr=f"{ytl + rng.uniform(50, 250):.1f}")
            if rng.random() < 0.5:
                pts = ";".join(f"{rng.uniform(0,1920):.1f},{rng.uniform(0,1080):.1f}"
                               for _ in range(rng.randint(4, 10)))
                ET.SubElement(img, "polygon", label=label, points=pts)
    ET.ElementTree(root).write(os.path.join(out_dir, "annotations.xml"),
                               encoding="utf-8", xml_declaration=True)


# --------------------------- 공용 유틸 ---------------------------
def _iter_files(root: str, ext: str):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(ext):
                yield os.path.join(dp, f)


def _iter_json(root: str):
    return _iter_files(root, ".json")


def _annotations_list(obj: dict) -> list[dict]:
    a = obj.get("annotations", [])
    if isinstance(a, dict):
        return [a]
    return a if isinstance(a, list) else []


def _parse_points(s: str) -> list[tuple[float, float]]:
    out = []
    for pair in s.split(";"):
        if "," in pair:
            x, y = pair.split(",")[:2]
            out.append((_to_float(x), _to_float(y)))
    return out


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------- CLI / self-test ---------------------------
PARSERS = {"71763": parse_71763, "71471": parse_71471, "622": parse_622}
GENERATORS = {"71763": generate_synthetic_71763,
              "71471": generate_synthetic_71471,
              "622": generate_synthetic_622}


def selftest() -> int:
    import tempfile
    ok = True
    for key in ("71763", "71471", "622"):
        with tempfile.TemporaryDirectory() as d:
            GENERATORS[key](d)
            df = PARSERS[key](d)
            n = len(df)
            print(f"[{key}] 합성→파싱 행 {n}, 열 {df.shape[1]}: "
                  f"{list(df.columns)[:6]}...")
            if n == 0:
                ok = False
                print(f"  ‼ {key} 파싱 결과가 비었습니다.")
            else:
                nonnull = df.notna().mean().mean()
                print(f"  평균 비결측 비율 {nonnull:.0%}")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "--selftest":
        return selftest()
    if len(argv) < 2:
        print("사용: python parse_aihub.py <71763|71471|622> <label_dir> "
              "[out.csv]  |  --selftest")
        return 1
    key, label_dir = argv[0], argv[1]
    if key not in PARSERS:
        print(f"알 수 없는 데이터셋: {key}")
        return 1
    df = PARSERS[key](label_dir)
    out = argv[2] if len(argv) >= 3 else f"competition/data/{key}_parsed.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[{key}] 파싱 완료: {len(df)}행 → {out}")
    print(df.head())
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
