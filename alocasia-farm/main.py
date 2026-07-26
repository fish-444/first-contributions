"""
알로카시아 스마트팜 3D 온실 - FastAPI 백엔드
=================================================

핵심 파이프라인만:
  [웹 UI] 사진 + 식물 이름 + 자리(A1~E10) 선택
      │  POST /api/plants  (name, file, pos)
      ▼
  [FastAPI] YOLO 모델 2개로 분석 →
      · 모델1 맨 위 잎(광합성) → top_leaf_size  → 3D 화분/식물 크기
      · 모델2 잎 단계          → shoot/mature/old → 3D 잎 색 + 모달 수치
      · 겹침 밀도(overlap), 크기 분류(대/중/소품)
      → 선택한 자리에 식물로 등록/저장
      ▼
  [3D 온실] 화분으로 시각화, 클릭 시 이름+분석결과 모달

센서·RAG 없음. 오직 [사진+이름+자리 → YOLO → 3D 반영 + 이름/방향 커스텀].

분석 엔진 자동 선택:
  1) 로보플로우(ROBOFLOW_API_KEY 있으면)  2) 로컬 ultralytics(.pt)  3) 데모(모델 없어도 동작)
실행법은 README.md 참고.
"""

import base64
import io
import math
import os
import time
import uuid
import random
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# --------------------------------------------------------------------------- 설정
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "find-leaf-and-object/1")
# 두 모델 (키 하나로 둘 다 호출)
ROBOFLOW_MODEL_TOP = os.environ.get("ROBOFLOW_MODEL_TOP", ROBOFLOW_MODEL_ID)      # 모델1: 맨 위 잎(광합성) → 3D
ROBOFLOW_MODEL_STAGE = os.environ.get("ROBOFLOW_MODEL_STAGE", ROBOFLOW_MODEL_ID)  # 모델2: 새순/성숙/노령 → 모달
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
# 워크플로(Workflow) 방식 — 모델 대신 로보플로우에서 구성한 파이프라인을 통째로 호출
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID", "")
ROBOFLOW_WORKFLOW_URL = os.environ.get("ROBOFLOW_WORKFLOW_URL", "")   # 전체 URL 직접 지정(선택)
WORKFLOW_IMAGE_INPUT = os.environ.get("ROBOFLOW_WORKFLOW_IMAGE_INPUT", "image")
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
CONFIDENCE = float(os.environ.get("CONFIDENCE", "25"))

_HAS_WORKFLOW = bool(ROBOFLOW_WORKFLOW_URL or (ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW_ID))
if ROBOFLOW_API_KEY and _HAS_WORKFLOW:
    ENGINE = "workflow"
elif ROBOFLOW_API_KEY:
    ENGINE = "roboflow"
elif os.path.exists(MODEL_PATH) or os.environ.get("USE_LOCAL"):
    ENGINE = "local"
else:
    ENGINE = "demo"

local_model = None
if ENGINE == "local":
    try:
        from ultralytics import YOLO
        print(f"[엔진=로컬] 모델 로딩: {MODEL_PATH}")
        local_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[경고] 로컬 모델 로딩 실패({e}) → 데모 모드")
        ENGINE = "demo"

# UI 에 표시할 이름 (어느 워크플로가 도는지 눈으로 확인) — 엔진 확정 후에 계산
ENGINE_LABEL = f"workflow · {ROBOFLOW_WORKFLOW_ID}" if ENGINE == "workflow" and ROBOFLOW_WORKFLOW_ID else ENGINE
print(f"[분석 엔진] {ENGINE_LABEL}")

app = FastAPI(title="Alocasia Smart Farm")

# --------------------------------------------------------------------------- 상태
PLANTS: Dict[str, dict] = {}          # id -> 식물 상태
FEATS: Dict[str, dict] = {}           # id -> 생김새 특징(모양 그룹 계산용, 응답에는 안 나감)
# 미리 지정한 화분 자리 — 모델에 pot 클래스를 라벨링하는 대신 쓴다.
# 화분은 안 움직이니 한 번만 찍어 두면 계속 재사용되고, 자리(슬롯)가 고정돼
# 매번 스캔해도 "이 화분 = 항상 같은 식물"이 유지된다. 선반 정규좌표(0~1).
POTS: List[dict] = []
# 번호 붙은 자리(슬롯): 온실 60x40cm 을 촘촘한 격자로 분할
# 기본 A1~E10 (5줄 x 10칸 = 50자리)
_ROWS = ["A", "B", "C", "D", "E"]
_COLS = 10
_W, _D = 60.0, 40.0
_CW, _CD = _W / _COLS, _D / len(_ROWS)      # 칸 크기 (6 x 8 cm)
SLOTS = [{"label": f"{r}{c + 1}",
          "x": round(-_W / 2 + _CW * (c + 0.5), 2),
          "z": round(-_D / 2 + _CD * (ri + 0.5), 2)}
         for ri, r in enumerate(_ROWS) for c in range(_COLS)]


# --------------------------------------------------------------------------- 저장
# 식물·화분 자리를 파일에 남긴다. 안 그러면 서버를 끌 때마다 손으로 고친 값과
# 새 잎 기록이 통째로 날아간다. FARM_DB="" 로 두면 저장하지 않는다(테스트용).
FARM_DB = os.environ.get("FARM_DB", "farm.db")


def _db():
    import sqlite3
    con = sqlite3.connect(FARM_DB)
    con.execute("CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)")
    return con


def save_state() -> None:
    """지금 상태를 저장. 바뀔 때마다 부른다 (개체 수가 적어 통째로 써도 싸다)."""
    if not FARM_DB:
        return
    import json
    try:
        with _db() as con:
            for key, val in (("plants", PLANTS), ("feats", FEATS), ("pots", POTS)):
                con.execute(
                    "INSERT INTO state(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(val, ensure_ascii=False)))
    except Exception as e:                      # 저장이 실패해도 서비스는 계속
        print(f"[경고] 저장 실패: {e}")


def load_state() -> None:
    """서버 시작 때 이전 상태를 되살린다."""
    if not FARM_DB or not os.path.exists(FARM_DB):
        return
    import json
    try:
        with _db() as con:
            rows = dict(con.execute("SELECT key, value FROM state").fetchall())
        PLANTS.update(json.loads(rows.get("plants", "{}")))
        FEATS.update(json.loads(rows.get("feats", "{}")))
        POTS.extend(json.loads(rows.get("pots", "[]")))
        if PLANTS or POTS:
            print(f"[저장소] 식물 {len(PLANTS)}개 · 화분자리 {len(POTS)}개 불러옴 ({FARM_DB})")
    except Exception as e:
        print(f"[경고] 불러오기 실패({e}) — 빈 상태로 시작합니다")


load_state()


def _slot_by_label(label: str):
    return next((s for s in SLOTS if s["label"] == label), None)


def _free_slot(prefer: str = None):
    used = {p.get("pos") for p in PLANTS.values()}
    if prefer:
        s = _slot_by_label(prefer)
        if s and s["label"] not in used:
            return s
    return next((s for s in SLOTS if s["label"] not in used), None)


# --------------------------------------------------------------------------- YOLO 탐지
def detect_boxes(image: Image.Image, model_id: str):
    if ENGINE == "workflow":
        return _detect_workflow(image)
    if ENGINE == "roboflow":
        return _detect_roboflow(image, model_id)
    if ENGINE == "local":
        return _detect_local(image)
    return _detect_demo(image)


def _jpeg_b64(image: Image.Image) -> str:
    buf = io.BytesIO(); image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _boxes_from_predictions(preds: List[dict]) -> List[dict]:
    """로보플로우 predictions(중심좌표 x,y + width,height) → 내부 박스 형식."""
    boxes = []
    for p in preds:
        try:
            w, h = float(p["width"]), float(p["height"])
            cx, cy = float(p["x"]), float(p["y"])
        except (KeyError, TypeError, ValueError):
            continue
        boxes.append({"cls": p.get("class") or p.get("class_name") or "leaf",
                      "conf": float(p.get("confidence", 0) or 0),
                      "x1": cx - w / 2, "y1": cy - h / 2,
                      "x2": cx + w / 2, "y2": cy + h / 2, "area": w * h})
    return boxes


def _detect_roboflow(image: Image.Image, model_id: str):
    import requests
    try:
        resp = requests.post(
            f"{ROBOFLOW_API_URL}/{model_id}",
            params={"api_key": ROBOFLOW_API_KEY, "confidence": CONFIDENCE},
            data=_jpeg_b64(image),
            headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=60,
        )
    except requests.RequestException:
        raise HTTPException(502, "로보플로우 서버 연결 실패 (인터넷 확인)")
    if resp.status_code in (401, 403):
        raise HTTPException(401, "로보플로우 개인(Private) API 키를 확인하세요.")
    if not resp.ok:
        raise HTTPException(502, f"로보플로우 오류: {resp.text[:150]}")
    return _boxes_from_predictions(resp.json().get("predictions", [])), image.width * image.height


# --------------------------------------------------------------------------- 워크플로
def _workflow_urls() -> List[str]:
    """호출할 워크플로 URL 후보. 로보플로우가 쓰는 두 경로 형식을 모두 시도한다."""
    if ROBOFLOW_WORKFLOW_URL:
        return [ROBOFLOW_WORKFLOW_URL]
    ws, wf = ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID
    return [f"{ROBOFLOW_API_URL}/infer/workflows/{ws}/{wf}",
            f"{ROBOFLOW_API_URL}/{ws}/workflows/{wf}"]


def _iter_prediction_lists(node):
    """워크플로 응답 어디에 박혀 있든 탐지 predictions 리스트를 전부 찾아낸다.

    출력 블록 이름(step 이름)이 워크플로마다 달라서 경로를 고정할 수 없다.
    그래서 응답 트리를 훑어 '박스처럼 생긴' predictions 리스트만 골라낸다.
    """
    if isinstance(node, dict):
        preds = node.get("predictions")
        if isinstance(preds, list) and any(
                isinstance(p, dict) and "x" in p and "width" in p for p in preds):
            yield preds
        for v in node.values():
            yield from _iter_prediction_lists(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_prediction_lists(v)


def _iter_image_dims(node):
    """응답에 실려 오는 이미지 크기({"image": {"width":…, "height":…}})를 찾는다."""
    if isinstance(node, dict):
        img = node.get("image")
        if isinstance(img, dict):
            w, h = img.get("width"), img.get("height")
            if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
                yield float(w) * float(h)
        for v in node.values():
            yield from _iter_image_dims(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_image_dims(v)


def _workflow_image_area(payload, fallback: float) -> float:
    """워크플로가 리사이즈를 포함하면 박스 좌표계가 업로드한 원본과 달라진다.
    응답이 알려 주는 이미지 크기를 우선 쓰고, 없으면 원본 크기로 되돌아간다."""
    return next(_iter_image_dims(payload), fallback)


def _extract_workflow_boxes(payload) -> List[dict]:
    """워크플로 응답 → 박스 목록. 중복 박스는 한 번만 센다."""
    boxes, seen = [], set()
    for preds in _iter_prediction_lists(payload):
        for b in _boxes_from_predictions(preds):
            key = (b["cls"], round(b["x1"], 1), round(b["y1"], 1),
                   round(b["x2"], 1), round(b["y2"], 1))
            if key in seen:
                continue
            seen.add(key)
            # 워크플로는 confidence 파라미터를 안 받는 경우가 있어 여기서 걸러 준다
            if b["conf"] and b["conf"] < CONFIDENCE / 100.0:
                continue
            boxes.append(b)
    return boxes


def _detect_workflow(image: Image.Image):
    import requests
    payload = {"api_key": ROBOFLOW_API_KEY,
               "inputs": {WORKFLOW_IMAGE_INPUT: {"type": "base64", "value": _jpeg_b64(image)}}}
    resp = None
    for url in _workflow_urls():
        try:
            resp = requests.post(url, json=payload, timeout=90)
        except requests.RequestException:
            raise HTTPException(502, "로보플로우 서버 연결 실패 (인터넷 확인)")
        if resp.status_code != 404:
            break                      # 404 면 다른 경로 형식으로 한 번 더
    if resp.status_code in (401, 403):
        raise HTTPException(401, "로보플로우 개인(Private) API 키를 확인하세요.")
    if resp.status_code == 404:
        raise HTTPException(502, "워크플로를 찾을 수 없어요. ROBOFLOW_WORKSPACE / ROBOFLOW_WORKFLOW_ID 확인")
    if not resp.ok:
        raise HTTPException(502, f"워크플로 오류: {resp.text[:150]}")
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(502, "워크플로 응답을 해석할 수 없어요.")
    area = _workflow_image_area(data, float(image.width * image.height))
    return _extract_workflow_boxes(data), area


def _detect_local(image: Image.Image):
    r = local_model.predict(image, conf=CONFIDENCE / 100.0, verbose=False)[0]
    names = r.names; boxes = []
    if r.boxes is not None:
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            boxes.append({"cls": names.get(int(b.cls[0]), "leaf"), "conf": float(b.conf[0]),
                          "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": (x2 - x1) * (y2 - y1)})
    return boxes, image.width * image.height


def _detect_demo(image: Image.Image):
    W, Hh = image.width, image.height; boxes = []
    for _ in range(random.randint(3, 11)):
        bw = random.uniform(0.12, 0.32) * W; bh = random.uniform(0.12, 0.32) * Hh
        cx = random.uniform(bw / 2, W - bw / 2); cy = random.uniform(bh / 2, Hh - bh / 2)
        cls = random.choices(["shoot", "mature leaf", "old leaf"], weights=[0.22, 0.55, 0.23])[0]
        boxes.append({"cls": cls, "conf": random.uniform(0.4, 0.95),
                      "x1": cx - bw / 2, "y1": cy - bh / 2, "x2": cx + bw / 2, "y2": cy + bh / 2, "area": bw * bh})
    return boxes, W * Hh


def _iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    if inter <= 0: return 0.0
    union = a["area"] + b["area"] - inter
    return inter / union if union > 0 else 0.0


# 잎이 아닌 클래스(화분·기준물 등)는 잎 계수에서 제외
NON_LEAF = {"object", "pot", "background", "ruler", "marker", "tag"}


def _stage(cls: str) -> str:
    """클래스 이름을 잎 단계(shoot/mature/old)로 매핑. 이름 표기 흔들려도 인식."""
    c = cls.lower().replace(" ", "").replace("_", "").replace("-", "")
    if "shoot" in c or "newleaf" in c or c.startswith("new"):
        return "shoot"        # 새순
    if "old" in c or "senes" in c:
        return "old"          # 노엽(노화잎)
    return "mature"           # 성엽(성숙잎) — 'leaf','matureleaf','mature'


# --------------------------------------------------------------------------- 원근 보정
# 비스듬히 찍힌 사진의 네 모서리를 직사각형으로 펴서, 왜곡 없는 탑뷰 좌표를 얻는다.
# 사진을 '이어 붙이는' 게 아니라 탐지된 좌표만 선반 좌표계로 옮긴다.
# → 잎이 촬영 사이에 움직여도 상관없고, 특징점 매칭도 필요 없다.

def _solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """가우스 소거법. 외부 라이브러리 없이 8x8 을 푼다."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise HTTPException(400, "모서리 네 점이 일직선이거나 겹칩니다. 다시 찍어 주세요.")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / pv
            if f:
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def homography(src: List[tuple], dst: List[tuple]) -> List[List[float]]:
    """네 점 대응 → 3x3 원근 변환 행렬. src/dst 는 [좌상, 우상, 우하, 좌하] 순서."""
    if len(src) != 4 or len(dst) != 4:
        raise HTTPException(400, "모서리는 정확히 4점이어야 해요.")
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = _solve_linear(A, b)
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def homography_lstsq(src: List[tuple], dst: List[tuple]) -> List[List[float]]:
    """대응점 4쌍 이상 → 최소제곱 원근 변환.

    모서리 대신 '이미 위치를 아는 화분'을 기준점으로 쓸 때 필요하다.
    점을 4개보다 많이 찍으면 클릭 오차가 평균돼 더 정확해진다.
    """
    if len(src) != len(dst) or len(src) < 4:
        raise HTTPException(400, "기준점이 4개 이상이어야 해요.")
    rows, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    # 정규방정식 AᵀA h = Aᵀb (8x8) 로 줄여서 푼다
    n = 8
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(rows[k][i] * b[k] for k in range(len(rows))) for i in range(n)]
    h = _solve_linear(ata, atb)
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def apply_h(H: List[List[float]], x: float, y: float) -> tuple:
    """점 하나를 원근 변환."""
    d = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(d) < 1e-12:
        d = 1e-12
    return ((H[0][0] * x + H[0][1] * y + H[0][2]) / d,
            (H[1][0] * x + H[1][1] * y + H[1][2]) / d)


def warp_box(H: List[List[float]], box: dict) -> dict:
    """박스의 네 꼭짓점을 변환한 뒤, 그걸 감싸는 축정렬 박스로 되돌린다."""
    pts = [apply_h(H, box["x1"], box["y1"]), apply_h(H, box["x2"], box["y1"]),
           apply_h(H, box["x2"], box["y2"]), apply_h(H, box["x1"], box["y2"])]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
    out = dict(box)
    out.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": max(0.0, (x2 - x1) * (y2 - y1))})
    return out


# --------------------------------------------------------------------------- 잎 → 식물 그룹화
# 탑뷰 사진 한 장에 여러 화분이 들어올 때, 어느 잎이 어느 식물인지 묶는다.
GROUP_GAP = float(os.environ.get("GROUP_GAP", "0.6"))   # 잎 크기 대비 '같은 무리' 인정 거리


def _center(b):
    return ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)


def _span(b):
    """잎 박스의 대표 크기(대각선 길이)."""
    return math.hypot(b["x2"] - b["x1"], b["y2"] - b["y1"])


def _contains(outer, px, py) -> bool:
    return outer["x1"] <= px <= outer["x2"] and outer["y1"] <= py <= outer["y2"]


# --- 잎 생김새(모양·색) 특징 ------------------------------------------------
# 품종마다 잎 모양·무늬·색이 뚜렷이 달라서, 같은 식물의 잎끼리는 서로 닮는다.
# 외부 API 없이 박스 비율 + 잘라낸 잎의 색으로 계산한다.
SHAPE_SIM = float(os.environ.get("SHAPE_SIM", "0.55"))   # 같은 모양으로 볼 최소 유사도(0~1)


def leaf_features(image: Image.Image, box: dict, scale: float = 1.0) -> dict:
    """잎 하나의 생김새 특징. 이미지가 없으면 박스 비율만으로 만든다."""
    w, h = box["x2"] - box["x1"], box["y2"] - box["y1"]
    if w <= 0 or h <= 0:
        return {"aspect": 1.0, "size": 1.0, "h": 0.0, "s": 0.0, "v": 0.0, "has_color": False}
    # 잎이 어느 방향으로 눕든 같은 값이 나오도록 긴 변/짧은 변
    feat = {"aspect": max(w, h) / min(w, h), "size": math.sqrt(w * h),
            "h": 0.0, "s": 0.0, "v": 0.0, "has_color": False}
    if image is None:
        return feat
    # 박스 가운데 60%만 봐서 가장자리 배경이 섞이는 걸 줄인다
    cx, cy = _center(box)
    hw, hh = w * 0.3, h * 0.3
    crop_box = (max(0, int((cx - hw) * scale)), max(0, int((cy - hh) * scale)),
                min(image.width, int((cx + hw) * scale)), min(image.height, int((cy + hh) * scale)))
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return feat
    px = list(image.crop(crop_box).resize((8, 8)).convert("HSV").getdata())
    if not px:
        return feat
    n = len(px)
    feat.update({"h": sum(p[0] for p in px) / n, "s": sum(p[1] for p in px) / n,
                 "v": sum(p[2] for p in px) / n, "has_color": True})
    return feat


def shape_similarity(a: dict, b: dict, size_ref: float = None) -> float:
    """잎 두 장이 얼마나 닮았는지 0(전혀)~1(똑같이).

    모양(긴변/짧은변 비율)·크기·색을 함께 본다. 색 정보가 없으면 모양·크기만.
    """
    d = (abs(a["aspect"] - b["aspect"]) / 1.2) ** 2
    ratio = a["size"] / b["size"] if b["size"] else 1.0
    d += (math.log(max(ratio, 1e-6)) / 0.9) ** 2
    if a.get("has_color") and b.get("has_color"):
        dh = abs(a["h"] - b["h"]); dh = min(dh, 256 - dh)      # 색상은 원형이라 최단거리
        d += (dh / 26.0) ** 2 + (abs(a["s"] - b["s"]) / 70.0) ** 2 + (abs(a["v"] - b["v"]) / 70.0) ** 2
    return math.exp(-math.sqrt(d))


def _label_shape_groups(items: List[dict], feats: List[dict]) -> None:
    """식물들을 생김새로 묶어 '모양 그룹' 딱지(A, B, C…)를 붙인다.

    같은 품종이면 다른 화분에 있어도 같은 딱지가 붙는다.
    """
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if shape_similarity(feats[i], feats[j]) >= SHAPE_SIM:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    # 큰 무리부터 A, B, C … 순서로
    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    order = sorted(buckets.values(), key=len, reverse=True)
    for gi, members in enumerate(order):
        tag = chr(ord("A") + gi) if gi < 26 else f"G{gi + 1}"
        for i in members:
            items[i]["shape_group"] = tag
            items[i]["shape_group_size"] = len(members)


def _mean_features(feats: List[dict]) -> dict:
    """잎 여러 장의 특징 평균 = 그 식물의 생김새."""
    if not feats:
        return {"aspect": 1.0, "size": 1.0, "h": 0.0, "s": 0.0, "v": 0.0, "has_color": False}
    n = len(feats)
    colored = [f for f in feats if f.get("has_color")]
    out = {k: sum(f[k] for f in feats) / n for k in ("aspect", "size")}
    if colored:
        m = len(colored)
        out.update({k: sum(f[k] for f in colored) / m for k in ("h", "s", "v")})
        out["has_color"] = True
    else:
        out.update({"h": 0.0, "s": 0.0, "v": 0.0, "has_color": False})
    return out


def synthetic_pots(canvas_w: float, canvas_h: float) -> List[dict]:
    """미리 지정한 화분 자리 → 화분 박스. 모델이 화분을 못 잡아도 그룹화가 된다.

    화분 크기는 이웃 화분과의 간격에서 추정한다(빽빽한 트레이에서 잘 맞는다).
    """
    if not POTS:
        return []
    pts = [(p["u"] * canvas_w, p["v"] * canvas_h) for p in POTS]
    if len(pts) > 1:
        nn = [min(math.dist(a, b) for j, b in enumerate(pts) if j != i) for i, a in enumerate(pts)]
        r = sorted(nn)[len(nn) // 2] * 0.45
    else:
        r = min(canvas_w, canvas_h) * 0.1
    return [{"cls": "pot", "conf": 1.0, "x1": x - r, "y1": y - r,
             "x2": x + r, "y2": y + r, "area": 4 * r * r} for x, y in pts]


def group_by_pots_indexed(leaves: List[dict], pots: List[dict],
                          feats: List[dict]) -> List[tuple]:
    """group_by_pots_and_shape 와 같되, 각 무리가 몇 번 화분에서 나왔는지도 알려 준다."""
    groups: List[List[dict]] = [[] for _ in pots]
    seed_feats: List[List[dict]] = [[] for _ in pots]
    strays = []
    for idx, lf in enumerate(leaves):
        lx, ly = _center(lf)
        inside = [i for i, p in enumerate(pots) if _contains(p, lx, ly)]
        if inside:
            best = min(inside, key=lambda i: math.dist(_center(pots[i]), (lx, ly)))
            groups[best].append(lf)
            seed_feats[best].append(feats[idx])
        else:
            strays.append(idx)

    ref = sum(_span(p) for p in pots) / len(pots)
    for idx in strays:
        lx_ly = _center(leaves[idx])

        def score(i):
            dist = math.dist(_center(pots[i]), lx_ly)
            near = ref / (ref + dist)
            sim = shape_similarity(feats[idx], _mean_features(seed_feats[i])) if seed_feats[i] else 0.5
            # 곱으로 본다 — 더하면 '아주 가깝다'가 '전혀 안 닮았다'를 덮어써서
            # 멀리 뻗은 잎이 옆 화분에 붙어 버린다. 곱이면 둘 다 맞아야 이긴다.
            return near * sim

        groups[max(range(len(pots)), key=score)].append(leaves[idx])
    return [(i, g) for i, g in enumerate(groups) if g]


def group_by_pots_and_shape(leaves: List[dict], pots: List[dict], feats: List[dict]) -> List[List[dict]]:
    """화분 + 생김새를 함께 쓰는 그룹화.

    · 화분 박스 '안'에 있는 잎 → 그 화분으로 확정 (씨앗)
    · 화분 밖으로 뻗은 잎 → 가까움 + 씨앗 잎과의 닮은 정도를 함께 보고 배정

    알로카시아는 잎이 화분 밖으로 멀리 뻗어서, 거리만 보면 옆 화분에 잘못 붙는다.
    이때 '같은 식물 잎끼리는 닮았다'는 성질이 교정해 준다.
    """
    return [g for _, g in group_by_pots_indexed(leaves, pots, feats)]


def group_by_distance_and_shape(leaves: List[dict], feats: List[dict],
                                gap: float = None) -> List[List[dict]]:
    """화분이 없을 때: 거리 + 생김새를 함께 본다.

    거리만 쓰면 뻗은 잎이 끊겨 과분할되는데, '닮았으면 조금 멀어도 같은 식물'로
    이어 줘서 이를 줄인다. 대신 안 닮은 잎끼리는 가까워도 잇지 않는다.
    """
    gap = GROUP_GAP if gap is None else gap
    n = len(leaves)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            base = (_span(leaves[i]) + _span(leaves[j])) / 2 * gap
            dist = math.dist(_center(leaves[i]), _center(leaves[j]))
            sim = shape_similarity(feats[i], feats[j])
            close = dist <= base or _iou(leaves[i], leaves[j]) > 0.05
            # 가깝다는 것만으로는 부족하다 — 겹친 잎이 옆 식물 것일 수 있어서
            # 최소한의 생김새 일치를 요구하고, 확실히 닮았으면 더 멀리까지 이어 준다.
            link = (close and sim >= SHAPE_SIM * 0.6) or (sim >= SHAPE_SIM and dist <= base * 4)
            if link:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    buckets: Dict[int, List[dict]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(leaves[i])
    return list(buckets.values())


def group_leaves(boxes: List[dict], image: Image.Image = None,
                 scale: float = 1.0, feats: List[dict] = None) -> List[List[dict]]:
    """탐지 박스 전체 → 식물별 잎 묶음.

    화분이 잡히면 화분+생김새, 아니면 거리+생김새로 묶는다.
    image 를 주면 잎을 잘라 색까지 보고, 없으면 박스 비율·크기만으로 판단한다.
    feats 를 주면 그걸 쓴다 — 사진이 여러 장이라 박스마다 원본이 다를 때 필요.
    """
    return group_plants(boxes, image, scale, feats)[0]


def group_plants(boxes: List[dict], image: Image.Image = None, scale: float = 1.0,
                 feats: List[dict] = None, canvas: tuple = None) -> tuple:
    """(무리 목록, 무리→고정자리) 를 돌려준다.

    canvas(=(폭, 높이)) 를 주고 화분 자리를 미리 지정해 뒀으면, 모델이 화분을
    못 잡아도 그 자리를 화분으로 삼는다. 이때 무리마다 자리가 고정된다.
    """
    leaves = [b for b in boxes if b["cls"].lower() not in NON_LEAF]
    pots = [b for b in boxes if b["cls"].lower() in NON_LEAF]
    if not leaves:
        return [], {}
    if feats is not None:
        by_id = {id(b): f for b, f in zip(boxes, feats)}
        lfeats = [by_id[id(b)] for b in leaves]
    else:
        lfeats = [leaf_features(image, b, scale) for b in leaves]

    predefined = False
    if not pots and POTS and canvas:
        pots = synthetic_pots(*canvas)
        predefined = True

    if pots:
        indexed = group_by_pots_indexed(leaves, pots, lfeats)
        groups = [g for _, g in indexed]
        slots = {id(g): POTS[i]["slot"] for i, g in indexed} if predefined else {}
        return groups, slots
    return group_by_distance_and_shape(leaves, lfeats), {}


def analyze_top(boxes: List[dict], img_area: float, ref_area: float = None) -> dict:
    """모델1: 식물의 '맨 위 잎'(광합성 주력) 크기 → 3D 온실 반영용.

    ref_area: 잎 크기를 재는 기준 면적. 기본은 사진 전체(=식물 1개를 찍은 사진).
    농장 전체를 한 장에 담은 경우엔 식물 1개 몫의 면적을 넘겨야 대/중/소엽이 맞는다.
    """
    leaves = [b for b in boxes if b["cls"].lower() not in NON_LEAF]
    if not leaves:
        return {"top_leaf_size": "없음", "top_leaf_pct": 0.0}
    ref = ref_area or img_area
    top = min(leaves, key=lambda b: b["y1"])          # 사진에서 가장 위쪽 잎
    # 좌표계가 어긋나도 100%를 넘지 않게 (워크플로 리사이즈 등)
    pct = min(round(top["area"] / ref * 100, 1), 100.0) if ref else 0.0
    size = "대엽" if pct > 18 else ("중엽" if pct > 8 else "소엽")
    return {"top_leaf_size": size, "top_leaf_pct": pct}


def analyze_metrics(boxes: List[dict], img_area: float) -> dict:
    leaves = [b for b in boxes if b["cls"].lower() not in NON_LEAF]
    leaf_count = len(leaves)

    # 단계별 개수 (Shoot / Mature / Old)
    counts = {"shoot": 0, "mature": 0, "old": 0}
    for b in leaves:
        counts[_stage(b["cls"])] += 1

    # 겹침
    OVERLAP_IOU = 0.12
    overlapping = set()
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            if _iou(leaves[i], leaves[j]) > OVERLAP_IOU:
                overlapping.add(i); overlapping.add(j)
    overlap_count = len(overlapping)
    overlap_density = round(overlap_count / leaf_count * 100) if leaf_count else 0

    # 크기 분류 (잎 면적 비율 + 개수)
    coverage = sum(b["area"] for b in leaves) / img_area if img_area else 0
    if coverage > 0.45 or leaf_count >= 9:
        size_class = "대품"
    elif coverage > 0.2 or leaf_count >= 5:
        size_class = "중품"
    else:
        size_class = "소품"

    return {"size_class": size_class, "leaf_count": leaf_count,
            "shoot_count": counts["shoot"], "mature_count": counts["mature"],
            "old_count": counts["old"], "overlap_count": overlap_count,
            "overlap_density": overlap_density}


def _recompute_shape_groups() -> None:
    """등록된 식물 전체를 생김새로 다시 묶어 '모양 그룹' 딱지를 갱신한다."""
    ids = [pid for pid in PLANTS if pid in FEATS]
    if ids:
        _label_shape_groups([PLANTS[pid] for pid in ids], [FEATS[pid] for pid in ids])


def _analyze_file(raw: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 읽을 수 없어요.")

    # 모델1(맨 위 잎) 실행
    boxes_top, img_area = detect_boxes(image, ROBOFLOW_MODEL_TOP)
    # 모델2(단계) 실행 — 두 모델이 다르고 로보플로우일 때만 따로 호출(아니면 재사용해 크레딧 절약)
    if ENGINE == "roboflow" and ROBOFLOW_MODEL_STAGE != ROBOFLOW_MODEL_TOP:
        boxes_stage, _ = detect_boxes(image, ROBOFLOW_MODEL_STAGE)
    else:
        boxes_stage = boxes_top

    metrics = {}
    metrics.update(analyze_top(boxes_top, img_area))        # 모델1 → 3D
    metrics.update(analyze_metrics(boxes_stage, img_area))  # 모델2 → 모달

    thumb = image.copy(); thumb.thumbnail((260, 260))
    tb = io.BytesIO(); thumb.save(tb, format="JPEG", quality=72)
    metrics["thumb"] = "data:image/jpeg;base64," + base64.b64encode(tb.getvalue()).decode()

    # 이 식물의 생김새 = 잎들의 특징 평균 (모양 그룹 딱지 계산에 쓰임)
    leaves = [b for b in boxes_top if b["cls"].lower() not in NON_LEAF]
    scale = image.width / math.sqrt(img_area * (image.width / image.height)) if img_area else 1.0
    feat = _mean_features([leaf_features(image, b, scale) for b in leaves])
    return metrics, feat


# --------------------------------------------------------------------------- 엔드포인트
@app.get("/")
def index():
    return FileResponse(os.path.join("static", "index.html"))


@app.get("/api/plants")
def list_plants():
    return {"engine": ENGINE_LABEL, "plants": list(PLANTS.values())}


@app.get("/api/slots")
def get_slots():
    """온실 자리(슬롯) 목록 + 점유 여부. 자리 지도·자리 선택에 사용."""
    used = {p.get("pos"): p["id"] for p in PLANTS.values()}
    return [{"label": s["label"], "x": s["x"], "z": s["z"],
             "occupied": s["label"] in used, "plant_id": used.get(s["label"])} for s in SLOTS]


@app.get("/api/pots")
def get_pots():
    """미리 지정해 둔 화분 자리."""
    return {"count": len(POTS), "pots": POTS}


@app.post("/api/pots")
async def set_pots(points: str = Form(None), points_px: str = Form(None),
                   corners: str = Form(None)):
    """화분 중심을 저장. 모델에 pot 클래스를 라벨링하는 대신 쓴다.

    화분은 안 움직이니 한 번만 지정하면 이후 스캔에서 계속 재사용되고,
    화분마다 자리(슬롯)가 고정돼 개체가 안 섞인다.

    · `points`    선반 정규좌표 `[[u,v], …]` (0~1)
    · `points_px` 사진 픽셀 좌표 + `corners`(네 모서리) → 서버가 원근 보정해서 변환
    """
    import json
    if points_px and corners:
        try:
            px, quad = json.loads(points_px), json.loads(corners)
        except ValueError:
            raise HTTPException(400, "좌표를 읽을 수 없어요 (JSON 형식).")
        H = homography([tuple(p) for p in quad], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        pts = [apply_h(H, float(p[0]), float(p[1])) for p in px]
    elif points:
        try:
            pts = json.loads(points)
        except ValueError:
            raise HTTPException(400, "points 를 읽을 수 없어요 (JSON 형식).")
    else:
        raise HTTPException(400, "points 또는 points_px+corners 가 필요해요.")
    if not isinstance(pts, list) or not pts:
        raise HTTPException(400, "화분을 하나 이상 찍어 주세요.")

    POTS.clear()
    taken = set()
    for i, p in enumerate(pts):
        try:
            u, v = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            raise HTTPException(400, f"{i + 1}번째 화분 좌표가 이상해요.")
        u, v = min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)
        slot = _nearest_slot((u - 0.5) * _W, (v - 0.5) * _D, taken)
        if slot is None:
            raise HTTPException(400, "자리가 모자라요. 화분 수를 줄여 주세요.")
        taken.add(slot["label"])
        POTS.append({"i": i, "u": round(u, 4), "v": round(v, 4), "slot": slot["label"]})
    save_state()
    return {"count": len(POTS), "pots": POTS}


@app.delete("/api/pots")
def clear_pots():
    POTS.clear()
    save_state()
    return {"ok": True}


@app.post("/api/plants")
async def add_plant(name: str = Form(...), file: UploadFile = File(...), pos: str = Form(None)):
    """사진 + 이름 + 자리로 식물을 3D 온실에 추가(분석 포함)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드할 수 있어요.")
    name = name.strip() or "이름없는 알로카시아"

    slot = _free_slot(pos)
    if slot is None:
        raise HTTPException(400, "빈 자리가 없어요. 식물을 제거해 자리를 비워 주세요.")

    raw = await file.read()
    metrics, feat = _analyze_file(raw)

    pid = uuid.uuid4().hex[:8]
    plant = {"id": pid, "name": name, "pos": slot["label"], "x": slot["x"], "z": slot["z"], "rot": 0,
             "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    PLANTS[pid] = plant
    FEATS[pid] = feat
    _recompute_shape_groups()
    save_state()
    return plant


def _nearest_slot(x: float, z: float, blocked: set):
    """3D 좌표에 가장 가까운, 아직 안 쓴 자리."""
    free = [s for s in SLOTS if s["label"] not in blocked]
    if not free:
        return None
    return min(free, key=lambda s: math.dist((s["x"], s["z"]), (x, z)))


def _crop_thumb(image: Image.Image, group: List[dict], scale: float) -> str:
    """식물 무리의 바운딩 박스를 잘라 썸네일로. 모달에서 그 개체만 보이게."""
    x1 = min(b["x1"] for b in group) * scale
    y1 = min(b["y1"] for b in group) * scale
    x2 = max(b["x2"] for b in group) * scale
    y2 = max(b["y2"] for b in group) * scale
    pad = 0.06 * max(x2 - x1, y2 - y1)
    box = (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
           min(image.width, int(x2 + pad)), min(image.height, int(y2 + pad)))
    if box[2] <= box[0] or box[3] <= box[1]:
        box = (0, 0, image.width, image.height)
    crop = image.crop(box); crop.thumbnail((260, 260))
    tb = io.BytesIO(); crop.save(tb, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(tb.getvalue()).decode()


@app.post("/api/scan")
async def scan_farm(file: UploadFile = File(...), replace: str = Form(None),
                    mode: str = Form(None)):
    """농장 전체를 찍은 탑뷰 사진 1장 → 잎을 식물별로 묶어 여러 개체를 한 번에 등록.

    사진 속 위치가 그대로 3D 온실 자리로 이어진다(사진 위쪽 = 온실 안쪽 A줄).
    같은 자리에 이미 식물이 있으면 이름·방향을 유지한 채 상태만 갱신한다.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드할 수 있어요.")
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 읽을 수 없어요.")

    boxes, img_area = detect_boxes(image, ROBOFLOW_MODEL_TOP)

    # 박스 좌표계의 폭·높이 (워크플로가 리사이즈했어도 비율은 유지된다고 본다)
    aspect = image.width / image.height
    cw = math.sqrt(img_area * aspect)
    ch = math.sqrt(img_area / aspect) if aspect else float(image.height)
    scale = image.width / cw if cw else 1.0

    groups, slots = group_plants(boxes, image, scale, canvas=(cw, ch))
    if not groups:
        raise HTTPException(400, "사진에서 잎을 찾지 못했어요. 조명·각도를 바꿔 다시 찍어 주세요.")

    scan_mode = _scan_mode(mode, replace)
    if scan_mode == "replace":
        PLANTS.clear()
        FEATS.clear()

    result = _register_groups(
        groups, cw, ch,
        thumb_of=lambda g: _crop_thumb(image, g, scale),
        feat_of=lambda g: _mean_features([leaf_features(image, b, scale) for b in g]),
        per_plant_area=img_area / len(groups),      # 식물 1개 몫 — 대/중/소엽 기준
        img_area=img_area,
        slot_of=(lambda g: slots.get(id(g))) if slots else None,
        mode=scan_mode)

    shapes = {p.get("shape_group") for p in result if p.get("shape_group")}
    return {"count": len(result), "grouped_by": _grouped_by(boxes, slots),
            "shape_groups": len(shapes), "mode": scan_mode,
            "new_leaves": sum(p.get("new_leaves", 0) for p in result),
            "plants": result}


SCAN_MODES = ("replace", "update", "keep")


def _scan_mode(mode: str, replace: str) -> str:
    """스캔 모드 정리. replace 는 예전 방식이라 mode 로 접어 준다."""
    if replace and not mode:
        return "replace"
    mode = (mode or "update").strip()
    if mode not in SCAN_MODES:
        raise HTTPException(400, f"스캔 모드는 {' / '.join(SCAN_MODES)} 중 하나여야 해요.")
    return mode


def _merge_keep(old: dict, new: dict) -> tuple:
    """'잎은 유지, 새 잎만 기록' 병합.

    사진마다 잎이 가려져 탐지가 들쭉날쭉한데, 그걸 그대로 반영하면 어제 8장이던
    식물이 오늘 5장이 된다. 그래서 잎 수가 줄어드는 건 '가려진 것'으로 보고 무시하고,
    늘어난 만큼만 새 잎으로 기록한다.

    단계 분포(새순/성엽/노엽)는 탐지가 기존만큼 잡았을 때만 갱신한다.
    그래야 새순이 성엽이 되는 변화는 따라가면서, 덜 잡힌 사진에 값이 깎이지 않는다.
    """
    o_tot = old.get("leaf_count", 0) or 0
    n_tot = new.get("leaf_count", 0) or 0
    added = max(0, n_tot - o_tot)
    out = dict(new)
    if n_tot < o_tot:                       # 덜 잡혔다 → 잎 수치는 기존 값을 지킨다
        for k in ("shoot_count", "mature_count", "old_count", "leaf_count", "size_class"):
            if k in old:
                out[k] = old[k]
    return out, added


def _pot_xz(slot_label: str):
    """지정한 화분의 실제 위치(cm). 격자 칸 중앙이 아니라 찍은 그 자리."""
    pot = next((p for p in POTS if p["slot"] == slot_label), None)
    if pot is None:
        return None
    return (pot["u"] - 0.5) * _W, (pot["v"] - 0.5) * _D


def _ensure_pot_slots(result: List[dict]) -> None:
    """지정해 둔 화분은 잎이 안 잡혀도 자리를 지킨다.

    잎이 하나도 탐지되지 않은 화분은 그룹이 안 만들어져 사라졌고, 그래서 등록된
    개체 수가 실제 화분 수보다 적게 나왔다. 빈 화분(흙만)이거나 큰 잎에 완전히
    가린 화분이 그런 경우다. 자리는 남겨 두고 '잎 0장'으로 표시한다.
    """
    if not POTS:
        return
    by_slot = {p["pos"] for p in PLANTS.values()}
    for pot in POTS:
        if pot["slot"] in by_slot:
            continue
        slot = _slot_by_label(pot["slot"])
        if slot is None:
            continue
        xz = _pot_xz(slot["label"]) or (slot["x"], slot["z"])
        pid = uuid.uuid4().hex[:8]
        plant = {"id": pid, "name": f"식물 {slot['label']}", "pos": slot["label"],
                 "x": xz[0], "z": xz[1], "rot": 0,
                 # 잎을 못 찾은 것뿐이지 '작은 식물'이라는 뜻이 아니다.
                 # 소품으로 적어 두면 실제 소품과 구분이 안 된다.
                 "size_class": "미검출", "leaf_count": 0, "shoot_count": 0,
                 "mature_count": 0, "old_count": 0,
                 "top_leaf_size": "없음", "top_leaf_pct": 0.0,
                 "overlap_count": 0, "overlap_density": 0, "empty": True,
                 "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        PLANTS[pid] = plant
        result.append(plant)


def _grouped_by(boxes: List[dict], slots: dict) -> str:
    """어떤 신호로 묶었는지 — UI 에 그대로 보여 준다."""
    if slots:
        return "pot_preset"                                    # 미리 지정한 화분 자리
    if any(b["cls"].lower() in NON_LEAF for b in boxes):
        return "pot"                                           # 모델이 잡은 화분
    return "distance"


def _register_groups(groups: List[List[dict]], canvas_w: float, canvas_h: float,
                     thumb_of, feat_of, per_plant_area: float, img_area: float,
                     slot_of=None, mode: str = "update") -> List[dict]:
    """무리 목록 → 자리 배정 후 등록/갱신. 스캔 엔드포인트들이 함께 쓴다.

    slot_of 를 주면(화분 자리를 미리 지정한 경우) 그 자리에 고정 배정한다.
    그러면 매번 스캔해도 같은 화분이 같은 자리로 가서 개체가 안 섞인다.
    """
    groups.sort(key=lambda g: sum(_center(b)[1] for b in g) / len(g))
    occupied = {p["pos"] for p in PLANTS.values()}
    claimed = set()
    by_slot = {p["pos"]: p for p in PLANTS.values()}
    result = []
    for g in groups:
        cx = sum(_center(b)[0] for b in g) / len(g)
        cy = sum(_center(b)[1] for b in g) / len(g)
        u = min(max(cx / canvas_w, 0.0), 1.0) if canvas_w else 0.5
        v = min(max(cy / canvas_h, 0.0), 1.0) if canvas_h else 0.5
        x, z = (u - 0.5) * _W, (v - 0.5) * _D

        forced = slot_of(g) if slot_of else None
        if forced:
            slot = _slot_by_label(forced)
            existing = by_slot.get(forced)
        else:
            existing = None
            cand = min((p for p in by_slot.values() if p["pos"] not in claimed),
                       key=lambda p: math.dist((p["x"], p["z"]), (x, z)), default=None)
            if cand is not None and math.dist((cand["x"], cand["z"]), (x, z)) <= max(_CW, _CD):
                existing = cand
            slot = _slot_by_label(existing["pos"]) if existing else _nearest_slot(x, z, occupied | claimed)
        if slot is None or slot["label"] in claimed:
            continue
        claimed.add(slot["label"])
        # 격자 칸 중앙으로 스냅하면 가까운 화분끼리 같은 줄로 뭉친다.
        # 지정한 화분이면 그 화분의 실제 자리를, 아니면 잎 무리의 무게중심을 쓴다.
        px, pz = _pot_xz(slot["label"]) or (x, z)

        metrics = {}
        metrics.update(analyze_top(g, img_area, ref_area=per_plant_area))
        metrics.update(analyze_metrics(g, img_area))
        metrics["thumb"] = thumb_of(g)
        feat = feat_of(g)

        if existing:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if mode == "keep":
                metrics, added = _merge_keep(existing, metrics)
                if added:
                    log = existing.setdefault("leaf_log", [])
                    log.append({"at": now, "added": added,
                                "total": metrics.get("leaf_count", 0)})
                    del log[:-30]                      # 최근 30건만
                    existing["new_leaves"] = added
                else:
                    existing["new_leaves"] = 0
            else:
                existing.pop("manual", None)           # 새 탐지값으로 덮어씀
            existing.pop("empty", None)                # 잎이 잡혔으니 빈 화분 아님
            existing["x"], existing["z"] = round(px, 2), round(pz, 2)
            existing.update(metrics)
            existing["updated"] = now
            FEATS[existing["id"]] = feat
            result.append(existing)
        else:
            pid = uuid.uuid4().hex[:8]
            plant = {"id": pid, "name": f"식물 {slot['label']}", "pos": slot["label"],
                     "x": round(px, 2), "z": round(pz, 2), "rot": 0,
                     "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
            PLANTS[pid] = plant
            FEATS[pid] = feat
            by_slot[slot["label"]] = plant
            result.append(plant)
    _ensure_pot_slots(result)
    _recompute_shape_groups()
    save_state()
    return result


@app.post("/api/scan-multi")
async def scan_multi(files: List[UploadFile] = File(...), corners: str = Form(None),
                     regions: str = Form(None), replace: str = Form(None),
                     pot_refs: str = Form(None), mode: str = Form(None)):
    """여러 장을 원근 보정해 하나의 선반 좌표계로 합친 뒤 식물을 등록.

    사진을 이어 붙이지 않는다. 사진마다 네 모서리로 원근 변환을 구해
    '탐지 결과'만 선반 좌표로 옮기고, 겹치는 구역의 중복 탐지를 지운다.
    잎이 촬영 사이에 움직여도 안전하고 특징점 매칭이 필요 없다.

    기준점은 둘 중 하나로 준다:
    · corners  사진별 [좌상, 우상, 우하, 좌하] 픽셀 좌표 + regions(그 사진이 덮는 구역)
    · pot_refs 사진별 [[화분번호, x, y], …] — 이미 위치를 아는 화분 4개 이상을 찍는다.
               케이지가 좁아 트레이 모서리가 프레임에 안 들어올 때 쓴다. 화분은 항상
               잘 보이고 안 움직이며, 프레임 안쪽에 퍼져 있어 외삽이 아니라 내삽이 된다.
    """
    import json
    refs = None
    if pot_refs:
        if not POTS:
            raise HTTPException(400, "화분 자리를 먼저 지정해 주세요 (🪴 화분 자리 지정).")
        try:
            refs = json.loads(pot_refs)
        except ValueError:
            raise HTTPException(400, "pot_refs 를 읽을 수 없어요 (JSON 형식).")
        if len(refs) != len(files):
            raise HTTPException(400, f"사진 {len(files)}장인데 기준점은 {len(refs)}장분이에요.")
    if corners:
        try:
            quads = json.loads(corners)
        except ValueError:
            raise HTTPException(400, "corners 를 읽을 수 없어요 (JSON 형식).")
        if len(quads) != len(files):
            raise HTTPException(400, f"사진 {len(files)}장인데 모서리는 {len(quads)}장분이에요.")
    elif refs:
        quads = [None] * len(files)
    else:
        raise HTTPException(400, "corners 또는 pot_refs 가 필요해요.")
    if regions:
        try:
            rects = json.loads(regions)
        except ValueError:
            raise HTTPException(400, "regions 를 읽을 수 없어요 (JSON 형식).")
        if len(rects) != len(files):
            raise HTTPException(400, "regions 개수가 사진 수와 달라요.")
    else:
        rects = [[0, 0, 1, 1]] * len(files)

    CANVAS = 1000.0                       # 선반 좌표계 폭(가상 픽셀). 높이는 선반 비율대로
    canvas_h = CANVAS * (_D / _W)

    merged, feats, sources = [], [], []
    images = []
    for i, f in enumerate(files):
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(400, "이미지 파일만 올릴 수 있어요.")
        try:
            image = Image.open(io.BytesIO(await f.read())).convert("RGB")
        except Exception:
            raise HTTPException(400, f"{i + 1}번째 이미지를 읽을 수 없어요.")
        images.append(image)

        boxes, img_area = detect_boxes(image, ROBOFLOW_MODEL_TOP)
        aspect = image.width / image.height
        cw = math.sqrt(img_area * aspect)
        scale = image.width / cw if cw else 1.0

        if refs and refs[i]:
            # 이미 위치를 아는 화분들을 기준점으로 (모서리가 안 보일 때)
            src, dst = [], []
            for ref in refs[i]:
                try:
                    pi, px, py = int(ref[0]), float(ref[1]), float(ref[2])
                except (TypeError, ValueError, IndexError):
                    raise HTTPException(400, f"{i + 1}번째 사진의 기준점 형식이 이상해요.")
                if not 0 <= pi < len(POTS):
                    raise HTTPException(400, f"{pi}번 화분이 없어요.")
                src.append((px / scale, py / scale))
                dst.append((POTS[pi]["u"] * CANVAS, POTS[pi]["v"] * canvas_h))
            H = homography_lstsq(src, dst)
        else:
            u0, v0, u1, v1 = rects[i]
            dst = [(u0 * CANVAS, v0 * canvas_h), (u1 * CANVAS, v0 * canvas_h),
                   (u1 * CANVAS, v1 * canvas_h), (u0 * CANVAS, v1 * canvas_h)]
            src = [(p[0] / scale, p[1] / scale) for p in quads[i]]   # 클릭은 원본 픽셀 기준
            H = homography(src, dst)

        for b in boxes:
            merged.append(warp_box(H, b))
            feats.append(leaf_features(image, b, scale))
            sources.append((i, b))

    # 겹치는 구역에서 같은 잎이 두 번 잡힌 것 제거 (사진이 다르고 위치가 거의 같으면 중복)
    keep = []
    for idx, b in enumerate(merged):
        dup = None
        for kidx in keep:
            if sources[kidx][0] != sources[idx][0] and _iou(merged[kidx], b) > 0.5:
                dup = kidx
                break
        if dup is None:
            keep.append(idx)
        elif b["conf"] > merged[dup]["conf"]:
            keep[keep.index(dup)] = idx                 # 더 확신하는 쪽을 남긴다
    dropped = len(merged) - len(keep)

    boxes_m = [merged[i] for i in keep]
    feats_m = [feats[i] for i in keep]
    groups, slots = group_plants(boxes_m, feats=feats_m, canvas=(CANVAS, canvas_h))
    if not groups:
        raise HTTPException(400, "사진에서 잎을 찾지 못했어요. 조명·각도를 바꿔 다시 찍어 주세요.")

    scan_mode = _scan_mode(mode, replace)
    if scan_mode == "replace":
        PLANTS.clear(); FEATS.clear()

    src_of = {id(merged[i]): sources[i] for i in keep}
    feat_of_box = {id(merged[i]): feats[i] for i in keep}

    def thumb_of(group):
        img_i, _ = src_of[id(max(group, key=lambda b: b["area"]))]
        orig = [src_of[id(b)][1] for b in group if src_of[id(b)][0] == img_i]
        return _crop_thumb(images[img_i], orig, 1.0)

    def feat_of(group):
        return _mean_features([feat_of_box[id(b)] for b in group])

    canvas_area = CANVAS * canvas_h
    result = _register_groups(groups, CANVAS, canvas_h, thumb_of, feat_of,
                              canvas_area / len(groups), canvas_area,
                              slot_of=(lambda g: slots.get(id(g))) if slots else None,
                              mode=scan_mode)
    shapes = {p.get("shape_group") for p in result if p.get("shape_group")}
    return {"count": len(result), "photos": len(files), "merged_boxes": len(boxes_m),
            "deduped": dropped, "shape_groups": len(shapes), "mode": scan_mode,
            "new_leaves": sum(p.get("new_leaves", 0) for p in result),
            "grouped_by": _grouped_by(boxes_m, slots), "plants": result}


@app.post("/api/plants/{pid}/reanalyze")
async def reanalyze(pid: str, file: UploadFile = File(...)):
    """기존 식물에 새 사진으로 상태 갱신."""
    if pid not in PLANTS:
        raise HTTPException(404, "없는 식물")
    raw = await file.read()
    metrics, feat = _analyze_file(raw)
    PLANTS[pid].update(metrics)
    PLANTS[pid]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    FEATS[pid] = feat
    _recompute_shape_groups()
    save_state()
    return PLANTS[pid]


SIZE_CLASSES = ("소품", "중품", "대품")


@app.patch("/api/plants/{pid}")
async def update_plant(pid: str, name: str = Form(None), rot: float = Form(None),
                       size_class: str = Form(None), shoot_count: int = Form(None),
                       mature_count: int = Form(None), old_count: int = Form(None)):
    """이름 · 화분 방향 · 그리고 사람이 직접 고치는 값들.

    사진이 뭉개지거나 잎이 가려지면 탐지가 틀립니다. 그럴 때 손으로 바로잡으라고
    크기 등급과 단계별 잎 수를 열어 뒀어요. 단계 이동(성엽→노엽)은 한쪽을 줄이고
    다른 쪽을 늘리면 됩니다. 직접 고친 식물은 manual 로 표시됩니다.
    """
    if pid not in PLANTS:
        raise HTTPException(404, "없는 식물")
    p = PLANTS[pid]
    if name is not None and name.strip():
        p["name"] = name.strip()
    if rot is not None:
        p["rot"] = float(rot) % 360

    touched = False
    if size_class is not None:
        if size_class not in SIZE_CLASSES:
            raise HTTPException(400, f"크기 등급은 {' / '.join(SIZE_CLASSES)} 중 하나여야 해요.")
        p["size_class"] = size_class
        touched = True
    for key, val in (("shoot_count", shoot_count), ("mature_count", mature_count),
                     ("old_count", old_count)):
        if val is not None:
            if val < 0:
                raise HTTPException(400, "잎 개수는 0보다 작을 수 없어요.")
            p[key] = int(val)
            touched = True
    if touched:
        p["leaf_count"] = sum(p.get(k, 0) or 0 for k in
                              ("shoot_count", "mature_count", "old_count"))
        p["manual"] = True
        p["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state()
    return p


@app.delete("/api/plants/{pid}")
def remove_plant(pid: str):
    """식물 제거."""
    if pid not in PLANTS:
        raise HTTPException(404, "없는 식물")
    del PLANTS[pid]
    FEATS.pop(pid, None)
    _recompute_shape_groups()
    save_state()
    return {"ok": True, "id": pid}


app.mount("/static", StaticFiles(directory="static"), name="static")
