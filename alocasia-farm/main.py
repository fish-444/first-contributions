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


def group_by_pots(leaves: List[dict], pots: List[dict]) -> List[List[dict]]:
    """화분 기준 그룹화 — 1화분 = 1식물. 잎은 자기가 속한(없으면 가장 가까운) 화분에 붙는다.

    화분은 서로 겹치지 않아서 잎끼리의 거리보다 훨씬 믿을 만한 신호다.
    """
    groups = [[] for _ in pots]
    for lf in leaves:
        lx, ly = _center(lf)
        inside = [i for i, p in enumerate(pots) if _contains(p, lx, ly)]
        if inside:                     # 화분 박스 안에 잎 중심이 있으면 그 화분
            best = min(inside, key=lambda i: math.dist(_center(pots[i]), (lx, ly)))
        else:                          # 아니면 가장 가까운 화분
            best = min(range(len(pots)), key=lambda i: math.dist(_center(pots[i]), (lx, ly)))
        groups[best].append(lf)
    return [g for g in groups if g]


def group_by_distance(leaves: List[dict], gap: float = None) -> List[List[dict]]:
    """거리 클러스터링 — 화분이 안 잡힐 때의 대비책.

    잎 중심이 서로 (평균 잎 크기 x GROUP_GAP) 안쪽이거나 서로 겹치면 같은 무리로 본다.
    캐노피가 맞닿은 개체끼리는 하나로 합쳐질 수 있다 (화분 방식이 더 정확한 이유).
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
            reach = (_span(leaves[i]) + _span(leaves[j])) / 2 * gap
            near = math.dist(_center(leaves[i]), _center(leaves[j])) <= reach
            if near or _iou(leaves[i], leaves[j]) > 0.05:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    buckets: Dict[int, List[dict]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(leaves[i])
    return list(buckets.values())


def group_leaves(boxes: List[dict]) -> List[List[dict]]:
    """탐지 박스 전체 → 식물별 잎 묶음. 화분이 잡히면 화분 기준, 아니면 거리 기준."""
    leaves = [b for b in boxes if b["cls"].lower() not in NON_LEAF]
    pots = [b for b in boxes if b["cls"].lower() in NON_LEAF]
    if not leaves:
        return []
    if pots:
        return group_by_pots(leaves, pots)
    return group_by_distance(leaves)


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
    return metrics


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
    metrics = _analyze_file(raw)

    pid = uuid.uuid4().hex[:8]
    plant = {"id": pid, "name": name, "pos": slot["label"], "x": slot["x"], "z": slot["z"], "rot": 0,
             "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    PLANTS[pid] = plant
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
async def scan_farm(file: UploadFile = File(...), replace: str = Form(None)):
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
    groups = group_leaves(boxes)
    if not groups:
        raise HTTPException(400, "사진에서 잎을 찾지 못했어요. 조명·각도를 바꿔 다시 찍어 주세요.")

    # 박스 좌표계의 폭·높이 (워크플로가 리사이즈했어도 비율은 유지된다고 본다)
    aspect = image.width / image.height
    cw = math.sqrt(img_area * aspect)
    ch = math.sqrt(img_area / aspect) if aspect else float(image.height)
    scale = image.width / cw if cw else 1.0

    if replace:
        PLANTS.clear()

    # 사진 위쪽(안쪽 줄)부터 처리해야 자리 배정이 안정적
    groups.sort(key=lambda g: sum(_center(b)[1] for b in g) / len(g))
    per_plant_area = img_area / len(groups)          # 식물 1개 몫 — 대/중/소엽 기준

    occupied = {p["pos"] for p in PLANTS.values()}   # 기존 식물이 이미 쓰는 자리
    claimed = set()                                  # 이번 스캔에서 배정한 자리
    by_slot = {p["pos"]: p for p in PLANTS.values()}
    result = []
    for g in groups:
        cx = sum(_center(b)[0] for b in g) / len(g)
        cy = sum(_center(b)[1] for b in g) / len(g)
        u = min(max(cx / cw, 0.0), 1.0) if cw else 0.5
        v = min(max(cy / ch, 0.0), 1.0) if ch else 0.5
        x, z = (u - 0.5) * _W, (v - 0.5) * _D

        # 이미 그 근처에 등록된 식물이 있으면 그걸 갱신(이름·방향 유지)
        existing = None
        cand = min((p for p in by_slot.values() if p["pos"] not in claimed),
                   key=lambda p: math.dist((p["x"], p["z"]), (x, z)), default=None)
        if cand is not None and math.dist((cand["x"], cand["z"]), (x, z)) <= max(_CW, _CD):
            existing = cand
        slot = _slot_by_label(existing["pos"]) if existing else _nearest_slot(x, z, occupied | claimed)
        if slot is None:
            break                                    # 자리가 다 찼으면 중단
        claimed.add(slot["label"])

        metrics = {}
        metrics.update(analyze_top(g, img_area, ref_area=per_plant_area))
        metrics.update(analyze_metrics(g, img_area))
        metrics["thumb"] = _crop_thumb(image, g, scale)

        if existing:
            existing.update(metrics)
            existing["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            result.append(existing)
        else:
            pid = uuid.uuid4().hex[:8]
            plant = {"id": pid, "name": f"식물 {slot['label']}", "pos": slot["label"],
                     "x": slot["x"], "z": slot["z"], "rot": 0,
                     "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
            PLANTS[pid] = plant
            by_slot[slot["label"]] = plant
            result.append(plant)

    return {"count": len(result), "grouped_by": "pot" if any(
        b["cls"].lower() in NON_LEAF for b in boxes) else "distance", "plants": result}


@app.post("/api/plants/{pid}/reanalyze")
async def reanalyze(pid: str, file: UploadFile = File(...)):
    """기존 식물에 새 사진으로 상태 갱신."""
    if pid not in PLANTS:
        raise HTTPException(404, "없는 식물")
    raw = await file.read()
    metrics = _analyze_file(raw)
    PLANTS[pid].update(metrics)
    PLANTS[pid]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return PLANTS[pid]


@app.patch("/api/plants/{pid}")
async def update_plant(pid: str, name: str = Form(None), rot: float = Form(None)):
    """식물 이름 커스텀 / 화분 방향(회전) 설정."""
    if pid not in PLANTS:
        raise HTTPException(404, "없는 식물")
    if name is not None and name.strip():
        PLANTS[pid]["name"] = name.strip()
    if rot is not None:
        PLANTS[pid]["rot"] = float(rot) % 360
    return PLANTS[pid]


@app.delete("/api/plants/{pid}")
def remove_plant(pid: str):
    """식물 제거."""
    if pid not in PLANTS:
        raise HTTPException(404, "없는 식물")
    del PLANTS[pid]
    return {"ok": True, "id": pid}


app.mount("/static", StaticFiles(directory="static"), name="static")
