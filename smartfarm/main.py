"""
스마트팜 온실 3D 대시보드 - FastAPI 백엔드
=============================================

전체 흐름:
  [브라우저] 특정 구역 식물 사진 업로드
      │  POST /api/analyze  (file, plant_id)
      ▼
  [FastAPI] YOLO 로 사진 분석 →
      · 식물 크기 분류(대품/중품/소품)
      · 총 잎 개수 / 겹친 잎 개수(overlapping)
      · 새순(new shoot) 유무
      +  가상 IoT 센서(온도, 빛 효율)
      +  RAG 기반 생장 피드백
      → 해당 식물 상태로 가공·저장
      ▼
  [브라우저] 3D 온실의 해당 식물 오브젝트가 실시간 갱신,
             클릭하면 상세 정보 모달 표시

YOLO 엔진은 3가지 중 자동 선택됩니다:
  1) 로보플로우(ROBOFLOW_API_KEY 있으면)  2) 로컬 ultralytics(.pt)  3) 데모(모델 없어도 동작)

실행법은 README.md 참고.
"""

import base64
import io
import os
import random
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ===========================================================================
# 설정
# ===========================================================================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "find-leaf-and-object/1")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
CONFIDENCE = float(os.environ.get("CONFIDENCE", "25"))  # 0~100

# 엔진 선택
if ROBOFLOW_API_KEY:
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
        print(f"[경고] 로컬 모델 로딩 실패({e}) → 데모 모드로 전환")
        ENGINE = "demo"

print(f"[분석 엔진] {ENGINE}")

app = FastAPI(title="Smart Farm 3D Dashboard")

# ===========================================================================
# 초기 온실 배치 (구역 A/B/C, 식물 12개) — 3D 가 이걸 보고 오브젝트를 만든다
# ===========================================================================
ZONES = ["A", "B", "C"]


def _new_plant(pid: str, zone: str, x: float, z: float) -> dict:
    return {
        "id": pid, "zone": zone, "x": x, "z": z,
        "analyzed": False,
        "size_class": "미분석",
        "leaf_count": 0,
        "overlap_count": 0,
        "new_shoot": False,
        "temp": None,
        "light_eff": None,
        "feedback": "아직 분석 전이에요. 사진을 업로드해 주세요.",
        "thumb": None,        # 분석한 사진 썸네일(base64)
        "updated": None,
    }


def _build_layout() -> Dict[str, dict]:
    plants: Dict[str, dict] = {}
    # 각 구역마다 4개씩, 가로로 배치
    for zi, zone in enumerate(ZONES):
        for i in range(4):
            pid = f"{zone}{i + 1}"
            x = -21 + i * 14                 # 가로 위치
            z = -14 + zi * 14                # 구역별 깊이
            plants[pid] = _new_plant(pid, zone, x, z)
    return plants


PLANTS: Dict[str, dict] = _build_layout()

# ===========================================================================
# YOLO 탐지 → 박스 목록으로 표준화
#   반환: [{ "cls": str, "conf": float, "x1","y1","x2","y2": float, "area": float }]
#   그리고 (이미지 넓이*높이)
# ===========================================================================
def detect_boxes(image: Image.Image):
    if ENGINE == "roboflow":
        return _detect_roboflow(image)
    if ENGINE == "local":
        return _detect_local(image)
    return _detect_demo(image)


def _detect_roboflow(image: Image.Image):
    import requests
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = requests.post(
            f"{ROBOFLOW_API_URL}/{ROBOFLOW_MODEL_ID}",
            params={"api_key": ROBOFLOW_API_KEY, "confidence": CONFIDENCE},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )
    except requests.RequestException:
        raise HTTPException(502, "로보플로우 서버 연결 실패 (인터넷 확인)")
    if resp.status_code in (401, 403):
        raise HTTPException(401, "로보플로우 API 키가 올바르지 않아요.")
    if not resp.ok:
        raise HTTPException(502, f"로보플로우 오류: {resp.text[:150]}")
    data = resp.json()
    boxes = []
    for p in data.get("predictions", []):
        w, h = p["width"], p["height"]
        boxes.append({
            "cls": p.get("class", "leaf"), "conf": float(p.get("confidence", 0)),
            "x1": p["x"] - w / 2, "y1": p["y"] - h / 2,
            "x2": p["x"] + w / 2, "y2": p["y"] + h / 2, "area": w * h,
        })
    return boxes, image.width * image.height


def _detect_local(image: Image.Image):
    results = local_model.predict(image, conf=CONFIDENCE / 100.0, verbose=False)
    r = results[0]
    names = r.names
    boxes = []
    if r.boxes is not None:
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            boxes.append({
                "cls": names.get(int(b.cls[0]), "leaf"), "conf": float(b.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": (x2 - x1) * (y2 - y1),
            })
    return boxes, image.width * image.height


def _detect_demo(image: Image.Image):
    """모델이 없어도 대시보드를 체험할 수 있게 그럴듯한 가짜 탐지를 생성."""
    W, Hh = image.width, image.height
    n = random.randint(3, 11)
    boxes = []
    for _ in range(n):
        bw = random.uniform(0.12, 0.32) * W
        bh = random.uniform(0.12, 0.32) * Hh
        cx = random.uniform(bw / 2, W - bw / 2)
        cy = random.uniform(bh / 2, Hh - bh / 2)
        cls = "newleaf" if random.random() < 0.25 else "leaf"
        boxes.append({
            "cls": cls, "conf": random.uniform(0.4, 0.95),
            "x1": cx - bw / 2, "y1": cy - bh / 2,
            "x2": cx + bw / 2, "y2": cy + bh / 2, "area": bw * bh,
        })
    return boxes, W * Hh


# ===========================================================================
# 박스 목록 → 식물 지표 (크기분류 / 잎수 / 겹침 / 새순)
# ===========================================================================
def _iou(a, b) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a["area"] + b["area"] - inter
    return inter / union if union > 0 else 0.0


def analyze_metrics(boxes: List[dict], img_area: float) -> dict:
    leaves = [b for b in boxes if b["cls"] in ("leaf", "newleaf", "new_shoot", "new-shoot")]
    leaf_count = len(leaves)

    # 겹친 잎: IoU 가 기준을 넘는 잎 쌍에 포함된 잎의 수
    OVERLAP_IOU = 0.12
    overlapping = set()
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            if _iou(leaves[i], leaves[j]) > OVERLAP_IOU:
                overlapping.add(i)
                overlapping.add(j)
    overlap_count = len(overlapping)

    # 새순: 'newleaf' 클래스가 있거나, 유난히 작은 잎(면적이 중앙값의 40% 미만)
    new_shoot = any(b["cls"] in ("newleaf", "new_shoot", "new-shoot") for b in boxes)
    if not new_shoot and leaves:
        areas = sorted(b["area"] for b in leaves)
        median = areas[len(areas) // 2]
        new_shoot = any(b["area"] < 0.4 * median for b in leaves)

    # 크기 분류: 잎이 차지하는 면적 비율 + 잎 개수
    coverage = sum(b["area"] for b in leaves) / img_area if img_area else 0
    if coverage > 0.45 or leaf_count >= 9:
        size_class = "대품"
    elif coverage > 0.2 or leaf_count >= 5:
        size_class = "중품"
    else:
        size_class = "소품"

    return {
        "size_class": size_class,
        "leaf_count": leaf_count,
        "overlap_count": overlap_count,
        "new_shoot": new_shoot,
        "coverage": round(coverage, 3),
    }


# ===========================================================================
# 가상 IoT 센서 (온도, 빛 효율)
# ===========================================================================
def read_sensors(plant: dict) -> dict:
    # 구역별로 약간 다른 경향 + 약간의 랜덤
    base_temp = {"A": 24, "B": 26, "C": 22}.get(plant["zone"], 24)
    temp = round(base_temp + random.uniform(-2.5, 3.5), 1)
    light_eff = max(20, min(98, int(random.gauss(70, 15))))
    return {"temp": temp, "light_eff": light_eff}


# ===========================================================================
# RAG 기반 생장 피드백
#   작은 지식베이스(규칙+문장)에서 현재 상태에 맞는 항목을 '검색'해 조합한다.
# ===========================================================================
KNOWLEDGE = [
    (lambda s: s["new_shoot"],
     "🌱 새순이 확인됩니다. 생장이 활발한 시기이니 광량과 수분을 안정적으로 유지하세요."),
    (lambda s: s["leaf_count"] > 0 and s["overlap_count"] / max(1, s["leaf_count"]) >= 0.4,
     "🍃 잎 겹침이 많아 통풍·광 투과가 떨어질 수 있어요. 잎을 솎거나 화분 간격을 넓혀 주세요."),
    (lambda s: s["light_eff"] is not None and s["light_eff"] < 50,
     "💡 빛 효율이 낮습니다. 조명에 더 가깝게 두거나 조명 각도를 조정하세요."),
    (lambda s: s["light_eff"] is not None and s["light_eff"] >= 80,
     "✅ 빛을 충분히 받고 있어요. 현재 배치를 유지하세요."),
    (lambda s: s["temp"] is not None and s["temp"] >= 28,
     "🌡️ 온도가 다소 높아요. 환기하거나 광원과의 거리를 늘려 열 스트레스를 줄이세요."),
    (lambda s: s["temp"] is not None and s["temp"] <= 19,
     "❄️ 온도가 낮은 편이에요. 보온에 신경 쓰세요."),
    (lambda s: s["size_class"] == "대품",
     "🪴 대품으로 성장했습니다. 분갈이나 지지대를 고려해 보세요."),
    (lambda s: s["size_class"] == "소품",
     "🌿 아직 소품 단계예요. 꾸준한 광·수분 관리로 생장을 도와주세요."),
]


def rag_feedback(state: dict) -> str:
    hits = [msg for cond, msg in KNOWLEDGE if cond(state)]
    if not hits:
        return "특이사항 없이 안정적으로 자라고 있어요."
    return " ".join(hits[:3])


# ===========================================================================
# 엔드포인트
# ===========================================================================
@app.get("/")
def index():
    return FileResponse(os.path.join("static", "index.html"))


@app.get("/api/plants")
def get_plants():
    """3D 온실이 초기 배치·현재 상태를 그리기 위해 호출."""
    return {"engine": ENGINE, "zones": ZONES, "plants": list(PLANTS.values())}


@app.get("/api/plants/{plant_id}")
def get_plant(plant_id: str):
    if plant_id not in PLANTS:
        raise HTTPException(404, "존재하지 않는 식물 ID")
    return PLANTS[plant_id]


@app.post("/api/analyze")
async def analyze(plant_id: str = Form(...), file: UploadFile = File(...)):
    if plant_id not in PLANTS:
        raise HTTPException(404, f"존재하지 않는 식물 ID: {plant_id}")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드할 수 있어요.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 읽을 수 없어요.")

    # 1) YOLO 분석
    boxes, img_area = detect_boxes(image)
    metrics = analyze_metrics(boxes, img_area)

    # 2) 가상 센서
    plant = PLANTS[plant_id]
    sensors = read_sensors(plant)

    # 3) 상태 가공·저장
    plant.update(metrics)
    plant.update(sensors)
    plant["analyzed"] = True
    plant["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # 4) RAG 피드백
    plant["feedback"] = rag_feedback(plant)

    # 썸네일(작게)
    thumb = image.copy()
    thumb.thumbnail((240, 240))
    tb = io.BytesIO()
    thumb.save(tb, format="JPEG", quality=70)
    plant["thumb"] = "data:image/jpeg;base64," + base64.b64encode(tb.getvalue()).decode()

    return plant


app.mount("/static", StaticFiles(directory="static"), name="static")
