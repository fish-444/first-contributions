"""
알로카시아 스마트팜 3D 온실 - FastAPI 백엔드
=================================================

핵심 파이프라인만:
  [웹 UI] 사진 업로드 + 식물 이름 입력
      │  POST /api/plants  (name, file)
      ▼
  [FastAPI] Roboflow(YOLO) 로 분석 →
      · 크기 분류(대/중/소품)
      · 잎 개수
      · 겹침 밀도(overlap)
      · 새순 유무
      → 이름을 가진 식물로 등록/저장
      ▼
  [3D 온실] 화분으로 시각화, 클릭 시 이름+분석결과 모달

센서·RAG 없음. 오직 [사진+이름 → YOLO → 3D 반영 + 이름 커스텀].

분석 엔진 자동 선택:
  1) 로보플로우(ROBOFLOW_API_KEY 있으면)  2) 로컬 ultralytics(.pt)  3) 데모(모델 없어도 동작)
실행법은 README.md 참고.
"""

import base64
import io
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
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
CONFIDENCE = float(os.environ.get("CONFIDENCE", "25"))

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
        print(f"[경고] 로컬 모델 로딩 실패({e}) → 데모 모드")
        ENGINE = "demo"
print(f"[분석 엔진] {ENGINE}")

app = FastAPI(title="Alocasia Smart Farm")

# --------------------------------------------------------------------------- 상태
PLANTS: Dict[str, dict] = {}          # id -> 식물 상태
# 번호 붙은 자리(슬롯): 온실 60x40cm 을 촘촘한 격자로 분할
# 기본 A1~E10 (5줄 x 10칸 = 50자리), 한 칸 = 12 x 8 cm
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
    if ENGINE == "roboflow":
        return _detect_roboflow(image, model_id)
    if ENGINE == "local":
        return _detect_local(image)
    return _detect_demo(image)


def _detect_roboflow(image: Image.Image, model_id: str):
    import requests
    buf = io.BytesIO(); image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = requests.post(
            f"{ROBOFLOW_API_URL}/{model_id}",
            params={"api_key": ROBOFLOW_API_KEY, "confidence": CONFIDENCE},
            data=img_b64, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=60,
        )
    except requests.RequestException:
        raise HTTPException(502, "로보플로우 서버 연결 실패 (인터넷 확인)")
    if resp.status_code in (401, 403):
        raise HTTPException(401, "로보플로우 개인(Private) API 키를 확인하세요.")
    if not resp.ok:
        raise HTTPException(502, f"로보플로우 오류: {resp.text[:150]}")
    boxes = []
    for p in resp.json().get("predictions", []):
        w, h = p["width"], p["height"]
        boxes.append({"cls": p.get("class", "leaf"), "conf": float(p.get("confidence", 0)),
                      "x1": p["x"] - w / 2, "y1": p["y"] - h / 2,
                      "x2": p["x"] + w / 2, "y2": p["y"] + h / 2, "area": w * h})
    return boxes, image.width * image.height


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


def analyze_top(boxes: List[dict], img_area: float) -> dict:
    """모델1: 식물의 '맨 위 잎'(광합성 주력) 크기 → 3D 온실 반영용."""
    leaves = [b for b in boxes if b["cls"].lower() not in NON_LEAF]
    if not leaves:
        return {"top_leaf_present": False, "top_leaf_size": "없음", "top_leaf_pct": 0.0}
    top = min(leaves, key=lambda b: b["y1"])          # 사진에서 가장 위쪽 잎
    pct = round(top["area"] / img_area * 100, 1) if img_area else 0.0
    size = "대엽" if pct > 18 else ("중엽" if pct > 8 else "소엽")
    return {"top_leaf_present": True, "top_leaf_size": size, "top_leaf_pct": pct}


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

    new_shoot = counts["shoot"] > 0

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
            "overlap_density": overlap_density, "new_shoot": new_shoot}


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
    return {"engine": ENGINE, "plants": list(PLANTS.values())}


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
