"""
잎(leaf) 탐지 웹앱 - FastAPI 백엔드
=====================================

이 파일은 서버 프로그램입니다.
브라우저에서 보낸 사진을 받아 YOLO 모델로 잎을 탐지하고,
탐지한 위치에 박스를 그려서 다시 브라우저로 돌려줍니다.

두 가지 방식(엔진)을 지원합니다:
  1) 로보플로우(Roboflow) 방식  - 로보플로우 서버에 사진을 보내 탐지 (API 키 필요)
  2) 로컬(local) 방식           - 내 컴퓨터에서 ultralytics YOLO 로 직접 탐지 (.pt 파일)

ROBOFLOW_API_KEY 환경변수를 넣어주면 자동으로 1) 로보플로우 방식으로 동작하고,
없으면 2) 로컬 방식으로 동작합니다.

실행 방법은 같은 폴더의 README.md 파일을 참고하세요.
"""

import base64
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

# ===========================================================================
# 설정
# ===========================================================================

# --- 로보플로우(Roboflow) 설정 ------------------------------------------------
# API 키는 비밀번호 같은 것이라 코드에 직접 쓰지 않고 "환경변수"로 넣습니다.
# (넣는 방법은 README 의 "로보플로우 연결하기" 참고)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# 사용할 로보플로우 모델. 형식은 "프로젝트이름/버전번호" 입니다.
# 예) find-leaf-mcfh8/1  (프로젝트 find-leaf-mcfh8 의 버전 1)
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "find-leaf-mcfh8/1")

# 로보플로우 탐지 서버 주소 (보통 바꿀 필요 없음)
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://detect.roboflow.com")

# --- 로컬(local) YOLO 설정 ---------------------------------------------------
# 로보플로우를 쓰지 않을 때만 사용됩니다.
# 직접 학습시킨 .pt 파일이 있으면 그 경로를 넣으세요. 없으면 기본 모델을 씁니다.
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")

# --- 공통 -------------------------------------------------------------------
# 탐지 신뢰도(confidence) 기준값(%). 0 ~ 100. 낮을수록 더 많이(부정확하게) 탐지.
CONFIDENCE = float(os.environ.get("CONFIDENCE", "25"))

# ROBOFLOW_API_KEY 가 있으면 로보플로우 방식, 없으면 로컬 방식
USE_ROBOFLOW = bool(ROBOFLOW_API_KEY)

# ===========================================================================
# 시작 시 준비
# ===========================================================================
local_model = None  # 로컬 방식일 때만 채워짐

if USE_ROBOFLOW:
    print(f"[로보플로우 방식] 모델: {ROBOFLOW_MODEL_ID}")
else:
    # 로컬 방식일 때만 무거운 ultralytics 를 불러옵니다.
    from ultralytics import YOLO

    print(f"[로컬 방식] 모델 로딩 중: {MODEL_PATH} ...")
    local_model = YOLO(MODEL_PATH)
    print("[로컬 방식] 모델 로딩 완료")

app = FastAPI(title="Leaf Detection Web App")


# ===========================================================================
# 도우미 함수
# ===========================================================================
def _pil_to_data_url(image: Image.Image) -> str:
    """PIL 이미지를 브라우저가 바로 보여줄 수 있는 문자열(data URL)로 변환."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def detect_with_roboflow(image: Image.Image):
    """로보플로우 서버로 사진을 보내 탐지한 뒤, 박스를 직접 그려서 돌려줍니다."""
    import requests  # 로보플로우 방식일 때만 필요

    # 사진을 JPEG -> base64 로 만들어 전송
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    try:
        resp = requests.post(
            f"{ROBOFLOW_API_URL}/{ROBOFLOW_MODEL_ID}",
            params={"api_key": ROBOFLOW_API_KEY, "confidence": CONFIDENCE},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="로보플로우 서버에 연결하지 못했어요. 인터넷 연결을 확인해 주세요.",
        )

    if resp.status_code == 401 or resp.status_code == 403:
        raise HTTPException(
            status_code=401,
            detail="로보플로우 API 키가 올바르지 않아요. 키를 다시 확인해 주세요.",
        )
    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"로보플로우 모델을 찾지 못했어요. 모델 이름/버전을 확인하세요: {ROBOFLOW_MODEL_ID}",
        )
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"로보플로우 오류: {resp.text[:200]}")

    predictions = resp.json().get("predictions", [])

    # 원본 위에 박스를 직접 그립니다.
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    detections = []
    for p in predictions:
        # 로보플로우는 박스의 "중심 좌표(x, y)"와 너비/높이를 줍니다.
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        left, top, right, bottom = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        label = p.get("class", "object")
        conf = float(p.get("confidence", 0))

        draw.rectangle([left, top, right, bottom], outline=(230, 30, 30), width=3)
        caption = f"{label} {conf * 100:.0f}%"
        draw.text((left + 2, max(top - 12, 0)), caption, fill=(230, 30, 30), font=font)

        detections.append({"label": label, "confidence": round(conf, 3)})

    return annotated, detections


def detect_with_local(image: Image.Image):
    """내 컴퓨터의 ultralytics YOLO 로 직접 탐지한 뒤, 박스를 그려서 돌려줍니다."""
    results = local_model.predict(image, conf=CONFIDENCE / 100.0, verbose=False)
    result = results[0]

    # result.plot() 은 박스가 그려진 그림을 numpy 배열(BGR 순서)로 돌려줍니다.
    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]  # BGR -> RGB
    annotated = Image.fromarray(plotted_rgb)

    detections = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "label": names.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf[0]), 3),
                }
            )
    return annotated, detections


# ===========================================================================
# 웹 주소(엔드포인트)
# ===========================================================================
@app.get("/")
def index():
    """홈페이지(업로드 화면)를 보여줍니다."""
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    브라우저가 업로드한 사진 한 장을 받아서:
      1. (로보플로우 또는 로컬) YOLO 로 잎을 탐지하고
      2. 탐지된 위치에 박스를 그린 뒤
      3. 박스가 그려진 이미지와 탐지 결과 목록을 돌려줍니다.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있어요.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없어요. 다른 파일로 시도해 보세요.")

    if USE_ROBOFLOW:
        annotated, detections = detect_with_roboflow(image)
    else:
        annotated, detections = detect_with_local(image)

    return {
        "count": len(detections),
        "detections": detections,
        "image": _pil_to_data_url(annotated),
    }


# 정적 파일(그림, CSS 등)이 필요하면 /static 아래에서 제공
app.mount("/static", StaticFiles(directory="static"), name="static")
