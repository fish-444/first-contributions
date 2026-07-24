"""
잎(leaf) 탐지 웹앱 - FastAPI 백엔드
=====================================

이 파일은 서버 프로그램입니다.
브라우저에서 보낸 사진을 받아 YOLO 모델로 잎을 탐지하고,
탐지한 위치에 박스를 그려서 다시 브라우저로 돌려줍니다.

실행 방법은 같은 폴더의 README.md 파일을 참고하세요.
"""

import base64
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 1) 사용할 YOLO weight(가중치) 파일 설정
# ---------------------------------------------------------------------------
# - 직접 학습시킨 weight 파일이 있다면, 그 파일 경로를 아래 MODEL_PATH 에 넣으세요.
#   예) MODEL_PATH = "/Users/내이름/Desktop/best.pt"
#
# - 없다면 그대로 두세요. 아래 기본값("yolov8n.pt")은 인터넷에서 자동으로
#   내려받는 기본 모델이라 바로 테스트해 볼 수 있습니다.
#   (기본 모델은 잎 전용이 아니라 사람/자동차 등 일반 사물을 탐지합니다.
#    잎을 제대로 탐지하려면 직접 학습시킨 weight 파일이 필요합니다.)
#
# - 환경변수 MODEL_PATH 로도 바꿀 수 있습니다. (README 참고)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")

# 탐지 신뢰도(confidence) 기준값. 0.0 ~ 1.0 사이.
# 값이 낮을수록 더 많이(하지만 부정확하게) 탐지하고,
# 높을수록 확실한 것만 탐지합니다.
CONFIDENCE = float(os.environ.get("CONFIDENCE", "0.25"))

# 서버가 켜질 때 모델을 한 번만 불러와 재사용합니다(속도를 위해).
print(f"[모델 로딩 중] {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print("[모델 로딩 완료]")

app = FastAPI(title="Leaf Detection Web App")


@app.get("/")
def index():
    """홈페이지(업로드 화면)를 보여줍니다."""
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    브라우저가 업로드한 사진 한 장을 받아서:
      1. YOLO 모델로 잎을 탐지하고
      2. 탐지된 위치에 박스를 그린 뒤
      3. 박스가 그려진 이미지와 탐지 결과 목록을 돌려줍니다.
    """
    # 업로드된 것이 이미지가 맞는지 확인
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있어요.")

    # 업로드된 파일을 읽어서 이미지로 변환
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없어요. 다른 파일로 시도해 보세요.")

    # YOLO 로 탐지 실행
    results = model.predict(image, conf=CONFIDENCE, verbose=False)
    result = results[0]

    # 탐지된 박스들을 그린 이미지를 만든다.
    # result.plot() 은 박스가 그려진 그림을 numpy 배열(BGR 순서)로 돌려줍니다.
    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]  # BGR -> RGB 로 순서를 바꿔줌
    annotated = Image.fromarray(plotted_rgb)

    # 그림을 PNG 로 바꾼 뒤 base64 문자열로 만들어 브라우저에 보냅니다.
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 무엇을 몇 개 찾았는지 목록으로 정리
    detections = []
    names = result.names  # 예: {0: 'leaf', ...}
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "label": names.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf[0]), 3),
                }
            )

    return {
        "count": len(detections),
        "detections": detections,
        "image": f"data:image/png;base64,{image_b64}",
    }


# 정적 파일(그림, CSS 등)이 필요하면 /static 아래에서 제공
app.mount("/static", StaticFiles(directory="static"), name="static")
