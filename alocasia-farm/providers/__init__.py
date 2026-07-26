"""탐지 제공자(provider) — 잎을 찾아 주는 쪽을 갈아 끼울 수 있게 분리한 층.

앱의 나머지 코드는 "누가 탐지하는지" 몰라도 되고, 아래 한 가지만 알면 된다:

    detector.detect(image) -> (boxes_px, img_area_px)

`boxes_px` 는 **이미지 픽셀 좌표**의 박스 목록:
    {"cls": 클래스명, "conf": 0~1, "x1","y1","x2","y2": px, "area": px²}

로보플로우를 걷어내고 로컬 모델(HuggingFace·SAM 등)로 바꿀 때도
이 파일에 제공자 하나를 더 붙이면 되고, 위쪽 코드는 건드릴 필요가 없다.
"""

import os
from typing import List, Protocol, Tuple

from PIL import Image

from .common import boxes_from_predictions, jpeg_b64          # noqa: F401  (제공자들이 씀)

CONFIDENCE = float(os.environ.get("CONFIDENCE", "25"))        # 0~100


class Detector(Protocol):
    """탐지 제공자가 지켜야 할 최소 약속."""

    name: str

    def detect(self, image: Image.Image) -> Tuple[List[dict], float]:
        """이미지 → (픽셀 좌표 박스 목록, 이미지 면적 px²)"""
        ...


def select() -> Tuple[Detector, Detector]:
    """환경변수를 보고 제공자를 고른다. → (모델1용, 모델2용)

    둘이 같은 객체면 위쪽 코드가 추론을 한 번만 돌려 크레딧을 아낀다.
    우선순위: 로보플로우 워크플로 → 로보플로우 모델 → 로컬 .pt → 데모
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "")
    workflow_id = os.environ.get("ROBOFLOW_WORKFLOW_ID", "")
    workflow_url = os.environ.get("ROBOFLOW_WORKFLOW_URL", "")
    model_path = os.environ.get("MODEL_PATH", "yolov8n.pt")

    if api_key and (workflow_url or (workspace and workflow_id)):
        from .roboflow_workflow import WorkflowDetector
        one = WorkflowDetector(api_key, workspace, workflow_id, workflow_url)
        return one, one                      # 워크플로는 한 번에 다 내주므로 재사용

    if api_key:
        from .roboflow_model import ModelDetector
        default_id = os.environ.get("ROBOFLOW_MODEL_ID", "find-leaf-and-object/1")
        top_id = os.environ.get("ROBOFLOW_MODEL_TOP", default_id)
        stage_id = os.environ.get("ROBOFLOW_MODEL_STAGE", default_id)
        top = ModelDetector(api_key, top_id)
        # 모델 이름이 같으면 같은 객체를 돌려줘서 호출을 한 번만 하게 한다
        return top, (top if stage_id == top_id else ModelDetector(api_key, stage_id))

    if os.path.exists(model_path) or os.environ.get("USE_LOCAL"):
        from .local_yolo import LocalYoloDetector
        try:
            one = LocalYoloDetector(model_path)
            return one, one
        except Exception as e:
            print(f"[경고] 로컬 모델 로딩 실패({e}) → 데모 모드")

    from .demo import DemoDetector
    one = DemoDetector()
    return one, one
