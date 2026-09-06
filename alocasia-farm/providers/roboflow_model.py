"""로보플로우 모델 엔드포인트 탐지기."""

from typing import List, Tuple

from fastapi import HTTPException
from PIL import Image

from . import CONFIDENCE
from .common import boxes_from_predictions, jpeg_b64

API_URL = __import__("os").environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")


class ModelDetector:
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.name = f"roboflow · {model_id}"

    def detect(self, image: Image.Image) -> Tuple[List[dict], float]:
        import requests
        try:
            resp = requests.post(
                f"{API_URL}/{self.model_id}",
                params={"api_key": self.api_key, "confidence": CONFIDENCE},
                data=jpeg_b64(image),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60)
        except requests.RequestException:
            raise HTTPException(502, "로보플로우 서버 연결 실패 (인터넷 확인)")
        if resp.status_code in (401, 403):
            raise HTTPException(401, "로보플로우 개인(Private) API 키를 확인하세요.")
        if not resp.ok:
            raise HTTPException(502, f"로보플로우 오류: {resp.text[:150]}")
        preds = resp.json().get("predictions", [])
        return boxes_from_predictions(preds), float(image.width * image.height)
