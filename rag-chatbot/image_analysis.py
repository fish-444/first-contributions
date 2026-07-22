"""이미지 분석 모듈 — Replicate API로 YOLO-World(객체 탐지)와 SAM-2(분할)를 호출한다.

로컬 GPU/모델 설치 없이, Replicate 클라우드에 이미지를 보내 결과만 받아온다.
사용하려면 환경변수 REPLICATE_API_TOKEN 이 설정되어 있어야 한다.

  - detect_objects():   YOLO-World로 "무엇이 어디에 있는지"(라벨 + 박스) 탐지
  - segment_image():    SAM-2로 이미지 안의 개별 객체 영역을 마스크로 분리
  - analyze_image():    위 둘을 함께 실행하고, 검색용 자연어 설명까지 만들어 반환
"""

import io
import json
from collections import Counter

import replicate

# ---------------------------------------------------------------------------
# 모델 설정
# ---------------------------------------------------------------------------

YOLO_MODEL = "zsxkib/yolo-world"   # 오픈 어휘 객체 탐지 (원하는 클래스를 자유롭게 지정)
SAM_MODEL = "meta/sam-2"           # Segment Anything v2 (이미지 자동 마스크 생성)

# YOLO-World는 "무엇을 찾을지" 클래스 목록을 직접 줘야 한다(오픈 어휘).
# 아래는 일상적으로 많이 나오는 객체들의 기본값. 요청 시 커스텀 목록으로 덮어쓸 수 있다.
DEFAULT_CLASSES = (
    "person, face, hand, dog, cat, bird, car, bicycle, motorcycle, bus, truck, "
    "chair, table, sofa, bed, tv, laptop, keyboard, mouse, cell phone, book, "
    "bottle, cup, bowl, plate, food, backpack, bag, umbrella, clock, potted plant, "
    "tree, building, road, sky"
)

SCORE_THRESHOLD = 0.05   # 이 점수 미만의 탐지는 버림 (0~1)
MAX_BOXES = 100          # 표시할 최대 박스 수


def _as_url(value) -> str:
    """Replicate 출력(FileOutput 객체 또는 문자열)을 URL 문자열로 통일한다."""
    if value is None:
        return ""
    # 최신 replicate 클라이언트는 FileOutput 객체를 돌려주며 .url 속성을 가진다.
    url = getattr(value, "url", None)
    return url if isinstance(url, str) else str(value)


# ---------------------------------------------------------------------------
# YOLO-World: 객체 탐지
# ---------------------------------------------------------------------------


def detect_objects(image_bytes: bytes, class_names: str | None = None) -> list[dict]:
    """이미지에서 객체를 탐지해 [{cls, score, box}, ...] 형태로 반환한다."""
    output = replicate.run(
        YOLO_MODEL,
        input={
            "input_media": io.BytesIO(image_bytes),
            "class_names": class_names or DEFAULT_CLASSES,
            "max_num_boxes": MAX_BOXES,
            "score_thr": SCORE_THRESHOLD,
            "nms_thr": 0.5,
            "return_json": True,   # 이미지가 아니라 탐지 결과 JSON을 받는다
        },
    )

    # return_json=True 이면 output 은 JSON 문자열
    # 예: {"Det-0": {"x0":..,"y0":..,"x1":..,"y1":..,"score":..,"cls":"person"}, ...}
    raw = output if isinstance(output, str) else _as_url(output)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    detections = []
    for key, det in data.items():
        if not key.startswith("Det-"):
            continue
        detections.append(
            {
                "cls": det.get("cls", "unknown"),
                "score": round(float(det.get("score", 0)), 3),
                "box": [
                    round(float(det.get("x0", 0))),
                    round(float(det.get("y0", 0))),
                    round(float(det.get("x1", 0))),
                    round(float(det.get("y1", 0))),
                ],
            }
        )
    # 점수 높은 순으로 정렬
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ---------------------------------------------------------------------------
# SAM-2: 객체 영역 분할(마스크)
# ---------------------------------------------------------------------------


def segment_image(image_bytes: bytes) -> dict:
    """SAM-2로 이미지를 자동 분할한다. 마스크 이미지 URL들을 반환한다."""
    output = replicate.run(
        SAM_MODEL,
        input={
            "image": io.BytesIO(image_bytes),
            "points_per_side": 32,   # 촘촘할수록 작은 객체까지 잡지만 느려짐
            "use_m2m": True,         # 마스크 품질 향상
        },
    )

    # 출력 예: {"combined_mask": <uri>, "individual_masks": [<uri>, ...]}
    if isinstance(output, dict):
        combined = _as_url(output.get("combined_mask"))
        individual = [_as_url(m) for m in output.get("individual_masks", [])]
    else:
        # 모델이 단일 결과만 주는 경우 대비
        combined = _as_url(output)
        individual = []

    return {
        "combined_mask": combined,
        "individual_masks": individual,
        "mask_count": len(individual),
    }


# ---------------------------------------------------------------------------
# 통합: 탐지 + 분할 + 자연어 설명 생성
# ---------------------------------------------------------------------------


def build_description(filename: str, detections: list[dict], mask_count: int) -> str:
    """탐지 결과를 사람이(그리고 검색엔진이) 이해할 문장으로 요약한다."""
    if not detections:
        return f"이미지 '{filename}'에서 지정한 클래스의 객체를 탐지하지 못했습니다."

    counts = Counter(d["cls"] for d in detections)
    parts = [f"{cls} {n}개" for cls, n in counts.most_common()]
    summary = ", ".join(parts)
    return (
        f"이미지 '{filename}' 분석 결과: 총 {len(detections)}개의 객체가 탐지되었습니다. "
        f"구성: {summary}. "
        f"SAM-2로 {mask_count}개의 영역(마스크)이 분리되었습니다."
    )


def analyze_image(
    image_bytes: bytes, filename: str, class_names: str | None = None
) -> dict:
    """YOLO-World + SAM-2를 함께 실행하고 결과를 하나로 묶어 반환한다."""
    detections = detect_objects(image_bytes, class_names=class_names)
    segmentation = segment_image(image_bytes)
    description = build_description(
        filename, detections, segmentation["mask_count"]
    )
    return {
        "filename": filename,
        "detections": detections,
        "segmentation": segmentation,
        "description": description,
    }
