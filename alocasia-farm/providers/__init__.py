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

CONFIDENCE = float(os.environ.get("CONFIDENCE", "25"))        # 0~100


def clean_api_key(raw: str) -> str:
    """환경변수로 들어온 키에서 따옴표·공백을 걷어내고 돌려준다.

    윈도우 배치 파일에서 키를 넣다가 걸리는 함정이 늘 같다. 파워셸 습관대로
    `set ROBOFLOW_API_KEY="abcd..."` 라고 쓰면 **따옴표까지 키 값이 된다**.
    복사·붙여넣기하면 줄 끝 공백도 자주 딸려온다. 둘 다 401 로만 돌아와서
    키가 죽은 줄 알고 새로 발급받게 되는데, 실제로는 글자 두 개 문제다.
    그래서 조용히 걷어낸다.

    무엇을 걷어냈는지 알리는 일은 paste_warnings() 가 따로 한다. 원래는
    (키, 경고목록) 을 한 번에 돌려줬는데, 그러면 경고를 찍는 쪽이 '키가 섞여
    있을 수 있는 값' 을 print 하는 모양이 된다 — 코드 스캐닝이 정확히 그 줄을
    비밀값 평문 로깅으로 잡았다(py/clear-text-logging-sensitive-data). 경고
    문구에는 키가 한 글자도 안 들어가므로, 애초에 키와 다른 경로로 내보낸다.

    이 함수는 이름에 key 가 들어가도 된다 — 실제로 비밀값을 돌려주니까.
    """
    key = (raw or "").strip()
    for q in ('"', "'"):
        if len(key) >= 2 and key.startswith(q) and key.endswith(q):
            key = key[1:-1].strip()
    return key


def paste_warnings(raw: str) -> List[str]:
    """붙여넣다 뭐가 섞였는지 사람이 읽을 문구로 돌려준다 — 키 자체는 담지 않는다.

    돌려주는 문구는 전부 고정 문장이다. 키에서 흘러 들어오는 건 '길이가 줄었나',
    'rf_ 로 시작하나' 같은 판정 결과뿐이라 로그에 비밀값이 새지 않는다.

    이름에 key 를 넣지 않은 건 우연이 아니다. 코드 스캐닝은 이름에 key 가 든
    함수의 반환값을 곧 비밀값으로 보고, 그걸 print 하는 줄을 평문 로깅으로
    잡는다 — 실제로 api_key_warnings 라는 이름일 때 그렇게 잡혔다. 이 함수의
    존재 이유가 '비밀값을 절대 담지 않는다' 인데 이름이 그 반대를 주장하고
    있었던 것이니, 스캐너 쪽이 맞았다. 이름에 key/secret/token 을 다시 넣지 말 것.
    """
    raw = raw or ""
    trimmed = raw.strip()
    key = clean_api_key(raw)

    warnings: List[str] = []
    if trimmed != raw:
        warnings.append("키 앞뒤의 공백을 떼고 씁니다. farm_env.bat 에서도 지워 주세요.")
    if len(key) < len(trimmed):                   # 감싼 따옴표를 벗겼다
        warnings.append("키를 감싼 따옴표를 떼고 씁니다 — 배치 파일에서는 "
                        "따옴표까지 키 값이 됩니다. farm_env.bat 에서도 지워 주세요.")
    if key.startswith("rf_"):
        warnings.append("rf_ 로 시작하는 키는 공개(publishable) 키라 막힙니다. "
                        "Private API Key 를 넣으세요.")
    if any(c.isspace() for c in key):
        warnings.append("키 중간에 공백이 있습니다 — 붙여넣다 잘린 것 같습니다.")
    return warnings


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
    raw_key = os.environ.get("ROBOFLOW_API_KEY", "")
    api_key = clean_api_key(raw_key)
    for w in paste_warnings(raw_key):
        print(f"[키 경고] {w}")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "")
    workflow_id = os.environ.get("ROBOFLOW_WORKFLOW_ID", "")
    workflow_url = os.environ.get("ROBOFLOW_WORKFLOW_URL", "")
    model_path = os.environ.get("MODEL_PATH", "yolov8n.pt")

    # 앱이 키를 들고 있는지, 그 키가 온전한 모양인지만 찍는다.
    #
    # 원래는 앞 4자리를 찍어 대시보드 값과 눈으로 맞춰 보게 했다. 로보플로우
    # 대시보드도 같은 방식으로 표시하니 편했지만, 그건 비밀값의 일부를 그대로
    # 로그에 남기는 것이고 콘솔 출력은 캡처·붙여넣기로 쉽게 밖으로 나간다.
    # 그래서 앞자리는 지웠다 — 되살리지 말 것.
    #
    # 길이만으로도 실제로 겪었던 경우는 다 갈린다: 따옴표가 값에 섞이면 2자
    # 늘고, 붙여넣다 잘리면 짧아진다. '파일은 고쳤는데 앱까지 안 왔다' 쪽은
    # main.py 의 _report_env_file() 이 따로 짚어 준다.
    if api_key:
        print(f"[키] 있음 ({len(api_key)}자) — 401 이 계속 나면 farm_env.bat 의 "
              f"키를 로보플로우 대시보드 값과 다시 비교해 보세요")
    else:
        print("[키] 없음 — farm_env.bat 의 ROBOFLOW_API_KEY 가 비어 있습니다 "
              "(farm_env.example.bat 이 아니라 farm_env.bat 입니다)")

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
