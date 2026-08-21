"""모돈 행동 15종 인스턴스 분할 — 프로그램 접목용 패키지.

이것은 **행동 분할** 모델이다. 분만 판정 모델이 아니다.
Parturition 클래스가 목록에 있으나 학습 데이터에 사실상 1건뿐이고
Coughing 은 0건이라, 두 클래스의 출력은 신뢰할 수 없다.
분만 판정은 이 모델 위에 얹는 별도 층이다 (README 참조).

사용법:
    from pig_behavior import PigBehaviorPredictor

    p = PigBehaviorPredictor(checkpoint='pig_polygon_epoch12.pth', device='cpu')
    for det in p.predict('frame.jpg'):
        print(det.label, det.score, det.bbox)
"""

from .predictor import CLASSES, Detection, PigBehaviorPredictor, RELIABLE_CLASSES

__all__ = ['PigBehaviorPredictor', 'Detection', 'CLASSES', 'RELIABLE_CLASSES']
__version__ = '0.1.0'
