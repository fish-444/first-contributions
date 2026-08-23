"""업로드된 행동 분할 모델을 `vision_contract` 에 꽂는 어댑터.

`pig_behavior`(모돈 행동 15종 인스턴스 분할, Mask R-CNN)가 **첫 실모델**이다.
계약이 요구하는 것은 `predict(frames, tracks) → BehaviorObs` 하나이므로
여기서 그 모양만 맞춘다 — 모델 코드도 계약도 고치지 않는다.

## 어휘를 15종이 아니라 4종으로 신고한다

모델은 15종을 출력하지만 홀드아웃 200장에서 **AP 0.2 를 넘은 것은 4종**뿐이다
(Resting 0.633 · Eating 0.631 · Walking 0.261 · Searching 0.242). Scrubbing 은
AP 0.0, Coughing 은 학습 표본 0 이다.

15종을 그대로 계약에 신고하면 `head_support()` 가 "분만징후 헤드가 돈다"
고 답한다 — Scrubbing 이 어휘에 *있으니까*. 그러나 그 출력은 근거가 없다.
**어휘 신고는 "낼 수 있는 것"이 아니라 "근거를 갖고 낼 수 있는 것"이어야
한다.** 그래서 `classes` 는 RELIABLE_CLASSES 만 신고하고, 그 결과로:

    발정·재발   ✅  Walking 이 활동량 원신호를 보강한다
    질병(섭식)  ✅  Eating·Resting — 섭식 급감은 질병의 앞선 신호다
    분만징후    ❌  둥지짓기 반쪽(Searching)만 있고 Scrubbing 이 없다
    질병(기침)  ❌  Coughing 학습 표본 0

분만징후가 막히는 것이 이 어댑터의 가장 중요한 출력이다 — **무엇을 더
학습해야 하는지**(Scrubbing 표본)를 등록 시점에 이름으로 말해 준다.

## 기준선에 대해

계약의 행동 10클래스 기준선 0.485 는 케글 시계열 분류 과제의 값이라 이
모델(정지영상 인스턴스 분할, AI Hub 622)과 **다른 자다.** 섞어 비교하지
않는다. 이 모델의 등록 근거는 자체 홀드아웃 AP 표이고, 원 학습이 보고한
0.953 은 train==val 이라 인용하지 않는다 — 그 판단은 `pig_behavior.predictor`
도입부에 있다.

    python competition/src/vision_pig_behavior.py            # 접목 상태 점검
    python competition/src/vision_pig_behavior.py --ckpt pig_polygon_epoch12.pth 프레임들/
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vision_contract as vc                                   # noqa: E402
from pig_behavior.predictor import (CLASSES, Detection,        # noqa: E402
                                    RELIABLE_CLASSES)

# 계약에 신고하는 어휘 — CLASSES 순서를 유지한 신뢰 4종
CONTRACT_CLASSES = tuple(c for c in CLASSES if c in RELIABLE_CLASSES)

# 홀드아웃 근거(처음 보는 200장, AI Hub 622 ts06). 어댑터가 들고 다닌다 —
# 응답만 보고도 "이 어휘의 근거가 뭐냐"에 답할 수 있어야 한다.
HOLDOUT_AP = {"Resting": 0.633, "Eating": 0.631,
              "Walking": 0.261, "Searching": 0.242}
HOLDOUT_NOTE = ("bbox mAP 0.205 · 처음 보는 200장(AI Hub 622 ts06). "
                "원 학습의 0.953 은 train==val 이라 인용하지 않는다.")


def fold(dets_by_time: list, camera_id: str, barn: str, pen: str,
         model: str, activity_px: float = 0.0,
         resp_bpm: float | None = None) -> list:
    """프레임별 검출 → 시간창 하나의 `BehaviorObs` (방 단위).

    `dets_by_time` 은 `[(iso시각, [Detection, ...]), ...]`.

    - **방 단위다.** 인스턴스 분할이 개체를 프레임 안에서는 가르지만
      프레임 사이로 잇지는 못한다(그건 추적기 몫, ID 일관성 0.77). 거짓
      확신 대신 계약이 이미 허용하는 `animal_id=None` 으로 내려간다.
    - **신뢰 4종만 분포에 넣는다.** 나머지 검출은 세되(`n_dropped`)
      확률에는 안 넣는다 — Scrubbing AP 0.0 이 분포에 섞이면 분만징후
      헤드가 근거 없는 수를 먹는다.
    - 분포는 점수 가중 구성비다. "이 시간창에 이 방의 행동이 어떻게
      구성돼 있었나"이지 개체 하나의 상태가 아니다.
    """
    if not dets_by_time:
        return []
    t0, t1 = dets_by_time[0][0], dets_by_time[-1][0]
    weight: dict = defaultdict(float)
    n_all = n_used = 0
    for _t, dets in dets_by_time:
        for d in dets:
            n_all += 1
            if d.label in RELIABLE_CLASSES:
                weight[d.label] += float(d.score)
                n_used += 1
    tot = sum(weight.values())
    probs = ({k: round(v / tot, 4) for k, v in weight.items()} if tot else {})
    obs = vc.BehaviorObs(camera_id=camera_id, barn=barn, pen=pen,
                         t0=str(t0), t1=str(t1), track_id=None,
                         animal_id=None, probs=probs,
                         activity_px=float(activity_px), resp_bpm=resp_bpm,
                         model=model)
    # BehaviorObs 는 frozen dataclass 라 부가정보는 따로 낸다
    return [{"obs": obs, "n_detections": n_all, "n_used": n_used,
             "n_dropped": n_all - n_used}]


class PigBehaviorModel:
    """`vision_contract.BehaviorModel` 구현 — 첫 실모델.

    무거운 초기화(mmdet·torch)는 **첫 predict 까지 미룬다.** 접목 상태
    점검(`head_support` 등)은 가중치 파일 없이도 돌아야 한다 — `*.pth` 는
    커밋하지 않으므로 이 저장소를 받은 사람 대부분이 그 상태다.
    """

    def __init__(self, checkpoint: str | None = None,
                 onnx: str | None = None, device: str = "cpu",
                 score_thr: float = 0.3):
        base = os.path.basename(checkpoint or onnx or "")
        self.version = f"pig-behavior-0.1.0/{base or '가중치 미지정'}"
        self.classes = CONTRACT_CLASSES
        self.holdout = {"ap": dict(HOLDOUT_AP), "note": HOLDOUT_NOTE}
        self._args = dict(checkpoint=checkpoint, onnx=onnx,
                          device=device, score_thr=score_thr)
        self._pred = None

    def _ensure(self):
        if self._pred is None:
            from pig_behavior.predictor import PigBehaviorPredictor

            self._pred = PigBehaviorPredictor(**self._args)
        return self._pred

    def predict(self, frames, tracks=None) -> list:
        """`frames` = `[(iso시각, 이미지), ...]` → 시간창 하나의 관측.

        `tracks` 는 받아만 둔다 — 이 모델은 프레임 단위라 추적을 모르고,
        활동량(`activity_px`)은 계약대로 추적기가 따로 낸다.
        """
        p = self._ensure()
        dets_by_time = [(t, p.predict(im, with_mask=False))
                        for t, im in frames]
        return fold(dets_by_time, camera_id="", barn="", pen="",
                    model=self.version)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="vision_pig_behavior")
    ap.add_argument("--ckpt", help="pig_polygon_epoch12.pth (없으면 점검만)")
    a = ap.parse_args(argv)

    m = PigBehaviorModel(checkpoint=a.ckpt)
    print("=" * 72)
    print(f"  행동 분할 모델 접목 점검 — {m.version}")
    print("=" * 72)
    print(f"  모델 출력 {len(CLASSES)}종 중 계약 신고 {len(m.classes)}종 "
          f"(홀드아웃 AP 0.2 이상만)")
    for c in m.classes:
        print(f"    {c:<10} AP {HOLDOUT_AP[c]:.3f}")
    print(f"  ⚠ {HOLDOUT_NOTE}\n")
    print("  이 어휘로 어느 헤드가 도는가:")
    for h, s in vc.head_support(m).items():
        mark = "✅" if s["runs"] else "❌"
        print(f"    {mark} {s['kr']:<6} {s['why']}")
    print("\n  분만징후가 막히는 것이 요점이다 — 다음 학습이 무엇을 채워야"
          "\n  하는지(Scrubbing 표본)를 등록 시점에 이름으로 말해 준다.")
    if not a.ckpt:
        print("\n  가중치 미지정 — 추론 없이 접목 상태만 점검했다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
