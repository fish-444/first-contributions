"""영상 모델이 꽂힐 자리 — 첫 실모델이 꽂혔다.

행동 분류 모델을 만드는 중이라, 여기는 완성된 모델이 들어올 계약과
**오늘 어느 개체를 볼지** 까지만 낸다. 산술은 `vision_contract` 에 있고
여기는 얇은 어댑터다 — 다른 라우터와 같은 규칙이다.

## 왜 겨냥이 먼저인가

모델이 전 개체를 24시간 보는 구조가 아니다. 분만 예정일을 알면 분만징후를
그 며칠에만 찾으면 되고, 이유일을 알면 발정을 그 창에서만 찾으면 된다.
**사전확률을 번식 달력이 공짜로 준다.** 창은 전부 기존 모듈에서 오고
(`pregnancy_check` 3주 관문 · `estrus_early_warning` 지평), 여기서 새 임계값을
만들지 않는다.

## 무엇을 주장하지 않는가

**판정하지 않는다.** 이 응답은 "이 개체를 오늘 보라" 까지이고, 분만이
임박했는지 발정이 왔는지는 말하지 않는다 — 그건 모델의 몫이고 모델이 아직
없다. 지금 붙일 수 있는 구현은 `ReplayModel` 뿐이라 응답에 그렇게 적는다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import vision_contract as vc                                   # noqa: E402

from ..schemas import HerdIn

router = APIRouter(prefix="/api/vision", tags=["vision"])


@router.get("/contract", summary="모델이 꽂힐 자리 — 계약과 헤드 넷")
def contract() -> dict:
    """**모델을 만드는 쪽이 읽는 문서다.** 어휘와 창을 여기서 확인한다."""
    return {
        "obs_fields": {
            "camera_id": "str", "barn": "str", "pen": "str",
            "t0": "ISO 로컬 시각", "t1": "ISO 로컬 시각",
            "track_id": "int | null — 영상 안에서만 유효",
            "animal_id": ("str | null — **null 을 허용한다.** 군사에서 트랙을 "
                          "며칠씩 끌고 갈 수 없다(ID 일관성 0.77). 거짓 확신 "
                          "대신 방 단위로 내려간다"),
            "probs": ("{클래스: 확률} — **argmax 가 아니라 분포다.** 행동 "
                      "10클래스 실측이 0.485 라, 확정 라벨로 넘기면 헤드가 "
                      "절반쯤 틀린 입력 위에 서고 오류가 곱해진다"),
            "activity_px": ("float — 모델과 **따로** 받는다. 활동/휴식은 "
                            "AUC 0.739 로 검증된 신호라 모델이 실패해도 "
                            "살아 있어야 하고, 새 모델의 기준선이기도 하다"),
            "model": "str — 버전 없는 관측은 받지 않는다",
        },
        "heads": [{"head": h, "kr": vc.HEAD_KR[h],
                   "needs": list(vc.HEAD_NEEDS[h]),
                   "helped_by": list(vc.HEAD_HELPS[h])} for h in vc.HEADS],
        "windows": {
            "estrus": {"from_wean": [vc.ESTRUS_WATCH_FROM, vc.ew.ANESTRUS_DAY],
                       "source": "estrus_early_warning.ANESTRUS_DAY"},
            "return": {"from_service": [vc.RETURN_FROM, vc.RETURN_TO],
                       "source": "pregnancy_check.CHECKPOINTS[0] — 3주 관문",
                       "why": ("CCTV 를 붙이면 이 관문 민감도가 0.70 → 0.92 "
                               "로 오른다는 것이 기존 측정이다")},
            "farrowing": {"from_expected": [-vc.fr.PRE_FARROW, 0],
                          "source": "farm_registry.stage_of — 분만사 pre_farrow",
                          "why": "임박 판정은 하지 않는다 — 모델의 몫이다"},
            "disease": {"source": None, "why": "달력이 없다 — 상시 관찰"},
        },
        "baseline": {
            "behavior_10cls": 0.485,
            "why": ("손피처 RF 실측. 새 모델은 같은 검증(개체 분리)으로 이걸 "
                    "넘는 것이 등록 조건이고, 못 넘으면 기준선이 그대로 "
                    "돈다 — 1D-CNN 이 0.427 로 진 적이 있다"),
        },
        "implemented": _implemented(),
        "note": ("판정은 모델이 하고 이 응답은 계약과 겨냥까지다. 꽂힌 "
                 "구현의 근거와 한계는 `implemented` 각 항목이 들고 있다 — "
                 "가중치 파일(`*.pth`)은 커밋하지 않으므로 실제 추론은 "
                 "파일이 있는 곳에서만 돈다."),
    }


def _implemented() -> list:
    """꽂혀 있는 구현들 — **각자 근거와 한계를 들고 다닌다.**

    이름만 나열하면 "모델이 있다" 로 읽힌다. 스텁은 스텁이라고, 실모델은
    15종 중 4종만 신뢰 가능하고 분만징후 헤드는 못 돈다고, 응답 자체가
    말해야 한다.
    """
    import vision_pig_behavior as vpb

    m = vpb.PigBehaviorModel()
    return [
        {"name": "ReplayModel", "kind": "스텁",
         "why": "배선 시험용 — 모델이 아니다. 미리 정한 분포를 되돌려 준다"},
        {"name": "PigBehaviorModel", "kind": "실모델(행동 분할)",
         "classes_out": len(vpb.CLASSES),
         "classes_contract": list(m.classes),
         "holdout": m.holdout,
         "heads": {h: s["runs"] for h, s in vc.head_support(m).items()},
         "why": ("출력 15종 중 홀드아웃 AP 0.2 이상 4종만 계약에 신고한다. "
                 "분만징후는 Scrubbing(AP 0.0) 이, 기침 질병은 Coughing"
                 "(학습 표본 0) 이 없어 못 돈다 — 다음 학습이 채울 것의 "
                 "이름이다")},
    ]


@router.post("/targets", summary="오늘 어느 개체를 어느 헤드로 볼 것인가")
def targets(body: HerdIn) -> dict:
    """개체 이력 + 기준일 → 헤드별 관찰 대상.

    기준일은 **파일에 적힌 것**을 쓴다. 스냅숏이라 다른 날짜로 읽으면 개체가
    통째로 빠지는데, 그건 `run_farm --herd` 가 이미 겪고 막아 둔 사고다.
    """
    if not body.records:
        raise HTTPException(422, "개체 이력이 비었다 — 겨냥할 대상이 없다")
    return vc.targets(body.records, body.as_of,
                      disease_all=body.include_disease)
