"""요청·응답 모델.

## 비운 값을 살려 둔다

성적란은 `float | None` 이고 **None 을 중앙값으로 채우지 않는다.** 이
프로젝트에서 실제로 겪은 버그다 — 진단 기본값을 실측 중앙값으로 깔았더니
격차가 늘 +0.00 으로 찍혔다. 중앙값을 중앙값과 비교하고 있었던 것이다.
같은 이유로 `0` 과 `None` 도 구별한다(방당 면적 미입력 vs 0㎡).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# 축사 용도·사육 방식 어휘는 farm_registry 가 정본이다. 여기서 새 낱말을
# 만들면 서버가 받아 준 값을 도메인 모듈이 거절한다.
import farm_registry as fr  # noqa: E402

STAGES = tuple(fr.BARN_STAGES)
HOUSINGS = tuple(fr.HOUSING)


class Barn(BaseModel):
    name: str = Field(..., max_length=40)
    stage: str = Field(..., description=f"축사 용도 — {', '.join(STAGES)}")
    rooms: int = Field(..., ge=1, le=99)
    per: int = Field(..., ge=1, le=9999, description="방당 자리(분만사는 분만틀 수)")
    housing: str = "group"
    area_m2: float | None = Field(None, ge=0, description="방당 면적. 비우면 밀사 판정만 못 한다")


class Performance(BaseModel):
    """**비운 것은 비운 채로 둔다.** 중앙값으로 채우면 격차가 늘 0 이 된다."""
    farrowing_rate: float | None = Field(None, ge=40, le=100)
    weaned: float | None = Field(None, ge=5, le=18)
    npd: float | None = Field(None, ge=0, le=200)
    wean_to_estrus: float | None = Field(None, ge=3, le=30)
    survival: float | None = Field(None, ge=50, le=100, description="이유후 육성률(%)")


class FarmSetup(BaseModel):
    """등록 화면(`farm_setup.html`)이 내보내는 JSON 과 같은 모양이다."""
    name: str | None = None
    n_sows: int | None = Field(None, ge=1, le=20000)
    interval_days: float = Field(21, ge=3.5, le=35)
    lactation_days: float = Field(24, ge=14, le=42)
    pre_farrow_days: float = Field(7, ge=0, le=14)
    washout_days: float = Field(7, ge=0, le=14)
    barns: list[Barn] = []
    performance: Performance = Performance()


class FarmIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    setup: FarmSetup


class FarmOut(BaseModel):
    id: int
    name: str
    setup: dict
    created_at: str
    updated_at: str


class WeaningIn(BaseModel):
    weaning_date: str = Field(..., description="이유일 YYYY-MM-DD")
    parity: str = Field("sow", pattern="^(sow|primiparous)$")
    season_hot: bool = False


class DetectionIn(BaseModel):
    """발정 점검 주기 → 그 주기에서 **가능한 최적** 시점으로 비교한다.

    시점을 고정한 채 주기만 늘리면 CCTV 가 부당하게 유리해진다 — 이 프로젝트가
    실제로 냈다가 고친 오류다.
    """
    check_interval_h: float = Field(24.0, gt=0, le=48)
    parity: str = Field("sow", pattern="^(sow|primiparous)$")
