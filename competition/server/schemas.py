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


class HerdIn(BaseModel):
    """개체 이력 스냅숏. **기준일이 같이 온다.**

    이 목록은 하루의 스냅숏이라 다른 날짜로 읽으면 한 주기(145일)를 벗어난
    개체가 통째로 빠진다 — `run_farm --herd` 가 이미 겪고 막아 둔 사고라
    여기서도 날짜를 밖에 두지 않는다.
    """
    as_of: str = Field(..., description="기준일 YYYY-MM-DD")
    records: list[dict] = Field(..., max_length=20000,
                                description="id·parity·weaning_date·"
                                            "service_date·farrow_date·outcome")
    include_disease: bool = Field(
        True, description="질병 헤드는 달력이 없어 전 개체가 대상이다")


class ExportIn(BaseModel):
    """내보낼 표에 필요한 것만 넣는다 — 표마다 다르다.

    성적은 `performance` 로 따로 줄 수도, `setup` 안의 것을 쓸 수도 있다.
    어느 쪽이든 **비운 칸은 비운 채로** 간다.
    """
    setup: FarmSetup | None = None
    performance: Performance | None = None
    herd: HerdIn | None = None
    sows: int | None = Field(None, ge=1, le=20000)


# -- ops 라우터 (교배 배정 · 환경 알람 · 행동 기준선) ------------------------
class Animal(BaseModel):
    """모돈·웅돈 한 마리. **혈통은 비워도 된다** — 대신 근친율이 하한이 된다.

    `sire`/`dam` 이 목록에 없는 번호면 시조로 친다. 같은 부 번호를 적은 두
    개체는 그만큼의 혈연으로 이어진다.
    """
    id: str = Field(..., min_length=1, max_length=40)
    index: float = Field(..., description="유전평가 인덱스 — 정확도는 입력의 질")
    sire: str | None = Field(None, max_length=40)
    dam: str | None = Field(None, max_length=40)
    max_services: int | None = Field(None, ge=1, le=50,
                                     description="웅돈 전용 사용 상한")


class MatingIn(BaseModel):
    max_f: float = Field(0.0625, ge=0, le=0.5,
                         description="근친 한도 — 기본은 지침(사촌 수준)")
    services: int = Field(3, ge=1, le=50, description="웅돈 상한 기본값")
    sows: list[Animal] = Field(..., max_length=2000)
    boars: list[Animal] = Field(..., max_length=200)


class BarnEnv(BaseModel):
    """한 돈사의 센서 이력. **마지막 값이 현재**고 그 앞이 기준선 재료다.

    센서는 있는 것만 넣는다 — 습도·황화수소가 없는 농장이 흔하고, 없는
    센서를 0 으로 채우면 그게 곧 위반으로 잡힌다.
    """
    barn: str = Field(..., min_length=1, max_length=40)
    stage: str = Field("임신돈·웅돈", max_length=20)
    temp_c: list[float] = Field(default_factory=list, max_length=100000)
    nh3_ppm: list[float] = Field(default_factory=list, max_length=100000)
    rh_pct: list[float] = Field(default_factory=list, max_length=100000)
    h2s_ppm: list[float] = Field(default_factory=list, max_length=100000)
    implantation: bool = Field(
        False, description="착상기(교배 후 7~21일) 모돈 재실 — 번식 달력이 "
                           "아는 정보다. 고온 위반 알람의 우선순위 표시에 쓴다")
    day_temps: list[float] = Field(default_factory=list, max_length=1000,
                                   description="같은 날 온도들 — 일교차 점검")
    spot_temps: list[float] = Field(default_factory=list, max_length=1000,
                                    description="같은 시각 지점별 — 자리 편차")


class EnvIn(BaseModel):
    barns: list[BarnEnv] = Field(..., max_length=200)
    guide: dict | None = Field(
        None, description="지침 오버라이드 — temp/rh/nh3/h2s 중 바꿀 것만. "
                          "농장 기준이 다르면 여기로 통째로 바꾼다")


class BaselineIn(BaseModel):
    """행동 구성비의 자기 기준선. `heads` 는 **달력이 연 헤드**만 넘긴다.

    발정과 분만 임박은 신호가 겹쳐서(둘 다 Eating↓·Walking↑) 점수로는 못
    가른다. 생략하면 전 헤드를 계산하지만 그건 시연·감사용이다.
    """
    key: str = Field("방", max_length=40)
    history: list[dict[str, float]] = Field(..., max_length=100000)
    now: dict[str, float] = Field(...)
    recent: list[dict[str, float]] = Field(default_factory=list,
                                           max_length=100)
    classes: list[str] = Field(default_factory=list, max_length=50)
    heads: list[str] = Field(default_factory=list, max_length=10)
