"""데이터 구조 (Data Model).

Core domain entities: Farm, Pig, MatingRecord and the PigStatus lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional


class PigStatus(str, Enum):
    """돼지 개체 상태 (Pig lifecycle status).

    Values are Korean domain terms; ``.label`` returns an English gloss.
    """

    GILT = "후보돈"        # candidate / replacement gilt (not yet bred)
    PREGNANT = "임신돈"    # confirmed pregnant sow
    LACTATING = "포유돈"   # farrowed / nursing sow
    OPEN = "공거돈"        # open / non-productive (weaned or returned, awaiting heat)
    MATING_READY = "교배적기"  # flagged as ready for mating

    @property
    def label(self) -> str:
        _labels = {
            PigStatus.GILT: "gilt",
            PigStatus.PREGNANT: "pregnant",
            PigStatus.LACTATING: "lactating",
            PigStatus.OPEN: "open",
            PigStatus.MATING_READY: "mating-ready",
        }
        return _labels[self]


@dataclass
class Farm:
    """농장 정보 (Farm)."""

    farm_id: str
    name: str
    manager_name: str = ""
    manager_contact: str = ""


@dataclass
class MatingRecord:
    """교배 이력 항목 (a single mating event)."""

    mating_date: date
    boar_id: Optional[str] = None
    # None = pending/unknown, True = confirmed pregnant, False = returned to heat.
    conception_confirmed: Optional[bool] = None
    note: str = ""


@dataclass
class Pig:
    """돼지 개체 정보 (Pig).

    A single animal identified by its ear-tag number, belonging to one farm.
    """

    ear_tag: str            # 개체 번호 (귀표 번호)
    farm_id: str            # 소속 농장 ID
    birth_date: date        # 생년월일
    status: PigStatus = PigStatus.GILT
    last_heat_date: Optional[date] = None   # 최근 발정일
    weight_kg: Optional[float] = None       # 체중 (kg)
    wean_date: Optional[date] = None        # 이유일 (Wean Date)
    mating_history: List[MatingRecord] = field(default_factory=list)

    def record_mating(self, record: MatingRecord) -> None:
        """교배 이력 추가 (append a mating record)."""
        self.mating_history.append(record)

    @property
    def last_mating(self) -> Optional[MatingRecord]:
        """가장 최근 교배 이력 (most recent mating record, if any)."""
        if not self.mating_history:
            return None
        return max(self.mating_history, key=lambda r: r.mating_date)
