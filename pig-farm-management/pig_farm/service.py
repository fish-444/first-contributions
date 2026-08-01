"""알림 서비스 계층 (Alert / service layer).

Aggregates pigs across farms and produces the alerts described in the spec so a
web or mobile front-end can notify managers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, Iterable, List, Optional

from .models import Farm, Pig, PigStatus
from .prediction import (
    is_gilt_ready_for_first_mating,
    is_heat_detection_target,
    is_in_weaning_heat_window,
    predict_next_heat_date,
    weaning_heat_window,
)


class AlertType(str, Enum):
    """알림 유형 (Alert categories)."""

    GILT_FIRST_MATING = "후보돈_초교배_적기"      # gilt ready for first mating
    HEAT_DETECTION = "발정_탐지_대상"            # cycle-based heat detection target
    WEANING_HEAT = "이유후_발정_탐지"            # post-weaning heat detection


@dataclass
class Alert:
    """관리자에게 제공되는 알림 (a single actionable alert)."""

    alert_type: AlertType
    farm_id: str
    ear_tag: str
    on_date: date
    message: str
    predicted_date: Optional[date] = None


class HeatPredictionService:
    """농장별 개체 관리 및 발정 예측 서비스.

    Register farms and pigs, then call the query/alert methods with the date
    the check is being run for (defaults to today).
    """

    def __init__(self) -> None:
        self._farms: Dict[str, Farm] = {}
        self._pigs: Dict[str, Pig] = {}

    # --- 등록 (registration) -------------------------------------------------

    def add_farm(self, farm: Farm) -> None:
        self._farms[farm.farm_id] = farm

    def add_pig(self, pig: Pig) -> None:
        self._pigs[pig.ear_tag] = pig

    def add_pigs(self, pigs: Iterable[Pig]) -> None:
        for pig in pigs:
            self.add_pig(pig)

    # --- 조회 (lookups) ------------------------------------------------------

    @property
    def pigs(self) -> List[Pig]:
        return list(self._pigs.values())

    def pigs_in_farm(self, farm_id: str) -> List[Pig]:
        return [p for p in self._pigs.values() if p.farm_id == farm_id]

    def _iter_pigs(self, farm_id: Optional[str]) -> Iterable[Pig]:
        if farm_id is None:
            return self._pigs.values()
        return self.pigs_in_farm(farm_id)

    # --- ① 후보돈 초교배 적기 --------------------------------------------------

    def gilts_ready_for_mating(
        self, on_date: date, farm_id: Optional[str] = None
    ) -> List[Pig]:
        """초교배 적기 조건을 충족한 후보돈 목록."""
        return [
            p
            for p in self._iter_pigs(farm_id)
            if p.status in (PigStatus.GILT, PigStatus.MATING_READY)
            and is_gilt_ready_for_first_mating(p, on_date)
        ]

    # --- ② 주기 기반 발정 탐지 -------------------------------------------------

    def heat_detection_targets(
        self, on_date: date, farm_id: Optional[str] = None
    ) -> List[Pig]:
        """예정일 ±2일 내외의 발정 탐지 대상 목록."""
        return [
            p
            for p in self._iter_pigs(farm_id)
            if is_heat_detection_target(p, on_date)
        ]

    # --- ③ 이유 후 발정 탐지 ---------------------------------------------------

    def weaning_heat_targets(
        self, on_date: date, farm_id: Optional[str] = None
    ) -> List[Pig]:
        """이유 후 발정 집중 탐지 기간 내 개체 목록."""
        return [
            p
            for p in self._iter_pigs(farm_id)
            if is_in_weaning_heat_window(p, on_date)
        ]

    # --- 알림 생성 (alert generation) ------------------------------------------

    def generate_alerts(
        self, on_date: date, farm_id: Optional[str] = None
    ) -> List[Alert]:
        """주어진 날짜 기준으로 모든 알림을 생성."""
        alerts: List[Alert] = []

        for pig in self.gilts_ready_for_mating(on_date, farm_id):
            alerts.append(
                Alert(
                    alert_type=AlertType.GILT_FIRST_MATING,
                    farm_id=pig.farm_id,
                    ear_tag=pig.ear_tag,
                    on_date=on_date,
                    message=(
                        f"후보돈 {pig.ear_tag}: 초교배 적기 도달 "
                        f"(일령/체중 조건 충족)"
                    ),
                )
            )

        for pig in self.heat_detection_targets(on_date, farm_id):
            predicted = predict_next_heat_date(pig)
            alerts.append(
                Alert(
                    alert_type=AlertType.HEAT_DETECTION,
                    farm_id=pig.farm_id,
                    ear_tag=pig.ear_tag,
                    on_date=on_date,
                    predicted_date=predicted,
                    message=(
                        f"개체 {pig.ear_tag}: 발정 예정일({predicted}) 근접 "
                        f"— 발정 탐지 요망"
                    ),
                )
            )

        for pig in self.weaning_heat_targets(on_date, farm_id):
            window = weaning_heat_window(pig)
            start, end = window  # window is not None for these pigs
            alerts.append(
                Alert(
                    alert_type=AlertType.WEANING_HEAT,
                    farm_id=pig.farm_id,
                    ear_tag=pig.ear_tag,
                    on_date=on_date,
                    predicted_date=start,
                    message=(
                        f"개체 {pig.ear_tag}: 이유 후 발정 집중 탐지 기간 "
                        f"({start} ~ {end})"
                    ),
                )
            )

        return alerts
