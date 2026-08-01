"""핵심 비즈니스 로직 (Core prediction logic).

Pure, side-effect-free functions implementing the three prediction rules from
the spec:

1. 일령 계산 및 후보돈 초교배 적기 판단
   (age calculation & gilt first-mating readiness)
2. 주기 기반 발정 예정일 예측
   (cycle-based next-heat prediction)
3. 이유 후 발정 예측
   (post-weaning heat prediction)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Tuple

from .models import Pig

# --- 기준 상수 (Business constants) -----------------------------------------

#: 돼지 평균 발정 주기 (average heat cycle length, days).
HEAT_CYCLE_DAYS = 21

#: 후보돈 초교배 최소 일령 (minimum age for gilt first mating, days).
GILT_MIN_AGE_DAYS = 210

#: 후보돈 초교배 최소 체중 (minimum weight for gilt first mating, kg).
GILT_MIN_WEIGHT_KG = 130.0

#: 발정 예정일 전후 탐지 범위 (heat detection window around predicted date, ±days).
HEAT_DETECTION_WINDOW_DAYS = 2

#: 이유 후 발정 집중 탐지 시작/종료 (post-weaning heat detection window, days).
WEANING_HEAT_MIN_DAYS = 4
WEANING_HEAT_MAX_DAYS = 7


def age_in_days(pig: Pig, on_date: date) -> int:
    """개체의 일령을 계산 (age in days on ``on_date``)."""
    return (on_date - pig.birth_date).days


# --- ① 후보돈 초교배 적기 판단 ------------------------------------------------

def is_gilt_ready_for_first_mating(pig: Pig, on_date: date) -> bool:
    """후보돈 초교배 적기 여부.

    기준: 일령 210일 이상 AND 체중 130kg 이상.
    Returns True only when both the age and weight thresholds are met.
    Weight of ``None`` is treated as "not yet weighed" and fails the check.
    """
    if pig.weight_kg is None:
        return False
    return (
        age_in_days(pig, on_date) >= GILT_MIN_AGE_DAYS
        and pig.weight_kg >= GILT_MIN_WEIGHT_KG
    )


# --- ② 주기 기반 발정 예정일 예측 --------------------------------------------

def predict_next_heat_date(pig: Pig) -> Optional[date]:
    """다음 발정 예정일 = 최근 발정일 + 21일.

    Returns None when the pig has no recorded last-heat date.
    """
    if pig.last_heat_date is None:
        return None
    return pig.last_heat_date + timedelta(days=HEAT_CYCLE_DAYS)


def is_heat_detection_target(
    pig: Pig,
    on_date: date,
    window_days: int = HEAT_DETECTION_WINDOW_DAYS,
) -> bool:
    """발정 탐지 대상 여부.

    예정일 기준 ±window_days(기본 2일) 내외이면 탐지 대상으로 판단.
    """
    predicted = predict_next_heat_date(pig)
    if predicted is None:
        return False
    delta = abs((on_date - predicted).days)
    return delta <= window_days


# --- ③ 이유 후 발정 예측 -----------------------------------------------------

def weaning_heat_window(pig: Pig) -> Optional[Tuple[date, date]]:
    """이유 후 발정 집중 탐지 구간 = 이유일 + 4일 ~ 7일.

    Returns an inclusive (start, end) date pair, or None if no wean date.
    """
    if pig.wean_date is None:
        return None
    start = pig.wean_date + timedelta(days=WEANING_HEAT_MIN_DAYS)
    end = pig.wean_date + timedelta(days=WEANING_HEAT_MAX_DAYS)
    return start, end


def is_in_weaning_heat_window(pig: Pig, on_date: date) -> bool:
    """이유 후 발정 재발현 집중 탐지 기간 내 여부."""
    window = weaning_heat_window(pig)
    if window is None:
        return False
    start, end = window
    return start <= on_date <= end
