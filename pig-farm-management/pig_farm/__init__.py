"""양돈 농장별 개체 관리 및 발정 주기 예측 프로그램.

Pig farm individual management and heat-cycle prediction system.

This package predicts heat cycles and optimal mating windows for pigs based on
per-farm and per-animal data (birth date, heat history, wean date) and surfaces
alerts for farm managers and veterinarians.
"""

from .models import Farm, MatingRecord, Pig, PigStatus
from .prediction import (
    GILT_MIN_AGE_DAYS,
    GILT_MIN_WEIGHT_KG,
    HEAT_CYCLE_DAYS,
    HEAT_DETECTION_WINDOW_DAYS,
    WEANING_HEAT_MAX_DAYS,
    WEANING_HEAT_MIN_DAYS,
    age_in_days,
    is_gilt_ready_for_first_mating,
    is_heat_detection_target,
    is_in_weaning_heat_window,
    predict_next_heat_date,
    weaning_heat_window,
)
from .service import Alert, AlertType, HeatPredictionService

__all__ = [
    "Farm",
    "MatingRecord",
    "Pig",
    "PigStatus",
    "GILT_MIN_AGE_DAYS",
    "GILT_MIN_WEIGHT_KG",
    "HEAT_CYCLE_DAYS",
    "HEAT_DETECTION_WINDOW_DAYS",
    "WEANING_HEAT_MAX_DAYS",
    "WEANING_HEAT_MIN_DAYS",
    "age_in_days",
    "is_gilt_ready_for_first_mating",
    "is_heat_detection_target",
    "is_in_weaning_heat_window",
    "predict_next_heat_date",
    "weaning_heat_window",
    "Alert",
    "AlertType",
    "HeatPredictionService",
]
