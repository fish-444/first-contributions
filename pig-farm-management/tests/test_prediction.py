"""핵심 로직 단위 테스트 (unit tests for the prediction logic)."""

from datetime import date

import pytest

from pig_farm import (
    Farm,
    HeatPredictionService,
    Pig,
    PigStatus,
    age_in_days,
    is_gilt_ready_for_first_mating,
    is_heat_detection_target,
    is_in_weaning_heat_window,
    predict_next_heat_date,
    weaning_heat_window,
)
from pig_farm.service import AlertType


def make_pig(**kwargs) -> Pig:
    defaults = dict(ear_tag="P001", farm_id="F1", birth_date=date(2025, 1, 1))
    defaults.update(kwargs)
    return Pig(**defaults)


# --- age & gilt readiness ---------------------------------------------------

def test_age_in_days():
    pig = make_pig(birth_date=date(2025, 1, 1))
    assert age_in_days(pig, date(2025, 1, 11)) == 10


def test_gilt_ready_when_age_and_weight_met():
    pig = make_pig(birth_date=date(2025, 1, 1), weight_kg=135)
    # 210 days after birth = 2025-07-30
    on = date(2025, 7, 30)
    assert age_in_days(pig, on) == 210
    assert is_gilt_ready_for_first_mating(pig, on) is True


def test_gilt_not_ready_when_underage():
    pig = make_pig(birth_date=date(2025, 1, 1), weight_kg=140)
    on = date(2025, 7, 29)  # 209 days
    assert is_gilt_ready_for_first_mating(pig, on) is False


def test_gilt_not_ready_when_underweight():
    pig = make_pig(birth_date=date(2025, 1, 1), weight_kg=129.9)
    on = date(2025, 8, 30)  # well over 210 days
    assert is_gilt_ready_for_first_mating(pig, on) is False


def test_gilt_not_ready_when_weight_unknown():
    pig = make_pig(birth_date=date(2025, 1, 1), weight_kg=None)
    on = date(2025, 12, 1)
    assert is_gilt_ready_for_first_mating(pig, on) is False


# --- cycle-based heat prediction --------------------------------------------

def test_predict_next_heat_date():
    pig = make_pig(last_heat_date=date(2025, 8, 1))
    assert predict_next_heat_date(pig) == date(2025, 8, 22)


def test_predict_next_heat_date_none_without_history():
    pig = make_pig(last_heat_date=None)
    assert predict_next_heat_date(pig) is None


@pytest.mark.parametrize(
    "on, expected",
    [
        (date(2025, 8, 20), True),   # -2 days
        (date(2025, 8, 22), True),   # exact
        (date(2025, 8, 24), True),   # +2 days
        (date(2025, 8, 19), False),  # -3 days
        (date(2025, 8, 25), False),  # +3 days
    ],
)
def test_heat_detection_window(on, expected):
    pig = make_pig(last_heat_date=date(2025, 8, 1))  # predicted 2025-08-22
    assert is_heat_detection_target(pig, on) is expected


def test_heat_detection_false_without_history():
    pig = make_pig(last_heat_date=None)
    assert is_heat_detection_target(pig, date(2025, 8, 22)) is False


# --- post-weaning heat window -----------------------------------------------

def test_weaning_heat_window_bounds():
    pig = make_pig(wean_date=date(2025, 8, 1))
    assert weaning_heat_window(pig) == (date(2025, 8, 5), date(2025, 8, 8))


def test_weaning_heat_window_none_without_wean_date():
    pig = make_pig(wean_date=None)
    assert weaning_heat_window(pig) is None


@pytest.mark.parametrize(
    "on, expected",
    [
        (date(2025, 8, 4), False),  # +3, before window
        (date(2025, 8, 5), True),   # +4, start (inclusive)
        (date(2025, 8, 8), True),   # +7, end (inclusive)
        (date(2025, 8, 9), False),  # +8, after window
    ],
)
def test_is_in_weaning_heat_window(on, expected):
    pig = make_pig(wean_date=date(2025, 8, 1))
    assert is_in_weaning_heat_window(pig, on) is expected


# --- service / alert layer --------------------------------------------------

def test_service_generates_expected_alerts():
    svc = HeatPredictionService()
    svc.add_farm(Farm("F1", "행복농장", "홍길동", "010-0000-0000"))

    gilt = Pig("G1", "F1", birth_date=date(2025, 1, 1),
               status=PigStatus.GILT, weight_kg=135)
    sow = Pig("S1", "F1", birth_date=date(2023, 1, 1),
              status=PigStatus.OPEN, last_heat_date=date(2025, 7, 10))
    weaned = Pig("W1", "F1", birth_date=date(2023, 1, 1),
                 status=PigStatus.OPEN, wean_date=date(2025, 7, 27))
    svc.add_pigs([gilt, sow, weaned])

    on = date(2025, 7, 31)
    # sow predicted heat = 2025-07-31 (exact), gilt is 211 days & 135kg,
    # weaned window = 07-31 ~ 08-03 (start day).
    alerts = svc.generate_alerts(on)
    types = {a.alert_type for a in alerts}
    assert AlertType.GILT_FIRST_MATING in types
    assert AlertType.HEAT_DETECTION in types
    assert AlertType.WEANING_HEAT in types
    assert len(alerts) == 3


def test_service_filters_by_farm():
    svc = HeatPredictionService()
    svc.add_pig(Pig("A", "F1", birth_date=date(2020, 1, 1),
                    last_heat_date=date(2025, 7, 10)))
    svc.add_pig(Pig("B", "F2", birth_date=date(2020, 1, 1),
                    last_heat_date=date(2025, 7, 10)))
    targets = svc.heat_detection_targets(date(2025, 7, 31), farm_id="F1")
    assert [p.ear_tag for p in targets] == ["A"]
