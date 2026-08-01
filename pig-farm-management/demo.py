"""데모 실행 스크립트 (runnable demo).

Builds a small farm with a few pigs and prints the alerts that the service
would surface for a given date.

    python demo.py
"""

from datetime import date

from pig_farm import Farm, HeatPredictionService, Pig, PigStatus


def build_demo_service() -> HeatPredictionService:
    svc = HeatPredictionService()
    svc.add_farm(Farm("F1", "행복양돈", manager_name="홍길동",
                       manager_contact="010-1234-5678"))

    svc.add_pigs([
        # 후보돈: 211일령, 135kg -> 초교배 적기
        Pig("G-101", "F1", birth_date=date(2025, 1, 1),
            status=PigStatus.GILT, weight_kg=135),
        # 후보돈: 아직 체중 미달
        Pig("G-102", "F1", birth_date=date(2025, 1, 1),
            status=PigStatus.GILT, weight_kg=120),
        # 공거돈: 최근 발정일 기준 다음 발정 예정일 근접
        Pig("S-201", "F1", birth_date=date(2023, 5, 1),
            status=PigStatus.OPEN, last_heat_date=date(2025, 7, 10)),
        # 공거돈: 이유 후 발정 집중 탐지 기간
        Pig("S-202", "F1", birth_date=date(2023, 6, 1),
            status=PigStatus.OPEN, wean_date=date(2025, 7, 27)),
    ])
    return svc


def main() -> None:
    svc = build_demo_service()
    today = date(2025, 7, 31)

    print(f"=== 발정/교배 알림 ({today}) ===\n")
    alerts = svc.generate_alerts(today)
    if not alerts:
        print("해당 날짜에 알림 대상 개체가 없습니다.")
        return

    for alert in alerts:
        print(f"[{alert.alert_type.value}] 농장 {alert.farm_id} / {alert.message}")


if __name__ == "__main__":
    main()
