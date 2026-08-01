# 🐷 양돈 농장별 개체 관리 및 발정 주기 예측 프로그램

Pig-farm individual management and **heat-cycle prediction** system.

농장별·개체별 생년월일, 발정 이력, 이유일 데이터를 기반으로 **발정 주기 및 교배
적기를 예측**하고, 관리자에게 알림을 제공합니다.

## 목적 (Purpose)

발정 주기를 놓쳐 발생하는 **공거일(비생산 일수)을 최소화**하여 농장 생산성을
높이고, 웹/모바일 알림으로 현장 작업 효율을 극대화합니다.

## 데이터 구조 (Data Model)

| 엔티티 | 필드 |
| --- | --- |
| **Farm** (농장) | `farm_id`, `name`, `manager_name`, `manager_contact` |
| **Pig** (개체) | `ear_tag`(귀표), `farm_id`, `birth_date`(생년월일), `status`(상태), `last_heat_date`(최근 발정일), `weight_kg`(체중), `wean_date`(이유일), `mating_history`(교배 이력) |
| **MatingRecord** (교배 이력) | `mating_date`, `boar_id`, `conception_confirmed`, `note` |

**상태(Status)**: 후보돈 · 임신돈 · 포유돈 · 공거돈 · 교배적기
(`PigStatus` enum).

## 핵심 비즈니스 로직 (Core Logic)

### ① 후보돈 초교배 적기 판단
- **기준**: 일령 **210일 이상** AND 체중 **130kg 이상**
- 조건 충족 시 교배 적기 상태로 전환 및 알림 발생
- `is_gilt_ready_for_first_mating(pig, on_date)`

### ② 주기 기반 발정 예정일 예측
- **평균 발정 주기 21일** 적용
- `다음 발정 예정일 = 최근 발정일 + 21일`
- 예정일 기준 **±2일 내외** 개체를 '발정 탐지 대상'으로 추출
- `predict_next_heat_date(pig)`, `is_heat_detection_target(pig, on_date)`

### ③ 이유 후 발정 예측 (분만돈)
- **이유일 + 4일 ~ 7일** 구간을 발정 재발현 집중 탐지 기간으로 설정
- `weaning_heat_window(pig)`, `is_in_weaning_heat_window(pig, on_date)`

각 기준값은 `pig_farm/prediction.py` 상단 상수로 정의되어 있어 손쉽게 조정할 수
있습니다 (`HEAT_CYCLE_DAYS`, `GILT_MIN_AGE_DAYS`, `GILT_MIN_WEIGHT_KG`,
`HEAT_DETECTION_WINDOW_DAYS`, `WEANING_HEAT_MIN_DAYS`, `WEANING_HEAT_MAX_DAYS`).

## 사용 예시 (Usage)

```python
from datetime import date
from pig_farm import Farm, HeatPredictionService, Pig, PigStatus

svc = HeatPredictionService()
svc.add_farm(Farm("F1", "행복양돈", manager_name="홍길동"))
svc.add_pig(Pig("S-201", "F1", birth_date=date(2023, 5, 1),
                status=PigStatus.OPEN, last_heat_date=date(2025, 7, 10)))

for alert in svc.generate_alerts(date(2025, 7, 31)):
    print(alert.alert_type.value, alert.message)
```

## 실행 (Run)

```bash
cd pig-farm-management

# 데모 실행 (표준 라이브러리만 사용)
python demo.py

# 테스트 실행 (pytest 필요)
pip install pytest
pytest
```

## 프로젝트 구조 (Layout)

```
pig-farm-management/
├── pig_farm/
│   ├── __init__.py       # 공개 API
│   ├── models.py         # Farm · Pig · MatingRecord · PigStatus
│   ├── prediction.py     # 핵심 예측 로직 (순수 함수)
│   └── service.py        # 알림 서비스 계층 (HeatPredictionService, Alert)
├── tests/
│   └── test_prediction.py
├── demo.py
└── README.md
```

> 핵심 로직(`pig_farm`)과 데모는 **표준 라이브러리만** 사용합니다. `pytest`는
> 테스트 실행에만 필요합니다.
