# 🌿 스마트팜 온실 3D 대시보드 (FastAPI + YOLO + Three.js)

사진을 업로드하면 백엔드 **YOLO**가 식물 상태를 분석하고, **3D 온실**의 해당 식물이
실시간으로 바뀌며, 식물을 **클릭**하면 분석 결과 + 가상 센서 + **RAG 생장 피드백**이
담긴 상세 팝업이 뜨는 인터랙티브 대시보드입니다.

> 코딩 몰라도 괜찮아요. 아래 순서대로 그대로 따라 하시면 됩니다.

---

## 🧩 전체 시스템 흐름

```
[브라우저] 구역 선택 + 식물 사진 업로드
    │  POST /api/analyze (plant_id, file)
    ▼
[FastAPI 백엔드]
    │  ① YOLO 분석:  크기분류(대/중/소품) · 총 잎 개수 · 겹친 잎 · 새순 유무
    │  ② 가상 IoT 센서:  온도 · 빛 효율
    │  ③ RAG 로직:  상태에 맞는 생장 피드백 문장 생성
    │  → 해당 식물 '상태'로 저장
    ▼
[3D 온실 뷰어]  해당 식물 오브젝트 크기·색 실시간 갱신
    │  식물 클릭
    ▼
[상세 모달]  잎 개수 · 겹침 · 새순 · 온도 · 빛효율 · RAG 피드백 표시
```

---

## 📁 폴더 구조

```
smartfarm/
├── main.py                 ← FastAPI 백엔드 (분석·센서·RAG·상태저장 API)
├── requirements.txt        ← 필요한 라이브러리 목록
├── README.md               ← 지금 이 파일
└── static/
    ├── index.html          ← 3D 온실 화면 + 업로드 + 모달 (프론트엔드)
    ├── three.min.js        ← 3D 라이브러리 (Three.js, 이미 포함)
    └── OrbitControls.js    ← 3D 카메라 조작 (이미 포함)
```

---

## ▶️ 실행 방법

### 0. 준비물
- **Python 3.9 이상** (터미널에서 `python3 --version` 으로 확인, Windows는 `python`)

### 1. 이 폴더로 이동
```bash
cd 경로/first-contributions/smartfarm
```

### 2. 필요한 프로그램 설치 (딱 한 번)
```bash
python3 -m pip install -r requirements.txt
```
> Windows: `python -m pip install -r requirements.txt`

### 3. 서버 실행
```bash
python3 -m uvicorn main:app --reload
```
`Uvicorn running on http://127.0.0.1:8000` 이 보이면 성공.

### 4. 브라우저에서 열기
주소창에 **`http://127.0.0.1:8000`** 입력 → 3D 온실이 나옵니다.
1. 오른쪽에서 **구역/식물**을 고르고
2. 그 식물의 **사진을 선택**한 뒤
3. **분석하기** → 3D가 갱신되고 상세 모달이 떠요.
4. 3D 속 아무 식물이나 **클릭**해도 최신 상태를 볼 수 있어요.

### 5. 끄기
터미널에서 **Ctrl + C**.

---

## 🤖 분석 엔진 3가지 (자동 선택)

`main.py` 는 아래 순서로 엔진을 자동 선택합니다.

| 우선순위 | 엔진 | 조건 | 특징 |
|---|---|---|---|
| 1 | **로보플로우** | `ROBOFLOW_API_KEY` 환경변수 있음 | 학습시킨 잎 모델로 실제 분석 |
| 2 | **로컬 YOLO** | `.pt` 파일 있음(`MODEL_PATH`) | 내 컴퓨터에서 직접 분석 |
| 3 | **데모** | 위 둘 다 없음 | 모델 없이도 그럴듯한 결과로 체험 |

> 처음엔 **데모 모드**로 바로 돌아가요. 화면 왼쪽 위 "분석 엔진: demo" 로 확인.

### 로보플로우 모델로 실제 분석하기
```bash
# Mac / Linux
ROBOFLOW_API_KEY="개인키" ROBOFLOW_MODEL_ID="find-leaf-and-object/1" python3 -m uvicorn main:app --reload
```
```powershell
# Windows PowerShell
$env:ROBOFLOW_API_KEY="개인키"; $env:ROBOFLOW_MODEL_ID="find-leaf-and-object/1"; python -m uvicorn main:app --reload
```
> ⚠️ **Private API Key** 를 쓰세요(공개키는 서버리스 추론에서 막혀요). 모델 버전은 본인 것에 맞게.

### 내 .pt 파일로 분석하기
```bash
MODEL_PATH="/경로/best.pt" python3 -m uvicorn main:app --reload
```

---

## 🔌 API 요약 (개발자용)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 3D 대시보드 화면 |
| GET | `/api/plants` | 전체 식물 배치 + 현재 상태 |
| GET | `/api/plants/{id}` | 특정 식물 상태 |
| POST | `/api/analyze` | `plant_id`+`file` 업로드 → 분석·저장 후 상태 반환 |

`/api/analyze` 응답 예시:
```json
{
  "id": "A1", "zone": "A", "size_class": "대품",
  "leaf_count": 11, "overlap_count": 4, "new_shoot": true,
  "temp": 25.3, "light_eff": 68,
  "feedback": "🌱 새순이 확인됩니다 … 🪴 대품으로 성장했습니다 …",
  "thumb": "data:image/jpeg;base64,…", "updated": "2026-07-24 23:22:42"
}
```

---

## ⚙️ 커스터마이즈 힌트

- **크기 분류 기준**: `main.py` 의 `analyze_metrics()` 에서 `coverage`/`leaf_count` 임계값 조정
- **겹친 잎 민감도**: `analyze_metrics()` 의 `OVERLAP_IOU`(기본 0.12)
- **RAG 피드백 문장**: `main.py` 의 `KNOWLEDGE` 리스트에 규칙(조건, 문장) 추가
- **온실 배치/구역**: `_build_layout()` 에서 화분 위치·개수 변경
- **센서 값 로직**: 실제 IoT 를 붙이려면 `read_sensors()` 를 센서 API 호출로 교체

즐거운 스마트팜 되세요! 🌱
