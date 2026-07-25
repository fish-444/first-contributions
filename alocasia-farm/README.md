# 🌿 알로카시아 스마트팜 3D 온실

사진 + 이름을 입력하면 **Roboflow(YOLO)** 가 분석하고, **3D 온실**에 화분으로 나타나요.
화분을 클릭하면 내가 정한 이름과 분석 결과(잎 개수·겹침·새순)가 뜹니다.
센서·RAG 없이 **[사진+이름 → YOLO → 3D 반영 + 이름 커스텀]** 에만 집중한 깔끔한 단일 웹앱.

> 코딩 몰라도 돼요. 아래 순서대로 따라 하시면 됩니다.

---

## 📁 폴더 구조
```
alocasia-farm/
├── main.py              ← FastAPI 백엔드 (분석·추가·제거·이름수정)
├── requirements.txt
├── README.md
└── static/
    ├── index.html       ← 3D 온실 + 업로드 폼 + 리스트 + 모달
    ├── three.min.js     ← 3D 라이브러리 (동봉)
    └── OrbitControls.js
```

## ▶️ 실행
```bash
cd 경로/first-contributions/alocasia-farm
python3 -m pip install -r requirements.txt     # 처음 한 번 (Windows: python)
python3 -m uvicorn main:app --reload           # 서버 실행
```
브라우저에서 **http://127.0.0.1:8000** 접속.

## 사용법
1. 왼쪽 **① 식물 이름** 입력 (예: 프라이덱)
2. **② 식물 사진** 선택
3. **③ 자리 선택** (A1~E10 중 빈 자리) → **🌱 온실에 추가**
4. 3D 온실 해당 자리에 화분이 나타나고, **자리 지도**와 **리스트**에도 반영됨
5. 화분·리스트·자리 지도 **클릭** → 분석 결과 **모달**
6. 모달에서 **이름 수정**, **화분 방향 회전**, **새 사진으로 갱신**, **제거** 가능

### 🗺️ 자리(슬롯) 시스템
사진만으로는 "무슨 식물인지" 구분할 수 없기 때문에, **자리로 개체를 식별**합니다.
온실 60 x 40 cm 를 **5줄 x 10칸 = 50자리** 로 나눠 `A1`~`E10` 라벨을 붙였어요 (한 칸 6 x 8 cm).

- 자리 지도에서 **빈자리 탭** → 그 자리가 자동 선택되고 이름 입력으로 이동
- 자리 지도에서 **찬 자리 탭** → 그 식물 상세 모달
- 식물 라벨은 3D 에서도 `A1 · 프라이덱` 형태로 표시

### 🎨 3D 표현 규칙
| 요소 | 색/크기 | 근거 |
|---|---|---|
| 화분 | 흰색 | 고정 |
| 새순(Shoot) | 연두 `#a4e05a` | 모델2 `shoot_count` |
| 성엽(Mature) | 녹색 `#35913c` | 모델2 `mature_count` |
| 노엽(Old) | 노랑 `#e0cf4a` | 모델2 `old_count` |
| 식물 크기 | 소엽 1.3 / 중엽 1.85 / 대엽 2.5 | 모델1 `top_leaf_size` |

화분 방향은 모달의 **🧭 슬라이더**로 0~359° 조절되며 서버에 저장됩니다.

## 분석 엔진 (자동 선택)
| 우선순위 | 조건 | 방식 |
|---|---|---|
| 1 | `ROBOFLOW_API_KEY` 있음 | 로보플로우(YOLO) 실제 분석 |
| 2 | `.pt` 파일 있음 | 로컬 ultralytics |
| 3 | 둘 다 없음 | 데모(모델 없이 체험) |

### 로보플로우로 실제 분석
```bash
ROBOFLOW_API_KEY="개인키" ROBOFLOW_MODEL_ID="find-leaf-and-object/1" python3 -m uvicorn main:app --reload
```
> **Private API Key** 사용 (공개키는 막힘). Windows PowerShell 은 `$env:ROBOFLOW_API_KEY="..."; ...`

### 모델 2개 연동 (키 하나로)
로보플로우 키 1개로 **모델 두 개**를 함께 씁니다. 각각 다른 영역에 반영돼요:

| 모델 | 환경변수 | 반영 위치 |
|---|---|---|
| **모델1** 맨 위 잎(광합성) | `ROBOFLOW_MODEL_TOP` | **3D 온실** (식물 크기·강조 잎) |
| **모델2** 새순/성숙/노령 | `ROBOFLOW_MODEL_STAGE` | **개체 특징 모달** (단계별 개수) |

```bash
ROBOFLOW_API_KEY="개인키" \
ROBOFLOW_MODEL_TOP="top-leaf-model/1" \
ROBOFLOW_MODEL_STAGE="leaf-stage-model/1" \
python3 -m uvicorn main:app --reload
```
> 두 모델을 **다르게** 지정하면 사진 1장당 추론이 2번 돌아 크레딧도 2배예요.
> 같게 두거나 하나만 지정하면 1번만 호출(재사용)해서 절약합니다.

## API
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/plants` | `name`+`file`+`pos` → 분석 후 식물 추가 |
| GET | `/api/plants` | 전체 목록 (+ 사용 중인 `engine`) |
| GET | `/api/slots` | 자리 50개 목록 + 점유 여부 |
| PATCH | `/api/plants/{id}` | `name` 수정 / `rot` 방향(0~359°) |
| POST | `/api/plants/{id}/reanalyze` | `file` → 새 사진으로 갱신 |
| DELETE | `/api/plants/{id}` | 제거 |

분석 결과 필드:

| 필드 | 출처 | 쓰이는 곳 |
|---|---|---|
| `top_leaf_size` (소엽/중엽/대엽/없음), `top_leaf_pct` | 모델1 | 3D 크기 · 모달 |
| `shoot_count`, `mature_count`, `old_count` | 모델2 | 3D 잎 색 · 모달 |
| `leaf_count`, `overlap_count`, `overlap_density`(%) | 모델2 | 모달 |
| `size_class` (대/중/소품) | 모델2 | 리스트 · 모달 |
| `pos`, `x`, `z`, `rot` | 자리 시스템 | 3D 배치 |

> 식물 상태는 **메모리에만** 저장됩니다. 서버를 재시작하면 초기화돼요.
