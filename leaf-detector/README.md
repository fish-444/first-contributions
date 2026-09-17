# 🌿 잎(Leaf) 탐지 웹앱

사진을 브라우저에서 업로드하면, Python 백엔드(FastAPI)가 **YOLO(ultralytics) 모델**로
잎을 탐지하고, 탐지된 위치에 **박스를 그려서** 다시 화면에 보여주는 웹앱입니다.

> 코딩을 몰라도 괜찮아요. 아래 순서대로 **한 줄씩 그대로 따라** 하시면 됩니다.
> 대부분의 명령어는 복사해서 붙여넣기만 하면 됩니다.

---

## 📁 폴더 구성

```
leaf-detector/
├── main.py             ← 서버 프로그램 (사진을 받아 YOLO로 탐지)
├── requirements.txt    ← 필요한 프로그램(라이브러리) 목록
├── README.md           ← 지금 보고 있는 설명서
└── static/
    └── index.html      ← 브라우저에 보이는 업로드 화면
```

---

## 0단계. 준비물 (딱 한 번만)

**Python(파이썬)** 이 설치되어 있어야 합니다. (3.9 버전 이상 권장)

- 이미 설치되어 있는지 확인하려면, **터미널**(Mac) 또는 **명령 프롬프트/PowerShell**(Windows)을 열고
  아래를 입력한 뒤 Enter를 누르세요:

  ```bash
  python3 --version
  ```

  `Python 3.11.x` 같은 글자가 나오면 설치되어 있는 겁니다. (Windows에서는 `python --version`)

- 만약 "명령을 찾을 수 없다"는 오류가 나오면, https://www.python.org/downloads/ 에서
  Python을 내려받아 설치하세요. (Windows에서는 설치 화면에서 **"Add Python to PATH"** 를 꼭 체크!)

> 💡 **터미널 여는 방법**
> - **Mac**: `Command + Space` → "터미널" 검색 → Enter
> - **Windows**: 시작 메뉴 → "PowerShell" 검색 → 실행

---

## 1단계. 이 폴더로 이동하기

터미널에서 이 `leaf-detector` 폴더로 들어갑니다.
(아래 경로는 이 프로젝트를 내려받은 위치에 맞게 바꿔 주세요.)

```bash
cd 경로/first-contributions/leaf-detector
```

> 💡 폴더를 터미널 창으로 **끌어다 놓으면** 경로가 자동으로 입력됩니다.

---

## 2단계. 필요한 프로그램 설치 (딱 한 번만)

아래 명령을 복사해서 붙여넣고 Enter를 누르세요. (인터넷 연결 필요, 몇 분 걸릴 수 있어요)

```bash
python3 -m pip install -r requirements.txt
```

> Windows라 위 명령이 안 되면 `python -m pip install -r requirements.txt` 로 시도하세요.

이 과정에서 FastAPI, YOLO(ultralytics) 등 필요한 도구들이 자동으로 설치됩니다.

---

## 3단계. 서버 실행하기

아래 명령을 입력하고 Enter:

```bash
python3 -m uvicorn main:app --reload
```

> Windows: `python -m uvicorn main:app --reload`

- 처음 실행하면 기본 YOLO 모델(`yolov8n.pt`)을 인터넷에서 자동으로 내려받습니다. (한 번만)
- 아래와 비슷한 글자가 보이면 성공입니다:

  ```
  [모델 로딩 완료]
  Uvicorn running on http://127.0.0.1:8000
  ```

---

## 4단계. 브라우저에서 사용하기

웹 브라우저(크롬, 사파리 등)를 열고 주소창에 아래를 입력하세요:

```
http://127.0.0.1:8000
```

1. 화면에서 📷 영역을 눌러 **사진을 선택**하거나, 사진을 끌어다 놓습니다.
2. **‘탐지하기’** 버튼을 누릅니다.
3. 잠시 후 오른쪽에 **박스가 그려진 결과 사진**과 탐지 목록이 나타납니다.

---

## 5단계. 서버 끄기

터미널에서 **`Ctrl + C`** (컨트롤 키 + C)를 누르면 서버가 꺼집니다.
다시 켜려면 3단계 명령을 또 실행하면 됩니다.

---

## 🟣 로보플로우(Roboflow) 모델 연결하기 — 추천 ✅

로보플로우에서 이미 학습시킨 모델(예: `find-leaf-mcfh8` 버전 1)이 있다면,
파일을 내려받지 않고 **로보플로우 서버에 바로 연결**해서 쓸 수 있습니다.

### ① API 키 찾기 (딱 한 번만)

API 키는 "이 모델을 쓸 권한이 있는 사람"임을 확인하는 비밀번호 같은 값입니다.

1. https://app.roboflow.com 에 로그인합니다.
2. 왼쪽 아래 **⚙️ Settings(설정)** 클릭 → 내 워크스페이스 선택.
3. **API Keys** 메뉴를 열면 **Private API Key** 가 보입니다. (예: `aB3xY...` 같은 긴 글자)
4. 그 값을 복사해 둡니다. **이 키는 남에게 보여주면 안 됩니다.**

> 💡 다른 위치: 모델 페이지에서 **Deploy** 탭에 들어가도 코드 예시 안에 `api_key="..."` 형태로 키가 보입니다.

### ② 모델 이름/버전 확인

우리 앱은 기본값으로 `find-leaf-mcfh8/1` (프로젝트 `find-leaf-mcfh8` 의 버전 `1`)을 사용합니다.
로보플로우 프로젝트 페이지에서 **Versions** 를 보면 버전 번호를 확인할 수 있어요.
다르면 아래 `ROBOFLOW_MODEL_ID` 값을 바꾸면 됩니다.

### ③ 실행하기

3단계 대신 아래처럼 **API 키를 넣어서** 서버를 켭니다.
(`여기에_복사한_키` 부분을 ①에서 복사한 실제 키로 바꾸세요.)

- **Mac / Linux:**
  ```bash
  ROBOFLOW_API_KEY="여기에_복사한_키" python3 -m uvicorn main:app --reload
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:ROBOFLOW_API_KEY="여기에_복사한_키"; python -m uvicorn main:app --reload
  ```

모델 이름/버전이 다르다면 이렇게 같이 지정하세요:
```bash
ROBOFLOW_API_KEY="여기에_복사한_키" ROBOFLOW_MODEL_ID="프로젝트이름/버전번호" python3 -m uvicorn main:app --reload
```

이제 4단계처럼 `http://127.0.0.1:8000` 에 접속해서 사진을 올리면
**로보플로우의 잎 탐지 모델**이 잎을 찾아 박스를 그려줍니다! 🌿

> 화면 아래 터미널에 `[로보플로우 방식] 모델: find-leaf-mcfh8/1` 이라고 뜨면 연결 성공입니다.
> 만약 `[로컬 방식]` 이라고 뜬다면 API 키가 안 들어간 것이니 ③을 다시 확인하세요.

> 🔒 이 방식은 `ultralytics` 라이브러리 없이도 동작합니다. 로보플로우만 쓸 거라면
> `requirements.txt` 의 `ultralytics` 줄은 설치하지 않아도 됩니다.

---

## 🎯 내가 학습시킨 YOLO weight 파일 사용하기 (로컬 방식)

기본 모델(`yolov8n.pt`)은 잎 전용이 아니라 사람·자동차 같은 **일반 사물**을 탐지합니다.
**직접 학습시킨 잎 탐지용 weight 파일(예: `best.pt`)** 이 있다면 아래처럼 바꿔 주세요.

### 방법 A) `main.py` 파일을 직접 수정 (가장 쉬움)

`main.py` 파일을 열어서 이 줄을 찾습니다:

```python
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
```

`"yolov8n.pt"` 부분을 여러분의 weight 파일 경로로 바꿉니다. 예:

```python
MODEL_PATH = os.environ.get("MODEL_PATH", "/Users/내이름/Desktop/best.pt")
```

저장한 뒤 서버를 껐다가(`Ctrl + C`) 3단계로 다시 켜면 적용됩니다.

### 방법 B) 명령어로 지정 (파일 수정 없이)

- **Mac / Linux:**
  ```bash
  MODEL_PATH="/경로/best.pt" python3 -m uvicorn main:app --reload
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:MODEL_PATH="C:\경로\best.pt"; python -m uvicorn main:app --reload
  ```

---

## ⚙️ 자주 하는 조정

- **탐지 민감도**: 잎이 잘 안 잡히면 `main.py`의 `CONFIDENCE` 값을 `25`에서 `10`처럼 낮춰 보세요.
  (0~100 사이 값이며, 낮을수록 더 많이 잡지만 오탐도 늘어납니다.) 명령어로도 지정할 수 있어요:
  ```bash
  CONFIDENCE=10 ROBOFLOW_API_KEY="여기에_복사한_키" python3 -m uvicorn main:app --reload
  ```

---

## ❓ 문제 해결 (자주 나는 오류)

| 증상 | 해결 방법 |
| --- | --- |
| `command not found: python3` | Python이 설치 안 됨 → 0단계 참고. Windows는 `python` 사용 |
| `No module named uvicorn` | 2단계 설치를 안 했거나 실패함 → 2단계 다시 실행 |
| 주소창 접속이 안 됨 | 3단계 서버가 켜져 있는지, 주소가 `http://127.0.0.1:8000` 인지 확인 |
| 결과 사진에 박스가 없음 | 기본 모델은 잎을 모릅니다 → 위 "weight 파일 사용하기" 참고 |
| `Address already in use` | 이미 서버가 켜져 있음. 기존 터미널에서 `Ctrl + C` 후 다시 실행 |

---

## 🔍 작동 원리 (궁금하다면)

```
[브라우저]  사진 업로드
    │  (인터넷 주소 /detect 로 사진 전송)
    ▼
[FastAPI 서버 main.py]
    │  YOLO 모델이 잎 위치를 찾음
    │  찾은 위치에 박스를 그림
    ▼
[브라우저]  박스가 그려진 사진을 화면에 표시
```

즐거운 잎 탐지 되세요! 🌱
