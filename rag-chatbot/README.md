# 📄🖼️ 문서 + 이미지 RAG 챗봇 MVP

문서(.txt / .md / .pdf)를 업로드하면 내용을 청크로 나눠 벡터DB(Chroma)에 저장하고,
질문하면 관련 내용을 검색해서 Claude API로 답변을 생성하는 챗봇입니다.
이미지는 **YOLO-World(객체 탐지)** 와 **SAM-2(영역 분할)** 로 분석하며,
분석 결과 설명이 벡터DB에 함께 저장되어 문서와 이미지를 통합 검색할 수 있습니다.

```
문서 업로드 → 청크 분할 → 임베딩 → Chroma 저장
이미지 업로드 → YOLO-World 탐지 + SAM-2 분할 → 설명 생성 → Chroma 저장
질문 입력   → 관련 내용 검색 → Claude API → 답변
```

이미지 분석은 로컬 GPU/모델 설치 없이 **Replicate API 호출**로 처리합니다.

코딩을 몰라도 아래 순서대로 터미널에 한 줄씩 입력하면 실행됩니다.

---

## 1. 준비물

- **Python 3.10 이상** — 터미널에서 `python3 --version` (Windows는 `python --version`)으로 확인.
  없다면 https://www.python.org/downloads/ 에서 설치하세요.
  (Windows 설치 시 **"Add Python to PATH"** 체크박스를 꼭 선택하세요.)
- **Anthropic API 키** — https://platform.claude.com 에 가입 → API Keys 메뉴에서 키 발급.
  `sk-ant-...` 로 시작하는 문자열입니다.
- **Replicate API 토큰** (이미지 분석용) — https://replicate.com 에 가입 →
  https://replicate.com/account/api-tokens 에서 토큰 발급. `r8_...` 로 시작합니다.
  (문서 챗봇만 쓸 거라면 없어도 됩니다. 이미지 분석 시에만 필요.)

## 2. 프로젝트 폴더로 이동

터미널(맥: 터미널 앱, 윈도우: PowerShell)을 열고:

```bash
cd first-contributions/rag-chatbot
```

(폴더 위치가 다르면 실제 경로에 맞게 바꿔 주세요.)

## 3. 가상환경 만들기 (처음 한 번만)

가상환경은 이 프로젝트 전용의 독립된 파이썬 공간입니다.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

성공하면 프롬프트 앞에 `(venv)` 가 표시됩니다.
> 다음에 다시 실행할 때도 `activate` 명령만 다시 입력하면 됩니다.

## 4. 필요한 패키지 설치 (처음 한 번만)

```bash
pip install -r requirements.txt
```

몇 분 걸릴 수 있습니다. 오류 없이 끝나면 성공입니다.

## 5. API 키 등록

**macOS / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-여기에-본인-키"
export REPLICATE_API_TOKEN="r8_여기에-본인-토큰"   # 이미지 분석 쓸 때만
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-여기에-본인-키"
$env:REPLICATE_API_TOKEN = "r8_여기에-본인-토큰"   # 이미지 분석 쓸 때만
```

> 터미널을 새로 열면 사라지므로, 실행 전마다 다시 입력해야 합니다.

## 6. 서버 실행

```bash
uvicorn main:app --reload
```

`Uvicorn running on http://127.0.0.1:8000` 메시지가 보이면 성공입니다.

## 7. 사용하기

브라우저에서 **http://127.0.0.1:8000** 을 엽니다.

1. **문서 업로드**: 파일 선택 → "업로드" 버튼. (.txt, .md, .pdf 지원)
   - 첫 업로드 때는 임베딩 모델(~80MB)을 자동 다운로드하므로 조금 오래 걸립니다.
2. **이미지 분석**: 이미지 선택 → (선택) 탐지할 클래스 입력 → "분석" 버튼.
   - YOLO-World가 객체를 탐지하고 SAM-2가 영역을 분할합니다. 결과(탐지 표 + 마스크 이미지)가
     화면에 표시되고, 설명은 벡터DB에 저장되어 이후 질문에서 검색됩니다.
   - Replicate 클라우드에서 실행되므로 수십 초 걸릴 수 있습니다.
3. **질문**: 입력창에 질문을 쓰고 "질문" 버튼 (또는 Enter).
   - 문서 내용과 이미지 분석 결과를 함께 검색해 답변합니다.
4. 답변 아래에 참고한 출처(문서/이미지 이름)가 함께 표시됩니다.

서버를 끄려면 터미널에서 `Ctrl + C` 를 누르세요.

## 자주 묻는 문제

| 증상 | 해결 방법 |
|---|---|
| `uvicorn: command not found` | 3번(가상환경 activate)과 4번(패키지 설치)을 다시 확인 |
| "Claude API 키가 올바르지 않습니다" | 5번의 API 키를 다시 확인하고 같은 터미널에서 서버 재실행 |
| PDF에서 텍스트를 못 찾음 | 스캔 이미지로 된 PDF는 텍스트 추출이 안 됩니다 (추후 OCR 필요) |
| 한국어 검색 정확도가 낮음 | MVP는 기본 임베딩 모델(영어 중심)을 사용합니다. 다음 단계에서 다국어 임베딩 모델로 교체 가능 |

## 다음 단계 (로드맵)

- [x] 이미지 업로드 + SAM-2(Replicate API) 연동 → 객체 영역 분할
- [x] YOLO-World(Replicate API) 연동 → 객체 탐지 (오픈 어휘)
- [x] 이미지 설명을 벡터DB에 함께 저장해서 문서+이미지 통합 검색
- [ ] 탐지된 객체 영역을 Claude Vision으로 상세 캡션 생성
- [ ] 대화 기록(멀티턴) 지원
- [ ] 한국어 특화 임베딩 모델로 교체

## 파일 구성

```
rag-chatbot/
├── main.py              # FastAPI 백엔드 (문서/이미지 업로드, 검색, 답변 생성)
├── image_analysis.py    # YOLO-World(탐지) + SAM-2(분할) Replicate 호출 모듈
├── static/index.html    # 웹 UI (문서/이미지 업로드 + 채팅 화면)
├── requirements.txt     # 필요한 파이썬 패키지 목록
├── chroma_db/           # 벡터DB 저장 폴더 (실행하면 자동 생성)
└── README.md            # 이 문서
```
