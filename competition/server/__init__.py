"""양돈 AI 백엔드 — 기존 도메인 모듈을 HTTP 로 노출한다.

## 이 패키지가 하지 않는 일

**계산을 다시 구현하지 않는다.** `src/` 의 68개 모듈이 정본이고, 여기는
얇은 어댑터다. 서버가 자기 산식을 갖는 순간 같은 농장에 대해 CLI 와 API 가
다른 답을 하게 된다 — 이 프로젝트는 화면과 모듈이 갈렸던 사고를 이미
여러 번 겪었다(등록 화면의 분만사 단위·복당 이유두수 12 vs 11 등).

그래서 라우터 안에는 **입력 검증 · 모듈 호출 · 직렬화**만 있고 산술이 없다.
테스트가 API 응답과 모듈 출력을 직접 대조한다.

## 구성

    server/app.py          FastAPI 앱 + 정적 파일 서빙
    server/db.py           SQLite (농장 등록 저장)
    server/schemas.py      요청·응답 모델
    server/routers/        farms · capacity · breeding

    python -m uvicorn competition.server.app:app --reload
"""
