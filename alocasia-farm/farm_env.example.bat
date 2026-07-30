@echo off
rem ────────────────────────────────────────────────────────────────
rem  이 파일을 farm_env.bat 로 복사한 뒤 개인키를 채우세요.
rem  farm_env.bat 는 깃에 올라가지 않습니다 (키가 새 나가지 않게).
rem ────────────────────────────────────────────────────────────────

rem 로보플로우 Private API Key (공개키는 막힙니다)
set ROBOFLOW_API_KEY=npP6TPQd5KvbAAXiTLsl

rem 워크플로 방식
set ROBOFLOW_WORKSPACE=s-workspace-br86f
set ROBOFLOW_WORKFLOW_ID=find-old-leaf-and-others

rem 모델 방식을 쓸 거면 위 두 줄을 지우고 아래를 쓰세요
rem set ROBOFLOW_MODEL_TOP=모델이름/1
rem set ROBOFLOW_MODEL_STAGE=모델이름/1

rem 탐지 민감도 (낮출수록 잎을 더 많이 잡음, 기본 25)
rem set CONFIDENCE=15

rem 저장 파일 위치 (기본: 이 폴더의 farm.db)
rem set FARM_DB=D:\백업\farm.db

rem 포트 고정 (install.bat 이 자동으로 넣어 줍니다).
rem 정해 두면 늘 이 포트를 쓰고, 안 정하면 8123 부터 빈 번호를 찾습니다.
rem set FARM_PORT=8123
