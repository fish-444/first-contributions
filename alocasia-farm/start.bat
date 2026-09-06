@echo off
chcp 65001 >nul
title 알로카시아 스마트팜
cd /d "%~dp0"

rem 로보플로우 키는 farm_env.bat 에 따로 둡니다 (깃에 안 올라감).
rem farm_env.example.bat 를 farm_env.bat 로 복사해 키를 채우세요.
if exist "farm_env.bat" (
    call "farm_env.bat"
) else (
    echo [알림] farm_env.bat 가 없어 데모 모드로 켭니다.
    echo        farm_env.example.bat 를 farm_env.bat 로 복사해 키를 넣으면
    echo        로보플로우로 실제 분석합니다.
    echo.
)

rem 8123 이 막혀 있으면 다음 번호로 넘어갑니다 (윈도우가 예약해 둔 포트가 있음)
set PORT=8123
netstat -ano | findstr ":%PORT% " >nul && set PORT=8234
netstat -ano | findstr ":%PORT% " >nul && set PORT=8345

echo 브라우저를 엽니다 - http://127.0.0.1:%PORT%
echo 끄려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo.

rem 서버가 뜰 시간을 주고 브라우저를 띄웁니다
start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:%PORT%"

python -m uvicorn main:app --port %PORT%

echo.
echo 서버가 멈췄습니다. 저장된 식물 정보는 farm.db 에 남아 있습니다.
pause
