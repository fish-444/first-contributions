@echo off
chcp 65001 >nul
title 알로카시아 스마트팜 - 로보플로우 설정
cd /d "%~dp0"

rem 실제 작업은 set_key.ps1 이 한다. 배치로 파일을 줄 단위로 고치려면
rem 따옴표 처리가 지저분해지고, BOM 없는 UTF-8 로 쓰기도 어렵다.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0set_key.ps1"

echo.
pause
