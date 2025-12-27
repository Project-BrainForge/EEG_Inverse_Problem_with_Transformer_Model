@echo off
echo Restarting Backend Server...
echo.

REM Kill any existing Python processes
taskkill /F /IM python.exe >nul 2>&1

echo Waiting 2 seconds...
timeout /t 2 /nobreak >nul

cd backend

REM Clear Python cache
if exist __pycache__ rmdir /s /q __pycache__
if exist .pytest_cache rmdir /s /q .pytest_cache

echo Starting backend server...
echo Backend will run on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause

