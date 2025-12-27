@echo off
echo ========================================
echo   EEG Source Localization Visualizer
echo ========================================
echo.
echo This script will start both the backend and frontend servers.
echo.
echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:3000
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo Starting Backend Server...
start "EEG Backend" cmd /k "cd backend && python app.py"

timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "EEG Frontend" cmd /k "cd frontend && npm start"

echo.
echo Both servers are starting in separate windows.
echo Please wait for them to fully start (about 10-30 seconds).
echo.
echo The browser should open automatically to http://localhost:3000
echo.
echo To stop the servers, close both command windows.
echo.
pause

