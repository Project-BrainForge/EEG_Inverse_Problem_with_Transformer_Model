@echo off
echo Starting EEG Visualization Frontend...
echo.

cd frontend

echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo.
echo Installing dependencies (this may take a few minutes on first run)...
if not exist "node_modules\" (
    npm install
) else (
    echo Dependencies already installed.
)

echo.
echo Starting React development server on http://localhost:3000
echo The browser will open automatically.
echo Press Ctrl+C to stop the server
echo.

npm start

pause

