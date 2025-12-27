@echo off
echo Starting EEG Visualization Backend...
echo.

cd backend

echo Activating virtual environment...
if exist "..\..\venv\Scripts\activate.bat" (
    call ..\..\venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

echo.
echo Installing/checking dependencies...
pip install -r requirements.txt

echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause

