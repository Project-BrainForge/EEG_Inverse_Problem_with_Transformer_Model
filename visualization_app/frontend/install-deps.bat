@echo off
echo Installing frontend dependencies...
echo This may take 3-5 minutes. Please wait...
echo.

REM Clean up old installations
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del /f /q package-lock.json

echo Cleaned up old dependencies.
echo.

REM Install dependencies
npm install

if %errorlevel% == 0 (
    echo.
    echo ========================================
    echo Installation successful!
    echo ========================================
    echo.
    echo You can now run: npm start
    echo.
) else (
    echo.
    echo ========================================
    echo Installation failed!
    echo ========================================
    echo.
    echo Please check the error messages above.
    echo.
)

pause

