@echo off
title FacePersona Web Application
cd /d "%~dp0"
echo ========================================================
echo Starting FacePersona Web Application...
echo ========================================================
echo.
echo Open your browser and go to:
echo   URL: http://127.0.0.1:5000
echo.
echo (Public Assessment Tool - Direct Access, No Login Required)
echo (Press Ctrl+C in this window to stop the server)
echo ========================================================
echo.
python app.py
pause
