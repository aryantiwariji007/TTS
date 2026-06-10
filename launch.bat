@echo off
title ScotAi TTS Studio
cd /d "%~dp0"
echo.
echo  Starting SCOT.AI TTS Studio...
echo  Open http://localhost:5000 in your browser
echo.
.venv\Scripts\python.exe app.py
pause
