@echo off
title DialogueAI Launcher
cls

echo.
echo =============================================================
echo                    DialogueAI Launcher
echo =============================================================
echo.
echo This will start your DialogueAI web application.
echo The browser will open automatically when ready.
echo.
echo Press any key to continue, or Ctrl+C to cancel...
pause >nul

echo.
echo Starting DialogueAI...
echo.

python start_dialogueai.py

echo.
echo =============================================================
echo Press any key to exit...
pause >nul
