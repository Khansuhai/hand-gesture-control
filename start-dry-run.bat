@echo off
rem Double-click to test gestures without controlling the mouse or keyboard.
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --dry-run %*
if errorlevel 1 pause
