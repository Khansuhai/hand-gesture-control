@echo off
rem Double-click to start hand-gesture control (starts PAUSED: show an open palm to begin).
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --start-paused %*
if errorlevel 1 pause
