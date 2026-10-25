@echo off
rem Double-click to measure your hand and save matching gesture thresholds.
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py --calibrate %*
pause
