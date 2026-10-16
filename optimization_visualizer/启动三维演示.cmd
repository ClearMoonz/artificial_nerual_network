@echo off
cd /d "%~dp0"
python "gradient_descent_visualizer\main.py" --3d
if errorlevel 1 pause
