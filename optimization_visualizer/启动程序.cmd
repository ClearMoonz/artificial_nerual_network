@echo off
cd /d "%~dp0"
python "gradient_descent_visualizer\main.py"
if errorlevel 1 pause
