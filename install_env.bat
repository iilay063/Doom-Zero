@echo off
cls
echo ==================================================
echo      Doom RL Project - Easy Installer
echo ==================================================
echo.
echo This script will:
echo 1. Create a Conda environment named 'doom-rl' with Python 3.11
echo 2. Install all necessary libraries (VizDoom, etc.)
echo.
pause

echo.
echo [1/2] Creating Conda Environment (Python 3.11)...
call conda create -y -n doom-rl python=3.11

echo.
echo [2/2] Installing Dependencies...
call conda run -n doom-rl pip install -r requirements.txt

echo.
echo ==================================================
echo      Installation Complete! 
echo ==================================================
echo.
echo To start training, run: run_train.bat
echo To play, run: run_play.bat
echo.
pause
