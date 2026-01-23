@echo off
cls
echo Starting Playback in doom-rl environment...
conda run -n doom-rl python play.py --episodes 5 %*
pause
