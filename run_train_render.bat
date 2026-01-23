@echo off
cls
echo Starting Training (Visual) in doom-rl environment...
echo Use --load to resume training, e.g.: run_train.bat --load
call "C:\Users\iilay\miniconda3\Scripts\activate.bat" doom-rl

set /p steps="Enter number of steps (default 100000): "
if "%steps%"=="" set steps=100000

echo Running for %steps% steps...
python train.py --steps %steps% --render %*
conda deactivate
pause
