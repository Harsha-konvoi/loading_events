@echo off
REM Activate Conda base
CALL C:\ProgramData\Miniconda3\condabin\conda.bat

REM Activate your environment
CALL conda activate deploy

REM Navigate to your script folder
cd C:\Code_base\shock_model-optimized

REM Run your Python scripts sequentially
python main.py