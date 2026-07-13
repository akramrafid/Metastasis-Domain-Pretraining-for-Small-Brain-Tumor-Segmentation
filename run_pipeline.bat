@echo off
set PYTHONPATH=.
set PYTHON_EXE=C:\Users\MSI\miniconda3\envs\dr_detection\python.exe

echo ==================================================
echo Starting End-to-End Brain Tumor SSL Pipeline
echo ==================================================

echo.
echo *** Running Arm 1 (Baseline pretraining on BraTS-GLI + Fine-tuning)
%PYTHON_EXE% scripts/run_arm1.py experiment@_global_=arm1_glioma_pretrain_baseline
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Arm 1 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo *** Running Arm 2 (Domain-matched pretraining on Pretreat-MetsToBrain + Fine-tuning)
%PYTHON_EXE% scripts/run_arm2.py experiment@_global_=arm2_metastasis_pretrain_proposed
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Arm 2 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo *** Running Arm 3 (Combined Pretraining + Loss Tradeoff Fine-tuning)
%PYTHON_EXE% scripts/run_arm3.py experiment@_global_=arm3_combined
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Arm 3 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo *** Running Arm 0 (Loss-tradeoff baseline Fine-tuning)
%PYTHON_EXE% scripts/run_arm0.py experiment@_global_=arm0_glioma_pretrain_lossfix
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Arm 0 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo *** Running External Evaluation on All Arms
%PYTHON_EXE% scripts/run_external_eval.py experiment@_global_=external_eval
if %ERRORLEVEL% neq 0 (
    echo [ERROR] External Evaluation failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo *** Generating Final Paper Tables and Metrics
%PYTHON_EXE% scripts/generate_paper_tables.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Table generation failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo ==================================================
echo Pipeline Completed Successfully!
echo ==================================================
