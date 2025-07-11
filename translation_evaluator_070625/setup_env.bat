@echo off
REM Создание и активация окружения через conda
conda env create -f environment.yml
conda activate translation-evaluator
