@echo off
REM Быстрая установка окружения через conda
conda env update -f environment.yml --prune
conda activate translation-evaluator
