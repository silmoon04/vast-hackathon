@echo off
rem Street Time Machine: UI, outputs and the live question engine on http://127.0.0.1:8780/
rem Extra args go to uvicorn (e.g. serve.cmd --port 8781). Set STM_WARM=0 to skip the startup warm-up.
cd /d "%~dp0"
if not defined STM_PY if exist ".venv\Scripts\python.exe" set "STM_PY=%~dp0.venv\Scripts\python.exe"
if not defined STM_PY if exist "..\.venv\Scripts\python.exe" set "STM_PY=%~dp0..\.venv\Scripts\python.exe"
if not defined STM_PY set "STM_PY=python"
"%STM_PY%" -m uvicorn stm.server:app --host 127.0.0.1 --port 8780 %*
