@echo off
setlocal
title MTL Studio
cd /d "%~dp0"
rem Always use UTF-8, whatever the Windows language (fixes cp949 errors on Korean Windows)
set PYTHONUTF8=1

rem ---- 1. Check the app's Python environment is here ----
if not exist ".venv\Scripts\python.exe" (
  echo [!] Could not find the .venv folder here.
  echo     Put this file in the mtl-studio folder, next to run.py.
  pause
  exit /b 1
)

rem ---- 2. Make sure Ollama is running (starts it if needed) ----
echo Checking Ollama...
curl.exe -s -o nul --max-time 2 http://localhost:11434
if errorlevel 1 (
  echo Starting Ollama...
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
  ) else (
    start "Ollama" /min ollama serve
  )
  for /l %%i in (1,1,30) do (
    curl.exe -s -o nul --max-time 2 http://localhost:11434 && goto ollama_ok
    timeout /t 1 /nobreak >nul
  )
  echo [!] Ollama did not start. Open Ollama from the Start menu, then run this file again.
  pause
  exit /b 1
)
:ollama_ok
echo Ollama is running.

rem ---- 3. Open the browser automatically once the app is ready ----
echo Starting MTL Studio. Your browser will open when it is ready...
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 180;$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:8000/api/health' | Out-Null; Start-Process 'http://127.0.0.1:8000'; break}catch{Start-Sleep -Seconds 1}}"

rem ---- 4. Start the app (uses the .venv Python directly, no activation needed) ----
".venv\Scripts\python.exe" run.py

echo.
echo MTL Studio has stopped.
pause
