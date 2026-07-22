@echo off
setlocal enabledelayedexpansion
title EngCMMS Server

rem ============================================================
rem  EngCMMS launcher - double-click to install (first time)
rem  and start the app, then open it in your browser.
rem
rem  If your files are NOT in C:\EngCMMS, change APP_DIR below.
rem  To store the database/uploads somewhere else (e.g. D drive),
rem  change DATA_DIR below to, for example:  D:\EngCMMS
rem ============================================================
set "APP_DIR=C:\EngCMMS"
set "DATA_DIR=C:\EngCMMS\data"

set "ENGCMMS_DATA_DIR=%DATA_DIR%"

cd /d "%APP_DIR%"
if not exist "run.py" (
  echo.
  echo [!] Could not find run.py in "%APP_DIR%".
  echo     Open this .bat in Notepad and set APP_DIR to the folder
  echo     that actually contains run.py, then try again.
  echo.
  pause
  exit /b 1
)

rem --- Locate a Python interpreter -----------------------------
set "PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe"  set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda31\python.exe" set "PY=%USERPROFILE%\anaconda31\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe"  set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY (
  echo.
  echo [!] Could not find Python. Install Python 3.10+ (python.org)
  echo     or Anaconda, then run this again.
  echo.
  pause
  exit /b 1
)
echo Using Python: %PY%

rem --- Install dependencies only if Flask is missing -----------
"%PY%" -c "import flask" 1>nul 2>nul
if errorlevel 1 (
  echo Installing dependencies ^(first run only, may take a minute^)...
  "%PY%" -m pip install -r requirements.txt
)

rem --- Open the browser a few seconds after the server starts ---
start "" /min cmd /c "timeout /t 5 >nul & start "" http://127.0.0.1:5000"

echo.
echo ============================================================
echo  EngCMMS is starting...
echo  Open http://127.0.0.1:5000  (sign in: admin / admin)
echo  Keep this window open. Press Ctrl+C or close it to stop.
echo ============================================================
echo.

"%PY%" run.py --seed

echo.
echo EngCMMS has stopped.
pause
