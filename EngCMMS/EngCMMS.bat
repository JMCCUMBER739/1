@echo off
setlocal enabledelayedexpansion
title EngCMMS Server

rem ============================================================
rem  EngCMMS launcher - double-click to install (first time)
rem  and start the app, then open it in your browser.
rem
rem  It automatically searches SEARCH_ROOT for run.py, so you do
rem  not need to know the exact folder. If your files are not on
rem  the C: drive, change SEARCH_ROOT below (e.g. to D:\EngCMMS).
rem  To store the database/uploads elsewhere, change DATA_DIR.
rem ============================================================
set "SEARCH_ROOT=C:\EngCMMS"
set "DATA_DIR=C:\EngCMMS\data"

rem To force a specific Python, put its full path here (recommended for
rem Anaconda users). Leave blank to auto-detect. Example:
rem   set "PY_OVERRIDE=C:\Users\jmccu\anaconda31\python.exe"
set "PY_OVERRIDE="

set "ENGCMMS_DATA_DIR=%DATA_DIR%"

rem --- Find the folder that contains run.py + the engcmms package
set "APP_DIR="
for /r "%SEARCH_ROOT%" %%F in (run.py) do (
  if not defined APP_DIR (
    if exist "%%~dpFengcmms\__init__.py" set "APP_DIR=%%~dpF"
  )
)

if not defined APP_DIR (
  echo.
  echo [!] Could not find run.py under "%SEARCH_ROOT%".
  echo     Make sure you extracted the EngCMMS files somewhere under
  echo     that folder, or edit SEARCH_ROOT at the top of this file
  echo     (for example D:\EngCMMS), then try again.
  echo.
  pause
  exit /b 1
)

cd /d "%APP_DIR%"
echo Found app in: %APP_DIR%

rem --- Locate a Python interpreter -----------------------------
set "PY=%PY_OVERRIDE%"
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
