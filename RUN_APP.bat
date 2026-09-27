@echo off
REM MedTriageAI+ Production App Launcher
REM Windows 10 | Python 3.13 | Streamlit

echo.
echo ========================================
echo   MedTriageAI+ v1.0
echo   AI Emergency Operations Center
echo ========================================
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo [2/3] Checking dependencies...
pip install -q streamlit pandas

REM Run the app
echo [3/3] Launching MedTriageAI+ on localhost:8501...
echo.
echo ========================================
echo Opening: http://localhost:8501
echo Press CTRL+C to stop
echo ========================================
echo.

streamlit run app_production.py

pause
