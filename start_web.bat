@echo off
echo Starting DialogueAI Web Application...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing/updating dependencies...
pip install -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create a .env file with your API keys:
    echo OPENAI_API_KEY=your_openai_key_here
    echo GOOGLE_API_KEY=your_google_key_here
    echo FLASK_SECRET_KEY=your_secret_key_here
    echo.
    echo The application will work with SentenceTransformers as fallback.
    pause
)

REM Start the web application
echo.
echo Starting web server at http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python web_app.py

pause
