"""
DialogueAI Web Application Launcher
Starts the Flask server and opens the browser automatically
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)  # Wait for Flask to start
    url = "http://localhost:5000"
    print(f"🌐 Opening browser at {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print(f"Please manually open: {url}")

def main():
    print("🔧 DialogueAI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("web_app.py").exists():
        print("❌ web_app.py not found!")
        print("Please run this from the DialogueAI directory")
        input("Press Enter to exit...")
        return
    
    # Check if templates exist
    if not Path("templates").exists():
        print("❌ Templates directory not found!")
        print("Web interface files are missing")
        input("Press Enter to exit...")
        return
    
    print("✅ All files found")
    print("🚀 Starting web server...")
    print("⏳ Browser will open automatically in 2 seconds...")
    print("")
    
    # Start browser opener in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the Flask app
    try:
        with open('web_app.py', 'r', encoding='utf-8') as f:
            exec(f.read())
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
