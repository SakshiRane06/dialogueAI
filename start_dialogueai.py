#!/usr/bin/env python3
"""
DialogueAI Startup Script
Complete setup and launch script with diagnostics
"""

import os
import sys
import time
import subprocess
import webbrowser
import threading
from pathlib import Path

def print_header():
    """Print startup header"""
    print("\n" + "="*70)
    print("🚀 DialogueAI Web Application Startup")
    print("="*70)

def check_python():
    """Check Python version"""
    print("🐍 Checking Python...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro} detected")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher required")
        return False
    
    print("✅ Python version OK")
    return True

def check_files():
    """Check required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        "simple_web_app.py",
        "templates",
        "templates/base.html",
        "templates/index.html", 
        "templates/processed.html",
        "templates/result.html"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Error: {len(missing_files)} required files missing")
        return False
    
    print("✅ All required files present")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Try to import Flask to see if it's already installed
        import flask
        print("✅ Flask already installed")
        return True
    except ImportError:
        print("   Installing Flask...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "Flask", "Werkzeug", "Jinja2", "--user"
            ])
            print("✅ Flask installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing Flask: {e}")
            return False

def check_port():
    """Check if port 5000 is available"""
    print("\n🔌 Checking port 5000...")
    
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 5000))
    sock.close()
    
    if result == 0:
        print("⚠️  Port 5000 is already in use")
        print("   Trying to find and stop existing process...")
        
        try:
            # Try to kill any existing python processes on port 5000
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                         capture_output=True, check=False)
            time.sleep(2)
            
            # Check again
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 5000))
            sock.close()
            
            if result == 0:
                print("❌ Port 5000 still occupied. Please close other applications using port 5000")
                return False
            else:
                print("✅ Port 5000 is now available")
                return True
        except Exception as e:
            print(f"❌ Error checking port: {e}")
            return False
    else:
        print("✅ Port 5000 is available")
        return True

def create_directories():
    """Create required directories"""
    print("\n📂 Creating directories...")
    
    dirs = ["uploads", "outputs", "static"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Directory: {dir_name}")

def open_browser_delayed():
    """Open browser after delay"""
    print("⏳ Waiting 3 seconds for server to start...")
    time.sleep(3)
    
    url = "http://localhost:5000"
    print(f"🌐 Opening browser: {url}")
    
    try:
        webbrowser.open(url)
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"   Please manually open: {url}")

def start_server():
    """Start the Flask server"""
    print("\n🚀 Starting DialogueAI server...")
    print("📍 URL: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 70)
    
    # Start browser opener in background
    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
    browser_thread.start()
    
    try:
        # Import and run the Flask app
        import simple_web_app
        simple_web_app.app.run(
            debug=False,  # Disable debug to avoid reloader issues
            host='127.0.0.1', 
            port=5000, 
            threaded=True,
            use_reloader=False  # Disable reloader
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure no other programs are using port 5000")
        print("2. Try running as administrator")
        print("3. Check Windows Firewall settings")

def main():
    """Main function"""
    print_header()
    
    # Run all checks
    checks = [
        ("Python version", check_python),
        ("Required files", check_files), 
        ("Dependencies", install_dependencies),
        ("Port availability", check_port)
    ]
    
    for check_name, check_func in checks:
        if not check_func():
            print(f"\n❌ Setup failed at: {check_name}")
            input("\nPress Enter to exit...")
            return
    
    # Create directories
    create_directories()
    
    print("\n" + "="*70)
    print("✅ All checks passed! Starting server...")
    print("="*70)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
