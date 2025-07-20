#!/usr/bin/env python3
"""Runner script for Text-to-JSON Conversion API"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not set - conversion endpoints won't work")
    
    print("🚀 Starting API at http://localhost:8000/docs")
    
    # Try uvicorn, fallback to module
    uvicorn_cmd = [str(Path.home() / ".local/bin/uvicorn"), "src.api.main:app"]
    try:
        subprocess.run(uvicorn_cmd + ["--host", "0.0.0.0", "--port", "8000", "--reload"])
    except FileNotFoundError:
        subprocess.run([sys.executable, "-m", "uvicorn"] + uvicorn_cmd[1:] + ["--host", "0.0.0.0", "--port", "8000", "--reload"]) 