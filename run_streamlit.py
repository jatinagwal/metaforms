#!/usr/bin/env python3
"""Runner script for Text-to-JSON Conversion Web App"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not set - you can still explore the interface")
    
    print("🌐 Starting web app...")
    
    # Run Streamlit
    streamlit_path = Path.home() / ".local" / "bin" / "streamlit"
    subprocess.run([
        str(streamlit_path), "run", 
        "src/web/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]) 