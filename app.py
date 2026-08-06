"""
Local Development Web Server Launcher.
Runs the Flask REST API and serves the Vercel public UI locally on http://localhost:5000.
Run with: python D:\Volatality_checker\app.py
"""

import sys
import os

# Ensure api directory is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from api.index import app

if __name__ == "__main__":
    print("=" * 70)
    print(" 🚀 SPY Options Volatility Checker Web Server Started!")
    print(" 🌐 Local Dashboard URL: http://localhost:5000")
    print(" 📦 Vercel Ready: Push project root to GitHub & deploy on Vercel")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
