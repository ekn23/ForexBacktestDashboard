#!/usr/bin/env python3
"""
Simple runner script for the Forex Backtest Dashboard
Uses uvicorn (ASGI server) instead of gunicorn (WSGI server) for FastAPI compatibility
"""

import uvicorn
import os

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("🚀 Starting Forex Backtest Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("📁 Upload your CSV files using the web interface")
    print("")
    
    # Run with uvicorn (ASGI server)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )