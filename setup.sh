#!/bin/bash

# Forex Backtest Dashboard Setup Script
echo "🚀 Starting Forex Backtest Dashboard setup..."

# Create data directory if it doesn't exist
mkdir -p data
echo "📁 Created data directory"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Install required Python packages
echo "📦 Installing Python dependencies..."
pip3 install fastapi uvicorn pandas numpy plotly python-multipart

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Kill any existing processes on port 5000
echo "🔄 Stopping any existing processes on port 5000..."
pkill -f "uvicorn.*5000" 2>/dev/null || true
lsof -ti:5000 | xargs kill -9 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 2

# Start the FastAPI server
echo "🌐 Starting Forex Backtest Dashboard on port 5000..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo ""
echo "📝 Instructions:"
echo "1. Upload your CSV files using the web interface"
echo "2. CSV files should follow the format: SYMBOL_Candlestick_TIMEFRAME_TYPE_*.csv"
echo "3. Example: EURUSD_Candlestick_5_M_BID_20240101.csv"
echo ""
echo "🔧 To stop the server, press Ctrl+C"
echo ""

# Start the server with auto-reload for development
python3 -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
