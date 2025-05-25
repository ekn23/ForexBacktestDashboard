from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load and validate CSV data from forex files."""
    try:
        df = pd.read_csv(filepath)
        required_columns = ['datetime', 'Open', 'High', 'Low', 'Close']
        
        for col in required_columns:
            if col not in df.columns:
                # Try different naming conventions
                if col.lower() in df.columns:
                    df = df.rename(columns={col.lower(): col})
                elif col == 'datetime' and 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'datetime'})
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def calculate_rsi(df: pd.DataFrame, period=14):
    """Calculate RSI."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(df: pd.DataFrame, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_atr(df: pd.DataFrame, period=14):
    """Calculate ATR."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return pd.Series(true_range).rolling(period).mean()

def smart_smc_inverted_strategy(df: pd.DataFrame, strategy_params=None):
    """Your Smart SMC EA Inverted MQL4 strategy"""
    # Your MQL4 parameters
    rsi_period = 14
    rsi_overbought = 70.0
    rsi_oversold = 30.0
    bb_period = 20
    bb_deviation = 2.0
    adx_threshold = 20.0
    sl_percent = 2.0
    tp_percent = 4.0
    
    # Calculate indicators
    rsi = calculate_rsi(df, period=rsi_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, period=bb_period, std_dev=bb_deviation)
    
    # Simplified SAR and ADX
    sar = df['Close'].rolling(window=5).mean()
    atr = calculate_atr(df, period=14)
    adx = (atr / df['Close']) * 100
    
    trades = []
    signals = []
    
    # Your inverted logic from MQL4
    for i in range(max(rsi_period, bb_period), len(df)):
        close = df.iloc[i]['Close']
        current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
        current_adx = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 25
        current_sar = sar.iloc[i]
        current_bb_upper = bb_upper.iloc[i]
        current_bb_lower = bb_lower.iloc[i]
        current_bb_middle = bb_middle.iloc[i]
        
        # Your inverted buy logic: adx < threshold && rsi > overbought && close > bb_upper && close < sar
        open_buy = (current_adx < adx_threshold and 
                   current_rsi > rsi_overbought and 
                   close > current_bb_upper and 
                   close < current_sar)
        
        # Your inverted sell logic: adx < threshold && rsi < oversold && close < bb_lower && close > sar  
        open_sell = (current_adx < adx_threshold and 
                    current_rsi < rsi_oversold and 
                    close < current_bb_lower and 
                    close > current_sar)
        
        if open_buy:
            entry_price = close
            entry_time = df.iloc[i]['datetime']
            
            # Your MQL4 SL/TP calculations
            sl = entry_price - (sl_percent / 100.0) * entry_price
            tp = entry_price + (tp_percent / 100.0) * entry_price
            
            # Find exit point
            exit_price = tp
            exit_time = entry_time
            
            for j in range(i + 1, min(i + 100, len(df))):
                future_close = df.iloc[j]['Close']
                future_rsi = rsi.iloc[j] if not pd.isna(rsi.iloc[j]) else 50
                future_sar = sar.iloc[j]
                future_bb_middle = bb_middle.iloc[j]
                
                if future_close <= sl:
                    exit_price = sl
                    exit_time = df.iloc[j]['datetime']
                    break
                elif future_close >= tp:
                    exit_price = tp
                    exit_time = df.iloc[j]['datetime']
                    break
                elif future_rsi < 50 or future_close > future_sar or future_close < future_bb_middle:
                    exit_price = future_close
                    exit_time = df.iloc[j]['datetime']
                    break
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': 'BUY',
                'direction': 'BUY'
            })
            
            signals.append({
                'time': entry_time,
                'price': entry_price,
                'type': 'BUY'
            })
        
        elif open_sell:
            entry_price = close
            entry_time = df.iloc[i]['datetime']
            
            sl = entry_price + (sl_percent / 100.0) * entry_price
            tp = entry_price - (tp_percent / 100.0) * entry_price
            
            exit_price = tp
            exit_time = entry_time
            
            for j in range(i + 1, min(i + 100, len(df))):
                future_close = df.iloc[j]['Close']
                future_rsi = rsi.iloc[j] if not pd.isna(rsi.iloc[j]) else 50
                future_sar = sar.iloc[j]
                future_bb_middle = bb_middle.iloc[j]
                
                if future_close >= sl:
                    exit_price = sl
                    exit_time = df.iloc[j]['datetime']
                    break
                elif future_close <= tp:
                    exit_price = tp
                    exit_time = df.iloc[j]['datetime']
                    break
                elif future_rsi > 50 or future_close < future_sar or future_close > future_bb_middle:
                    exit_price = future_close
                    exit_time = df.iloc[j]['datetime']
                    break
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': 'SELL',
                'direction': 'SELL'
            })
            
            signals.append({
                'time': entry_time,
                'price': entry_price,
                'type': 'SELL'
            })
    
    return {
        'trades': trades,
        'signals': signals
    }

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart SMC EA Inverted Test</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-4">
            <h1>Your Smart SMC EA Inverted Strategy Test</h1>
            <div class="card">
                <div class="card-body">
                    <button onclick="testStrategy()" class="btn btn-primary">Test Your MQL4 Strategy</button>
                    <div id="results" class="mt-3"></div>
                </div>
            </div>
        </div>
        
        <script>
        async function testStrategy() {
            document.getElementById('results').innerHTML = 'Testing your Smart SMC EA Inverted strategy...';
            try {
                const response = await fetch('/test');
                const result = await response.json();
                document.getElementById('results').innerHTML = 
                    `<h4>Your Strategy Results:</h4>
                     <p>Total Trades: ${result.total_trades}</p>
                     <p>Winning Trades: ${result.winning_trades}</p>
                     <p>Win Rate: ${result.win_rate}%</p>
                     <p>Total Profit: ${result.total_profit} pips</p>`;
            } catch (error) {
                document.getElementById('results').innerHTML = 'Error testing strategy: ' + error.message;
            }
        }
        </script>
    </body>
    </html>
    ''')

@app.route('/test')
def test_strategy():
    # Test with available data
    data_files = []
    for file in os.listdir('.'):
        if file.endswith('.csv') and any(pair in file for pair in ['EURUSD', 'GBPUSD', 'USDJPY']):
            data_files.append(file)
    
    if not data_files:
        return jsonify({'error': 'No forex data files found'})
    
    # Use first available file
    df = load_csv_data(data_files[0])
    if df.empty:
        return jsonify({'error': 'Could not load data'})
    
    # Run your Smart SMC EA Inverted strategy
    result = smart_smc_inverted_strategy(df)
    
    # Calculate metrics
    trades = result['trades']
    total_trades = len(trades)
    
    if total_trades == 0:
        return jsonify({
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0,
            'total_profit': 0
        })
    
    winning_trades = 0
    total_profit = 0
    
    for trade in trades:
        if trade['type'] == 'BUY':
            profit = (trade['exit_price'] - trade['entry_price']) * 10000
        else:
            profit = (trade['entry_price'] - trade['exit_price']) * 10000
        
        total_profit += profit
        if profit > 0:
            winning_trades += 1
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    return jsonify({
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': round(win_rate, 1),
        'total_profit': round(total_profit, 1)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)