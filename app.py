import os
import pandas as pd
import numpy as np
import json
import logging
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

DATA_DIR = "data"
UPLOAD_FOLDER = DATA_DIR
ALLOWED_EXTENSIONS = {'csv'}

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_filename(filename: str) -> dict:
    """Parse forex CSV filename to extract symbol and timeframe."""
    try:
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 4:
            symbol = parts[0]
            timeframe = f"{parts[2]}_{parts[3]}"
            return {"symbol": symbol, "timeframe": timeframe}
        else:
            symbol = parts[0] if parts else "UNKNOWN"
            timeframe = "1_M"
            return {"symbol": symbol, "timeframe": timeframe}
    except Exception as e:
        logger.error(f"Error parsing filename {filename}: {e}")
        return {"symbol": "UNKNOWN", "timeframe": "1_M"}

def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close',
            'Volume': 'Volume', 'Timestamp': 'Timestamp', 'Date': 'Timestamp',
            'Time': 'Timestamp', 'EntryTime': 'EntryTime', 'ExitTime': 'ExitTime',
            'EntryPrice': 'EntryPrice', 'ExitPrice': 'ExitPrice'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns in {filepath}: {missing_cols}")
            for col in missing_cols:
                df[col] = 0.0
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {filepath}: {e}")
        return pd.DataFrame()

def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate trading metrics from DataFrame."""
    if df.empty:
        return {"net_profit": 0, "total_trades": 0, "wins": 0, "losses": 0, "max_drawdown": 0}
    
    try:
        if 'EntryPrice' in df.columns and 'ExitPrice' in df.columns:
            df['pnl'] = df['ExitPrice'] - df['EntryPrice']
        else:
            df['pnl'] = df['Close'] - df['Open']
        
        df['pnl'] = df['pnl'].fillna(0)
        
        net_profit = float(df['pnl'].sum())
        total_trades = len(df)
        wins = int((df['pnl'] > 0).sum())
        losses = int((df['pnl'] <= 0).sum())
        
        cumulative_pnl = df['pnl'].cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = peak - cumulative_pnl
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0
        
        return {
            "net_profit": net_profit,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "max_drawdown": max_drawdown
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"net_profit": 0, "total_trades": 0, "wins": 0, "losses": 0, "max_drawdown": 0}

def load_backtest_stats():
    """Load and aggregate backtest statistics from all CSV files."""
    try:
        if not os.path.exists(DATA_DIR):
            return {"net_profit": [], "total_trades": [], "win": 0, "loss": 0, "max_drawdown": 0}
        
        net_profit_data = []
        total_trades_data = []
        total_wins = 0
        total_losses = 0
        max_drawdown = 0
        
        csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
        
        if not csv_files:
            logger.warning("No CSV files found in data directory")
            return {"net_profit": [], "total_trades": [], "win": 0, "loss": 0, "max_drawdown": 0}
        
        for filename in csv_files:
            filepath = os.path.join(DATA_DIR, filename)
            file_info = parse_filename(filename)
            
            df = load_csv_data(filepath)
            if df.empty:
                continue
                
            metrics = calculate_metrics(df)
            
            net_profit_data.append({
                "symbol": file_info["symbol"],
                "timeframe": file_info["timeframe"],
                "profit": metrics["net_profit"]
            })
            
            total_trades_data.append({
                "symbol": file_info["symbol"],
                "timeframe": file_info["timeframe"],
                "trades": metrics["total_trades"]
            })
            
            total_wins += metrics["wins"]
            total_losses += metrics["losses"]
            max_drawdown = max(max_drawdown, metrics["max_drawdown"])
        
        return {
            "net_profit": net_profit_data,
            "total_trades": total_trades_data,
            "win": total_wins,
            "loss": total_losses,
            "max_drawdown": max_drawdown
        }
    except Exception as e:
        logger.error(f"Error in load_backtest_stats: {e}")
        return {"net_profit": [], "total_trades": [], "win": 0, "loss": 0, "max_drawdown": 0}

@app.route('/')
def index():
    """Serve the main dashboard page."""
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Dashboard not found", 404

@app.route('/api/backtest_results')
def backtest_results():
    """Get aggregated backtest results."""
    try:
        stats = load_backtest_stats()
        return jsonify({
            "net_profit": stats["net_profit"],
            "total_trades": stats["total_trades"],
            "max_drawdown": stats["max_drawdown"]
        })
    except Exception as e:
        logger.error(f"Error in backtest_results endpoint: {e}")
        return jsonify({"error": f"Error loading backtest results: {str(e)}"}), 500

@app.route('/api/win_loss_pie')
def win_loss_pie():
    """Get win/loss data for pie chart."""
    try:
        stats = load_backtest_stats()
        return jsonify({"win": stats["win"], "loss": stats["loss"]})
    except Exception as e:
        logger.error(f"Error in win_loss_pie endpoint: {e}")
        return jsonify({"error": f"Error loading win/loss data: {str(e)}"}), 500

@app.route('/api/detailed_trades')
def detailed_trades():
    """Get detailed trade information."""
    try:
        details = []
        
        if not os.path.exists(DATA_DIR):
            return jsonify(details)
            
        csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
        
        for filename in csv_files:
            filepath = os.path.join(DATA_DIR, filename)
            file_info = parse_filename(filename)
            
            df = load_csv_data(filepath)
            if df.empty:
                continue
            
            for idx, row in df.iterrows():
                try:
                    if 'EntryPrice' in row and 'ExitPrice' in row and pd.notna(row['EntryPrice']) and pd.notna(row['ExitPrice']):
                        pnl = float(row['ExitPrice'] - row['EntryPrice'])
                    else:
                        pnl = float(row['Close'] - row['Open'])
                    
                    duration = None
                    if 'EntryTime' in row and 'ExitTime' in row:
                        try:
                            entry_time = pd.to_datetime(row['EntryTime'])
                            exit_time = pd.to_datetime(row['ExitTime'])
                            duration = (exit_time - entry_time).total_seconds()
                        except:
                            duration = None
                    
                    details.append({
                        "symbol": file_info["symbol"],
                        "timeframe": file_info["timeframe"],
                        "entry": str(row.get('EntryTime', '')) if 'EntryTime' in row else '',
                        "exit": str(row.get('ExitTime', '')) if 'ExitTime' in row else '',
                        "pnl": pnl,
                        "duration": duration
                    })
                except Exception as e:
                    logger.warning(f"Error processing row {idx} in {filename}: {e}")
                    continue
        
        return jsonify(details)
    except Exception as e:
        logger.error(f"Error in detailed_trades endpoint: {e}")
        return jsonify({"error": f"Error loading detailed trades: {str(e)}"}), 500

@app.route('/api/csv_data/<symbol>/<timeframe>')
def get_csv_data(symbol, timeframe):
    """Get raw CSV data for a specific symbol and timeframe."""
    try:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
        
        for filename in csv_files:
            file_info = parse_filename(filename)
            if file_info["symbol"] == symbol and file_info["timeframe"] == timeframe:
                filepath = os.path.join(DATA_DIR, filename)
                df = load_csv_data(filepath)
                
                if df.empty:
                    continue
                
                data = []
                for idx, row in df.iterrows():
                    data.append({
                        "timestamp": str(row.get('Timestamp', idx)),
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row.get('Volume', 0))
                    })
                
                return jsonify({"data": data})
        
        return jsonify({"error": f"No data found for {symbol} {timeframe}"}), 404
    except Exception as e:
        logger.error(f"Error in get_csv_data endpoint: {e}")
        return jsonify({"error": f"Error loading CSV data: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload CSV files to the data directory."""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_files.append(filename)
                logger.info(f"Uploaded file: {filename}")
        
        return jsonify({
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        })
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        return jsonify({"error": f"Error uploading files: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "data_dir_exists": os.path.exists(DATA_DIR)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)