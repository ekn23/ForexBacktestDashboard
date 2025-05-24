"""
Base Strategy Class for Forex Trading Strategies
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    """
    
    def __init__(self, symbol: str, timeframe: str, initial_balance: float = 10000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.trades = []
        self.open_positions = []
        self.parameters = {}
        
    @abstractmethod
    def on_tick(self, data: pd.Series) -> Optional[Dict]:
        """
        Called on each new price tick
        Should return trading signal or None
        Format: {'action': 'BUY'/'SELL'/'CLOSE', 'lot_size': 0.1, 'sl': price, 'tp': price}
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """
        Return strategy parameters for optimization
        """
        pass
    
    def add_indicator(self, data: pd.DataFrame, name: str, **kwargs) -> pd.DataFrame:
        """
        Add technical indicators to the data
        """
        if name == 'sma':
            period = kwargs.get('period', 14)
            data[f'sma_{period}'] = data['Close'].rolling(window=period).mean()
        
        elif name == 'ema':
            period = kwargs.get('period', 14)
            data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
        
        elif name == 'rsi':
            period = kwargs.get('period', 14)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        elif name == 'macd':
            fast = kwargs.get('fast', 12)
            slow = kwargs.get('slow', 26)
            signal = kwargs.get('signal', 9)
            
            ema_fast = data['Close'].ewm(span=fast).mean()
            ema_slow = data['Close'].ewm(span=slow).mean()
            data['macd'] = ema_fast - ema_slow
            data['macd_signal'] = data['macd'].ewm(span=signal).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        return data
    
    def execute_trade(self, signal: Dict, current_price: float, timestamp: str):
        """
        Execute a trade based on the signal
        """
        if signal['action'] in ['BUY', 'SELL']:
            trade = {
                'id': len(self.trades) + 1,
                'symbol': self.symbol,
                'action': signal['action'],
                'entry_price': current_price,
                'lot_size': signal.get('lot_size', 0.1),
                'entry_time': timestamp,
                'sl': signal.get('sl'),
                'tp': signal.get('tp'),
                'status': 'OPEN'
            }
            self.open_positions.append(trade)
            self.trades.append(trade)
        
        elif signal['action'] == 'CLOSE':
            # Close all open positions
            for position in self.open_positions:
                position['exit_price'] = current_price
                position['exit_time'] = timestamp
                position['status'] = 'CLOSED'
                
                # Calculate P&L
                if position['action'] == 'BUY':
                    pnl = (position['exit_price'] - position['entry_price']) * position['lot_size'] * 100000
                else:  # SELL
                    pnl = (position['entry_price'] - position['exit_price']) * position['lot_size'] * 100000
                
                position['pnl'] = pnl
                self.balance += pnl
            
            self.open_positions = []
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        """
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100
        gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'balance': round(self.balance, 2)
        }