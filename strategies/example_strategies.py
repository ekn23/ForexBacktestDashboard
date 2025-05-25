"""
Example Trading Strategies for Testing
"""
from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Optional

class MovingAverageCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy
    """
    
    def __init__(self, symbol: str, timeframe: str, fast_period: int = 10, slow_period: int = 20, initial_balance: float = 10000):
        super().__init__(symbol, timeframe, initial_balance)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
    
    def on_tick(self, data: pd.Series) -> Optional[Dict]:
        """
        Generate signals based on MA crossover
        """
        # Need at least slow_period + 1 bars for calculation
        if len(data) < self.slow_period + 1:
            return None
        
        # Calculate moving averages
        fast_ma = data['Close'].rolling(window=self.fast_period).mean().iloc[-1]
        slow_ma = data['Close'].rolling(window=self.slow_period).mean().iloc[-1]
        
        prev_fast_ma = data['Close'].rolling(window=self.fast_period).mean().iloc[-2]
        prev_slow_ma = data['Close'].rolling(window=self.slow_period).mean().iloc[-2]
        
        current_price = data['Close'].iloc[-1]
        
        # Golden Cross - Fast MA crosses above Slow MA
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma and not self.open_positions:
            return {
                'action': 'BUY',
                'lot_size': 0.1,
                'sl': current_price * 0.99,  # 1% stop loss
                'tp': current_price * 1.02   # 2% take profit
            }
        
        # Death Cross - Fast MA crosses below Slow MA
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma and not self.open_positions:
            return {
                'action': 'SELL',
                'lot_size': 0.1,
                'sl': current_price * 1.01,  # 1% stop loss
                'tp': current_price * 0.98   # 2% take profit
            }
        
        # Close positions if opposite signal
        elif self.open_positions:
            current_position = self.open_positions[0]['action']
            if (current_position == 'BUY' and prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma) or \
               (current_position == 'SELL' and prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma):
                return {'action': 'CLOSE'}
        
        return None
    
    def get_parameters(self) -> Dict:
        return self.parameters



class MACDStrategy(BaseStrategy):
    """
    MACD Signal Line Crossover Strategy
    """
    
    def __init__(self, symbol: str, timeframe: str, fast_ema: int = 12, slow_ema: int = 26, signal_ema: int = 9, initial_balance: float = 10000):
        super().__init__(symbol, timeframe, initial_balance)
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_ema = signal_ema
        self.parameters = {
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'signal_ema': signal_ema
        }
    
    def calculate_macd(self, prices: pd.Series):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=self.fast_ema).mean()
        ema_slow = prices.ewm(span=self.slow_ema).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_ema).mean()
        return macd_line.iloc[-1], signal_line.iloc[-1], macd_line.iloc[-2], signal_line.iloc[-2]
    
    def on_tick(self, data: pd.Series) -> Optional[Dict]:
        """
        Generate signals based on MACD crossover
        """
        if len(data) < self.slow_ema + self.signal_ema:
            return None
        
        macd, signal, prev_macd, prev_signal = self.calculate_macd(data['Close'])
        current_price = data['Close'].iloc[-1]
        
        # Bullish crossover - MACD crosses above signal
        if prev_macd <= prev_signal and macd > signal and not self.open_positions:
            return {
                'action': 'BUY',
                'lot_size': 0.1,
                'sl': current_price * 0.99,   # 1% stop loss
                'tp': current_price * 1.025   # 2.5% take profit
            }
        
        # Bearish crossover - MACD crosses below signal
        elif prev_macd >= prev_signal and macd < signal and not self.open_positions:
            return {
                'action': 'SELL',
                'lot_size': 0.1,
                'sl': current_price * 1.01,   # 1% stop loss
                'tp': current_price * 0.975   # 2.5% take profit
            }
        
        # Close on opposite signal
        elif self.open_positions:
            current_position = self.open_positions[0]['action']
            if (current_position == 'BUY' and prev_macd >= prev_signal and macd < signal) or \
               (current_position == 'SELL' and prev_macd <= prev_signal and macd > signal):
                return {'action': 'CLOSE'}
        
        return None
    
    def get_parameters(self) -> Dict:
        return self.parameters