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
    
    def __init__(self, symbol: str, timeframe: str, trading_config=None):
        from trading_config import TradingConfig, OANDATradeExecutor
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = trading_config or TradingConfig()
        self.executor = OANDATradeExecutor(self.config)
        
        self.initial_balance = self.config.starting_capital
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_balance = self.initial_balance
        
        self.trades = []
        self.open_positions = []
        self.parameters = {}
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_margin_used = 0.0
        self.max_drawdown_reached = 0.0
        
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
        Add technical indicators to the data using comprehensive indicator library
        """
        from indicators.technical_indicators import TechnicalIndicators
        from indicators.pattern_recognition import PatternRecognition
        
        # Moving Averages
        if name == 'sma':
            period = kwargs.get('period', 14)
            data[f'sma_{period}'] = TechnicalIndicators.sma(data['Close'], period)
        
        elif name == 'ema':
            period = kwargs.get('period', 14)
            data[f'ema_{period}'] = TechnicalIndicators.ema(data['Close'], period)
        
        elif name == 'wma':
            period = kwargs.get('period', 14)
            data[f'wma_{period}'] = TechnicalIndicators.wma(data['Close'], period)
        
        elif name == 'vwma' and 'Volume' in data.columns:
            period = kwargs.get('period', 14)
            data[f'vwma_{period}'] = TechnicalIndicators.vwma(data['Close'], data['Volume'], period)
        
        # Oscillators
        elif name == 'rsi':
            period = kwargs.get('period', 14)
            data[f'rsi_{period}'] = TechnicalIndicators.rsi(data['Close'], period)
        
        elif name == 'stochastic':
            k_period = kwargs.get('k_period', 14)
            d_period = kwargs.get('d_period', 3)
            k, d = TechnicalIndicators.stochastic(data['High'], data['Low'], data['Close'], k_period, d_period)
            data[f'stoch_k_{k_period}'] = k
            data[f'stoch_d_{d_period}'] = d
        
        elif name == 'williams_r':
            period = kwargs.get('period', 14)
            data[f'williams_r_{period}'] = TechnicalIndicators.williams_r(data['High'], data['Low'], data['Close'], period)
        
        elif name == 'cci':
            period = kwargs.get('period', 20)
            data[f'cci_{period}'] = TechnicalIndicators.cci(data['High'], data['Low'], data['Close'], period)
        
        # MACD
        elif name == 'macd':
            fast = kwargs.get('fast', 12)
            slow = kwargs.get('slow', 26)
            signal = kwargs.get('signal', 9)
            macd, signal_line, histogram = TechnicalIndicators.macd(data['Close'], fast, slow, signal)
            data['macd'] = macd
            data['macd_signal'] = signal_line
            data['macd_histogram'] = histogram
        
        # Bands and Channels
        elif name == 'bollinger_bands':
            period = kwargs.get('period', 20)
            std_dev = kwargs.get('std_dev', 2)
            upper, middle, lower = TechnicalIndicators.bollinger_bands(data['Close'], period, std_dev)
            data['bb_upper'] = upper
            data['bb_middle'] = middle
            data['bb_lower'] = lower
        
        elif name == 'keltner_channels':
            period = kwargs.get('period', 20)
            multiplier = kwargs.get('multiplier', 2)
            upper, middle, lower = TechnicalIndicators.keltner_channels(data['High'], data['Low'], data['Close'], period, multiplier)
            data['kc_upper'] = upper
            data['kc_middle'] = middle
            data['kc_lower'] = lower
        
        elif name == 'donchian_channels':
            period = kwargs.get('period', 20)
            upper, middle, lower = TechnicalIndicators.donchian_channels(data['High'], data['Low'], period)
            data['dc_upper'] = upper
            data['dc_middle'] = middle
            data['dc_lower'] = lower
        
        # Trend Indicators
        elif name == 'atr':
            period = kwargs.get('period', 14)
            data[f'atr_{period}'] = TechnicalIndicators.atr(data['High'], data['Low'], data['Close'], period)
        
        elif name == 'adx':
            period = kwargs.get('period', 14)
            adx, di_plus, di_minus = TechnicalIndicators.adx(data['High'], data['Low'], data['Close'], period)
            data[f'adx_{period}'] = adx
            data[f'di_plus_{period}'] = di_plus
            data[f'di_minus_{period}'] = di_minus
        
        elif name == 'parabolic_sar':
            af_start = kwargs.get('af_start', 0.02)
            af_increment = kwargs.get('af_increment', 0.02)
            af_max = kwargs.get('af_max', 0.2)
            data['parabolic_sar'] = TechnicalIndicators.parabolic_sar(data['High'], data['Low'], af_start, af_increment, af_max)
        
        # Ichimoku Cloud
        elif name == 'ichimoku':
            tenkan, kijun, senkou_a, senkou_b, chikou = TechnicalIndicators.ichimoku(data['High'], data['Low'], data['Close'])
            data['ichimoku_tenkan'] = tenkan
            data['ichimoku_kijun'] = kijun
            data['ichimoku_senkou_a'] = senkou_a
            data['ichimoku_senkou_b'] = senkou_b
            data['ichimoku_chikou'] = chikou
        
        # Volume Indicators
        elif name == 'obv' and 'Volume' in data.columns:
            data['obv'] = TechnicalIndicators.obv(data['Close'], data['Volume'])
        
        elif name == 'mfi' and 'Volume' in data.columns:
            period = kwargs.get('period', 14)
            data[f'mfi_{period}'] = TechnicalIndicators.mfi(data['High'], data['Low'], data['Close'], data['Volume'], period)
        
        # Advanced Oscillators
        elif name == 'awesome_oscillator':
            data['ao'] = TechnicalIndicators.awesome_oscillator(data['High'], data['Low'])
        
        elif name == 'accelerator_oscillator':
            data['ac'] = TechnicalIndicators.accelerator_oscillator(data['High'], data['Low'], data['Close'])
        
        elif name == 'aroon':
            period = kwargs.get('period', 14)
            aroon_up, aroon_down = TechnicalIndicators.aroon(data['High'], data['Low'], period)
            data[f'aroon_up_{period}'] = aroon_up
            data[f'aroon_down_{period}'] = aroon_down
        
        elif name == 'cmo':
            period = kwargs.get('period', 14)
            data[f'cmo_{period}'] = TechnicalIndicators.chande_momentum_oscillator(data['Close'], period)
        
        elif name == 'vortex':
            period = kwargs.get('period', 14)
            vi_plus, vi_minus = TechnicalIndicators.vortex(data['High'], data['Low'], data['Close'], period)
            data[f'vi_plus_{period}'] = vi_plus
            data[f'vi_minus_{period}'] = vi_minus
        
        elif name == 'elder_ray':
            period = kwargs.get('period', 13)
            bull_power, bear_power = TechnicalIndicators.elder_ray(data['High'], data['Low'], data['Close'], period)
            data[f'bull_power_{period}'] = bull_power
            data[f'bear_power_{period}'] = bear_power
        
        elif name == 'ultimate_oscillator':
            period1 = kwargs.get('period1', 7)
            period2 = kwargs.get('period2', 14)
            period3 = kwargs.get('period3', 28)
            data['ultimate_oscillator'] = TechnicalIndicators.ultimate_oscillator(data['High'], data['Low'], data['Close'], period1, period2, period3)
        
        # Pattern Recognition
        elif name == 'candlestick_patterns':
            patterns = PatternRecognition.detect_candlestick_patterns(data)
            for pattern_name, pattern_data in patterns.items():
                data[f'pattern_{pattern_name}'] = pattern_data
        
        elif name == 'support_resistance':
            window = kwargs.get('window', 20)
            min_touches = kwargs.get('min_touches', 2)
            levels = PatternRecognition.detect_support_resistance(data, window, min_touches)
            # Store as metadata since these are price levels, not time series
            data.attrs['support_levels'] = levels['support']
            data.attrs['resistance_levels'] = levels['resistance']
        
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