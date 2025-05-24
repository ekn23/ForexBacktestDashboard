"""
Advanced Multi-Indicator Strategy Examples
Demonstrates how to combine multiple indicators for sophisticated trading strategies
"""
from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Optional

class SuperTrendStrategy(BaseStrategy):
    """
    Multi-indicator strategy combining RSI, MACD, Bollinger Bands, and ADX
    """
    
    def __init__(self, symbol: str, timeframe: str, initial_balance: float = 10000):
        super().__init__(symbol, timeframe, initial_balance)
        self.parameters = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2,
            'adx_period': 14,
            'adx_threshold': 25
        }
    
    def on_tick(self, data: pd.DataFrame) -> Optional[Dict]:
        """Multi-indicator signal generation"""
        if len(data) < 50:
            return None
        
        # Add all required indicators
        data = self.add_indicator(data, 'rsi', period=self.parameters['rsi_period'])
        data = self.add_indicator(data, 'macd')
        data = self.add_indicator(data, 'bollinger_bands', period=self.parameters['bb_period'], std_dev=self.parameters['bb_std'])
        data = self.add_indicator(data, 'adx', period=self.parameters['adx_period'])
        
        # Get current values
        current_price = data['Close'].iloc[-1]
        rsi = data[f'rsi_{self.parameters["rsi_period"]}'].iloc[-1]
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        adx = data[f'adx_{self.parameters["adx_period"]}'].iloc[-1]
        
        # Check if we have valid data
        if pd.isna(rsi) or pd.isna(macd) or pd.isna(adx):
            return None
        
        # Strong trend confirmation required
        trend_strong = adx > self.parameters['adx_threshold']
        
        # Bullish conditions
        bullish_rsi = rsi < self.parameters['rsi_oversold']
        bullish_macd = macd > macd_signal
        bullish_bb = current_price <= bb_lower
        
        # Bearish conditions  
        bearish_rsi = rsi > self.parameters['rsi_overbought']
        bearish_macd = macd < macd_signal
        bearish_bb = current_price >= bb_upper
        
        # Entry signals - require multiple confirmations
        if trend_strong and not self.open_positions:
            if bullish_rsi and bullish_macd and bullish_bb:
                return {
                    'action': 'BUY',
                    'lot_size': 0.1,
                    'sl': current_price * 0.98,   # 2% stop loss
                    'tp': current_price * 1.04    # 4% take profit
                }
            
            elif bearish_rsi and bearish_macd and bearish_bb:
                return {
                    'action': 'SELL',
                    'lot_size': 0.1,
                    'sl': current_price * 1.02,   # 2% stop loss
                    'tp': current_price * 0.96    # 4% take profit
                }
        
        # Exit conditions
        elif self.open_positions:
            # Exit if RSI moves to neutral zone
            if 40 < rsi < 60:
                return {'action': 'CLOSE'}
        
        return None
    
    def get_parameters(self) -> Dict:
        return self.parameters

class IchimokuStrategy(BaseStrategy):
    """
    Ichimoku Cloud strategy with additional confirmations
    """
    
    def __init__(self, symbol: str, timeframe: str, initial_balance: float = 10000):
        super().__init__(symbol, timeframe, initial_balance)
        self.parameters = {
            'atr_period': 14,
            'rsi_period': 14
        }
    
    def on_tick(self, data: pd.DataFrame) -> Optional[Dict]:
        """Ichimoku-based signal generation"""
        if len(data) < 60:
            return None
        
        # Add indicators
        data = self.add_indicator(data, 'ichimoku')
        data = self.add_indicator(data, 'atr', period=self.parameters['atr_period'])
        data = self.add_indicator(data, 'rsi', period=self.parameters['rsi_period'])
        
        current_price = data['Close'].iloc[-1]
        tenkan = data['ichimoku_tenkan'].iloc[-1]
        kijun = data['ichimoku_kijun'].iloc[-1]
        senkou_a = data['ichimoku_senkou_a'].iloc[-1]
        senkou_b = data['ichimoku_senkou_b'].iloc[-1]
        atr = data[f'atr_{self.parameters["atr_period"]}'].iloc[-1]
        rsi = data[f'rsi_{self.parameters["rsi_period"]}'].iloc[-1]
        
        if pd.isna(tenkan) or pd.isna(kijun) or pd.isna(atr):
            return None
        
        # Ichimoku signals
        tk_cross_bullish = tenkan > kijun
        tk_cross_bearish = tenkan < kijun
        
        # Cloud analysis
        if not pd.isna(senkou_a) and not pd.isna(senkou_b):
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            
            above_cloud = current_price > cloud_top
            below_cloud = current_price < cloud_bottom
        else:
            above_cloud = below_cloud = False
        
        # Dynamic stop loss using ATR
        atr_multiplier = 2
        
        if not self.open_positions:
            # Bullish signal: TK cross bullish + price above cloud + RSI not overbought
            if tk_cross_bullish and above_cloud and rsi < 70:
                return {
                    'action': 'BUY',
                    'lot_size': 0.1,
                    'sl': current_price - (atr * atr_multiplier),
                    'tp': current_price + (atr * atr_multiplier * 2)
                }
            
            # Bearish signal: TK cross bearish + price below cloud + RSI not oversold
            elif tk_cross_bearish and below_cloud and rsi > 30:
                return {
                    'action': 'SELL',
                    'lot_size': 0.1,
                    'sl': current_price + (atr * atr_multiplier),
                    'tp': current_price - (atr * atr_multiplier * 2)
                }
        
        # Exit conditions
        elif self.open_positions:
            position_type = self.open_positions[0]['action']
            
            # Exit long if TK turns bearish or price drops below cloud
            if position_type == 'BUY' and (not tk_cross_bullish or below_cloud):
                return {'action': 'CLOSE'}
            
            # Exit short if TK turns bullish or price rises above cloud
            elif position_type == 'SELL' and (not tk_cross_bearish or above_cloud):
                return {'action': 'CLOSE'}
        
        return None
    
    def get_parameters(self) -> Dict:
        return self.parameters

class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping strategy using multiple short-term indicators
    """
    
    def __init__(self, symbol: str, timeframe: str, initial_balance: float = 10000):
        super().__init__(symbol, timeframe, initial_balance)
        self.parameters = {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 7,
            'stoch_k': 5,
            'stoch_d': 3,
            'risk_percent': 1  # Risk 1% per trade
        }
    
    def on_tick(self, data: pd.DataFrame) -> Optional[Dict]:
        """Fast scalping signals"""
        if len(data) < 30:
            return None
        
        # Add short-term indicators
        data = self.add_indicator(data, 'ema', period=self.parameters['ema_fast'])
        data = self.add_indicator(data, 'ema', period=self.parameters['ema_slow'])
        data = self.add_indicator(data, 'rsi', period=self.parameters['rsi_period'])
        data = self.add_indicator(data, 'stochastic', k_period=self.parameters['stoch_k'], d_period=self.parameters['stoch_d'])
        data = self.add_indicator(data, 'atr', period=14)
        
        current_price = data['Close'].iloc[-1]
        ema_fast = data[f'ema_{self.parameters["ema_fast"]}'].iloc[-1]
        ema_slow = data[f'ema_{self.parameters["ema_slow"]}'].iloc[-1]
        rsi = data[f'rsi_{self.parameters["rsi_period"]}'].iloc[-1]
        stoch_k = data[f'stoch_k_{self.parameters["stoch_k"]}'].iloc[-1]
        atr = data['atr_14'].iloc[-1]
        
        if pd.isna(ema_fast) or pd.isna(rsi) or pd.isna(atr):
            return None
        
        # Calculate position size based on risk
        risk_amount = self.balance * (self.parameters['risk_percent'] / 100)
        stop_distance = atr * 1.5
        lot_size = min(0.5, risk_amount / (stop_distance * 100000))  # Max 0.5 lots
        
        # Quick entry conditions
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow
        
        if not self.open_positions:
            # Bullish scalp: EMA bullish + RSI oversold + Stoch turning up
            if ema_bullish and rsi < 40 and stoch_k > 20:
                return {
                    'action': 'BUY',
                    'lot_size': lot_size,
                    'sl': current_price - stop_distance,
                    'tp': current_price + (stop_distance * 2)  # 2:1 R/R
                }
            
            # Bearish scalp: EMA bearish + RSI overbought + Stoch turning down
            elif ema_bearish and rsi > 60 and stoch_k < 80:
                return {
                    'action': 'SELL',
                    'lot_size': lot_size,
                    'sl': current_price + stop_distance,
                    'tp': current_price - (stop_distance * 2)  # 2:1 R/R
                }
        
        # Quick exits for scalping
        elif self.open_positions:
            position_type = self.open_positions[0]['action']
            
            # Exit long on EMA cross or RSI overbought
            if position_type == 'BUY' and (not ema_bullish or rsi > 70):
                return {'action': 'CLOSE'}
            
            # Exit short on EMA cross or RSI oversold
            elif position_type == 'SELL' and (not ema_bearish or rsi < 30):
                return {'action': 'CLOSE'}
        
        return None
    
    def get_parameters(self) -> Dict:
        return self.parameters