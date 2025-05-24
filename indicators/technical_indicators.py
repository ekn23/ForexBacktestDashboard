"""
Comprehensive Technical Indicators Library
Includes all popular indicators from TradingView and more
"""
import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional

class TechnicalIndicators:
    """
    Complete collection of technical indicators for forex trading
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    @staticmethod
    def vwma(data: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Volume Weighted Moving Average"""
        return (data * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma) / (0.015 * mean_deviation)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return pd.Series(true_range).rolling(window=period).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), low.diff().abs(), 0)
        
        di_plus = 100 * pd.Series(dm_plus).rolling(window=period).mean() / tr.rolling(window=period).mean()
        di_minus = 100 * pd.Series(dm_minus).rolling(window=period).mean() / tr.rolling(window=period).mean()
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx, di_plus, di_minus
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        length = len(high)
        psar = np.zeros(length)
        af = af_start
        ep = 0
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        psar[0] = low.iloc[0]
        
        for i in range(1, length):
            if trend == 1:  # Uptrend
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_increment, af_max)
                if low.iloc[i] < psar[i]:
                    trend = -1
                    psar[i] = ep
                    ep = low.iloc[i]
                    af = af_start
            else:  # Downtrend
                psar[i] = psar[i-1] - af * (psar[i-1] - ep)
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_increment, af_max)
                if high.iloc[i] > psar[i]:
                    trend = 1
                    psar[i] = ep
                    ep = high.iloc[i]
                    af = af_start
        
        return pd.Series(psar, index=high.index)
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    @staticmethod
    def fibonacci_retracement(high_price: float, low_price: float) -> dict:
        """Fibonacci Retracement Levels"""
        difference = high_price - low_price
        levels = {
            '0%': high_price,
            '23.6%': high_price - 0.236 * difference,
            '38.2%': high_price - 0.382 * difference,
            '50%': high_price - 0.5 * difference,
            '61.8%': high_price - 0.618 * difference,
            '78.6%': high_price - 0.786 * difference,
            '100%': low_price
        }
        return levels
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> dict:
        """Pivot Points"""
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'PP': pivot,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = np.where(close > close.shift(), volume, 
               np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv).cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(period).sum()
        
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))
    
    @staticmethod
    def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator"""
        vm_plus = np.abs(high - low.shift())
        vm_minus = np.abs(low - high.shift())
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
        vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
        
        return vi_plus, vi_minus
    
    @staticmethod
    def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> Tuple[pd.Series, pd.Series]:
        """Elder Ray Index"""
        ema = close.ewm(span=period).mean()
        bull_power = high - ema
        bear_power = low - ema
        return bull_power, bear_power
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels"""
        middle_line = close.ewm(span=period).mean()
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return upper_channel, middle_line, lower_channel
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels"""
        upper_channel = high.rolling(period).max()
        lower_channel = low.rolling(period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return upper_channel, middle_channel, lower_channel
    
    @staticmethod
    def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
        """Awesome Oscillator"""
        median_price = (high + low) / 2
        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        return ao
    
    @staticmethod
    def accelerator_oscillator(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Accelerator Oscillator"""
        ao = TechnicalIndicators.awesome_oscillator(high, low)
        median_price = (high + low) / 2
        sma_5 = median_price.rolling(5).mean()
        
        ac = ao - ao.rolling(5).mean()
        return ac
    
    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Aroon Indicator"""
        aroon_up = ((period - high.rolling(period).apply(lambda x: period - 1 - np.argmax(x))) / period) * 100
        aroon_down = ((period - low.rolling(period).apply(lambda x: period - 1 - np.argmin(x))) / period) * 100
        
        return aroon_up, aroon_down
    
    @staticmethod
    def chande_momentum_oscillator(close: pd.Series, period: int = 14) -> pd.Series:
        """Chande Momentum Oscillator"""
        momentum = close.diff()
        
        positive_sum = momentum.where(momentum > 0, 0).rolling(period).sum()
        negative_sum = momentum.where(momentum < 0, 0).abs().rolling(period).sum()
        
        return 100 * (positive_sum - negative_sum) / (positive_sum + negative_sum)
    
    @staticmethod
    def detrended_price_oscillator(close: pd.Series, period: int = 14) -> pd.Series:
        """Detrended Price Oscillator"""
        sma = close.rolling(period).mean()
        shift_period = int(period / 2) + 1
        return close - sma.shift(shift_period)
    
    @staticmethod
    def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        """Force Index"""
        fi = (close - close.shift()) * volume
        return fi.ewm(span=period).mean()
    
    @staticmethod
    def mass_index(high: pd.Series, low: pd.Series, period1: int = 9, period2: int = 25) -> pd.Series:
        """Mass Index"""
        hl_range = high - low
        ema1 = hl_range.ewm(span=period1).mean()
        ema2 = ema1.ewm(span=period1).mean()
        mass_index = (ema1 / ema2).rolling(period2).sum()
        return mass_index
    
    @staticmethod
    def trix(close: pd.Series, period: int = 14) -> pd.Series:
        """TRIX"""
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        trix = (ema3 / ema3.shift()) - 1
        return trix * 10000  # Convert to basis points
    
    @staticmethod
    def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                          period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Ultimate Oscillator"""
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        bp = close - np.minimum(low, close.shift())
        
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return uo