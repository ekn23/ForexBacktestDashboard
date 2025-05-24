"""
Advanced Pattern Recognition and Market Structure Analysis
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

class PatternRecognition:
    """
    Advanced pattern recognition for forex trading
    """
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20, min_touches: int = 2) -> Dict:
        """Detect Support and Resistance Levels"""
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        # Find pivot highs and lows
        pivot_highs = data['High'][data['High'] == highs]
        pivot_lows = data['Low'][data['Low'] == lows]
        
        # Group similar levels
        resistance_levels = []
        support_levels = []
        
        tolerance = data['Close'].std() * 0.1  # Dynamic tolerance
        
        for price in pivot_highs:
            if not np.isnan(price):
                touches = sum(abs(h - price) <= tolerance for h in pivot_highs if not np.isnan(h))
                if touches >= min_touches:
                    resistance_levels.append(price)
        
        for price in pivot_lows:
            if not np.isnan(price):
                touches = sum(abs(l - price) <= tolerance for l in pivot_lows if not np.isnan(l))
                if touches >= min_touches:
                    support_levels.append(price)
        
        return {
            'resistance': list(set(resistance_levels)),
            'support': list(set(support_levels))
        }
    
    @staticmethod
    def detect_trend_lines(data: pd.DataFrame, window: int = 50) -> Dict:
        """Detect Trend Lines"""
        highs = data['High'].rolling(window=window//2, center=True).max()
        lows = data['Low'].rolling(window=window//2, center=True).min()
        
        # Find significant highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(len(data)):
            if data['High'].iloc[i] == highs.iloc[i]:
                pivot_highs.append((i, data['High'].iloc[i]))
            if data['Low'].iloc[i] == lows.iloc[i]:
                pivot_lows.append((i, data['Low'].iloc[i]))
        
        # Calculate trend lines (simplified)
        uptrend_line = None
        downtrend_line = None
        
        if len(pivot_lows) >= 2:
            # Calculate uptrend line slope
            x1, y1 = pivot_lows[-2]
            x2, y2 = pivot_lows[-1]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            uptrend_line = {'slope': slope, 'intercept': y2 - slope * x2}
        
        if len(pivot_highs) >= 2:
            # Calculate downtrend line slope
            x1, y1 = pivot_highs[-2]
            x2, y2 = pivot_highs[-1]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            downtrend_line = {'slope': slope, 'intercept': y2 - slope * x2}
        
        return {
            'uptrend': uptrend_line,
            'downtrend': downtrend_line,
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows
        }
    
    @staticmethod
    def detect_candlestick_patterns(data: pd.DataFrame) -> pd.DataFrame:
        """Detect Candlestick Patterns"""
        patterns = pd.DataFrame(index=data.index)
        
        # Calculate candle properties
        body = abs(data['Close'] - data['Open'])
        upper_shadow = data['High'] - np.maximum(data['Close'], data['Open'])
        lower_shadow = np.minimum(data['Close'], data['Open']) - data['Low']
        
        # Doji
        patterns['doji'] = body <= (data['High'] - data['Low']) * 0.1
        
        # Hammer/Hanging Man
        hammer_condition = (lower_shadow >= 2 * body) & (upper_shadow <= body * 0.5)
        patterns['hammer'] = hammer_condition & (data['Close'] > data['Open'])
        patterns['hanging_man'] = hammer_condition & (data['Close'] < data['Open'])
        
        # Shooting Star/Inverted Hammer
        shooting_condition = (upper_shadow >= 2 * body) & (lower_shadow <= body * 0.5)
        patterns['shooting_star'] = shooting_condition & (data['Close'] < data['Open'])
        patterns['inverted_hammer'] = shooting_condition & (data['Close'] > data['Open'])
        
        # Engulfing Patterns
        prev_body = body.shift(1)
        prev_bullish = data['Close'].shift(1) > data['Open'].shift(1)
        prev_bearish = data['Close'].shift(1) < data['Open'].shift(1)
        
        patterns['bullish_engulfing'] = (
            prev_bearish & 
            (data['Close'] > data['Open']) & 
            (data['Close'] > data['Open'].shift(1)) & 
            (data['Open'] < data['Close'].shift(1))
        )
        
        patterns['bearish_engulfing'] = (
            prev_bullish & 
            (data['Close'] < data['Open']) & 
            (data['Close'] < data['Open'].shift(1)) & 
            (data['Open'] > data['Close'].shift(1))
        )
        
        # Morning/Evening Star
        middle_doji = patterns['doji'].shift(1)
        patterns['morning_star'] = (
            prev_bearish & 
            middle_doji & 
            (data['Close'] > data['Open']) & 
            (data['Close'] > (data['Open'].shift(1) + data['Close'].shift(1)) / 2)
        )
        
        patterns['evening_star'] = (
            prev_bullish & 
            middle_doji & 
            (data['Close'] < data['Open']) & 
            (data['Close'] < (data['Open'].shift(1) + data['Close'].shift(1)) / 2)
        )
        
        return patterns
    
    @staticmethod
    def detect_chart_patterns(data: pd.DataFrame, window: int = 50) -> Dict:
        """Detect Chart Patterns (Head & Shoulders, Triangles, etc.)"""
        patterns = {}
        
        # Find significant highs and lows
        highs = data['High'].rolling(window=window//5, center=True).max()
        lows = data['Low'].rolling(window=window//5, center=True).min()
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(len(data)):
            if data['High'].iloc[i] == highs.iloc[i]:
                pivot_highs.append((i, data['High'].iloc[i]))
            if data['Low'].iloc[i] == lows.iloc[i]:
                pivot_lows.append((i, data['Low'].iloc[i]))
        
        # Head and Shoulders Pattern
        if len(pivot_highs) >= 3:
            last_three_highs = pivot_highs[-3:]
            left_shoulder, head, right_shoulder = last_three_highs
            
            # Check if it forms a head and shoulders pattern
            if (head[1] > left_shoulder[1] and head[1] > right_shoulder[1] and
                abs(left_shoulder[1] - right_shoulder[1]) < data['Close'].std() * 0.5):
                patterns['head_and_shoulders'] = {
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': (left_shoulder[1] + right_shoulder[1]) / 2
                }
        
        # Double Top/Bottom
        if len(pivot_highs) >= 2:
            last_two_highs = pivot_highs[-2:]
            if abs(last_two_highs[0][1] - last_two_highs[1][1]) < data['Close'].std() * 0.3:
                patterns['double_top'] = last_two_highs
        
        if len(pivot_lows) >= 2:
            last_two_lows = pivot_lows[-2:]
            if abs(last_two_lows[0][1] - last_two_lows[1][1]) < data['Close'].std() * 0.3:
                patterns['double_bottom'] = last_two_lows
        
        # Triangle Patterns (simplified detection)
        if len(pivot_highs) >= 3 and len(pivot_lows) >= 3:
            recent_highs = [h[1] for h in pivot_highs[-3:]]
            recent_lows = [l[1] for l in pivot_lows[-3:]]
            
            # Ascending Triangle
            if (recent_highs[0] <= recent_highs[1] <= recent_highs[2] and
                recent_lows[0] < recent_lows[1] < recent_lows[2]):
                patterns['ascending_triangle'] = {
                    'highs': pivot_highs[-3:],
                    'lows': pivot_lows[-3:]
                }
            
            # Descending Triangle
            elif (recent_highs[0] > recent_highs[1] > recent_highs[2] and
                  recent_lows[0] >= recent_lows[1] >= recent_lows[2]):
                patterns['descending_triangle'] = {
                    'highs': pivot_highs[-3:],
                    'lows': pivot_lows[-3:]
                }
        
        return patterns
    
    @staticmethod
    def market_structure_analysis(data: pd.DataFrame) -> Dict:
        """Advanced Market Structure Analysis"""
        # Calculate swing highs and lows
        swing_window = 10
        
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_window, len(data) - swing_window):
            window_data = data[i-swing_window:i+swing_window+1]
            
            if data['High'].iloc[i] == window_data['High'].max():
                swing_highs.append((i, data['High'].iloc[i]))
            
            if data['Low'].iloc[i] == window_data['Low'].min():
                swing_lows.append((i, data['Low'].iloc[i]))
        
        # Determine market structure
        structure = "sideways"
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            recent_highs = [h[1] for h in swing_highs[-2:]]
            recent_lows = [l[1] for l in swing_lows[-2:]]
            
            # Higher highs and higher lows = uptrend
            if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
                structure = "uptrend"
            # Lower highs and lower lows = downtrend
            elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
                structure = "downtrend"
        
        # Calculate order blocks (simplified)
        order_blocks = []
        volume_threshold = data['Volume'].quantile(0.8) if 'Volume' in data.columns else None
        
        if volume_threshold:
            high_volume_candles = data[data['Volume'] > volume_threshold]
            for idx, candle in high_volume_candles.iterrows():
                order_blocks.append({
                    'timestamp': idx,
                    'high': candle['High'],
                    'low': candle['Low'],
                    'type': 'bullish' if candle['Close'] > candle['Open'] else 'bearish'
                })
        
        return {
            'market_structure': structure,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'order_blocks': order_blocks
        }
    
    @staticmethod
    def fibonacci_extensions(high_price: float, low_price: float, retrace_level: float) -> Dict:
        """Calculate Fibonacci Extensions"""
        difference = high_price - low_price
        
        extensions = {
            '127.2%': retrace_level + (difference * 1.272),
            '161.8%': retrace_level + (difference * 1.618),
            '200%': retrace_level + (difference * 2.0),
            '261.8%': retrace_level + (difference * 2.618),
            '314.6%': retrace_level + (difference * 3.146),
            '423.6%': retrace_level + (difference * 4.236)
        }
        
        return extensions
    
    @staticmethod
    def volume_profile(data: pd.DataFrame, bins: int = 50) -> Dict:
        """Calculate Volume Profile"""
        if 'Volume' not in data.columns:
            return {}
        
        price_range = data['High'].max() - data['Low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for i in range(bins):
            price_level = data['Low'].min() + (i * bin_size)
            volume_at_level = 0
            
            for _, row in data.iterrows():
                if price_level >= row['Low'] and price_level <= row['High']:
                    volume_at_level += row['Volume']
            
            volume_profile[round(price_level, 5)] = volume_at_level
        
        # Find Point of Control (POC) - highest volume level
        poc = max(volume_profile, key=volume_profile.get)
        
        # Find Value Area (70% of volume)
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(volume_profile.values())
        value_area_volume = 0
        value_area_levels = []
        
        for price, volume in sorted_levels:
            value_area_levels.append(price)
            value_area_volume += volume
            if value_area_volume >= total_volume * 0.7:
                break
        
        return {
            'volume_profile': volume_profile,
            'poc': poc,
            'value_area_high': max(value_area_levels),
            'value_area_low': min(value_area_levels)
        }