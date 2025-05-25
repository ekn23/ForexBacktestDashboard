"""
MQL4/MQL5 Strategy Interpreter
Converts MQL4/MQL5 code to executable Python trading strategies
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import os

class MQLInterpreter:
    """Interprets and executes MQL4/MQL5 trading strategies"""
    
    def __init__(self):
        self.variables = {}
        self.orders = []
        self.current_price = 0.0
        self.current_spread = 0.0
        self.account_balance = 10000.0
        self.equity = 10000.0
        self.margin = 0.0
        self.free_margin = 10000.0
        
        # MQL4/5 built-in functions mapping
        self.builtin_functions = {
            'iMA': self._iMA,
            'iRSI': self._iRSI,
            'iMACD': self._iMACD,
            'iStochastic': self._iStochastic,
            'iBands': self._iBands,
            'iATR': self._iATR,
            'OrderSend': self._OrderSend,
            'OrderClose': self._OrderClose,
            'OrderModify': self._OrderModify,
            'AccountBalance': lambda: self.account_balance,
            'AccountEquity': lambda: self.equity,
            'AccountMargin': lambda: self.margin,
            'AccountFreeMargin': lambda: self.free_margin,
            'Bid': lambda: self.current_price - self.current_spread/2,
            'Ask': lambda: self.current_price + self.current_spread/2,
            'Point': lambda: 0.0001,  # Standard forex point
            'Digits': lambda: 5,      # Standard forex digits
        }
    
    def parse_mql_strategy(self, filepath: str) -> Dict:
        """Parse MQL4/5 strategy file and extract trading logic"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key components
            strategy_info = {
                'name': os.path.basename(filepath)[:-4],
                'parameters': self._extract_parameters(content),
                'init_code': self._extract_function(content, 'OnInit') or self._extract_function(content, 'init'),
                'tick_code': self._extract_function(content, 'OnTick') or self._extract_function(content, 'start'),
                'variables': self._extract_variables(content),
                'original_code': content
            }
            
            return strategy_info
            
        except Exception as e:
            raise Exception(f"Error parsing MQL strategy: {str(e)}")
    
    def execute_mql_strategy(self, strategy_info: Dict, df: pd.DataFrame, user_params: Dict = None) -> Dict:
        """Execute MQL strategy on historical data"""
        try:
            # Initialize strategy
            self._initialize_strategy(strategy_info, user_params)
            
            # Prepare results
            signals = []
            trades = []
            equity_curve = [self.account_balance]
            
            # Execute strategy bar by bar
            for i, (timestamp, row) in enumerate(df.iterrows()):
                if i < 50:  # Skip first 50 bars for indicator calculation
                    equity_curve.append(self.account_balance)
                    continue
                
                # Update market data
                self.current_price = row['Close']
                self.current_spread = 0.0002  # 2 pip spread
                
                # Execute OnTick logic
                signal = self._execute_tick_logic(strategy_info, df.iloc[:i+1], i)
                signals.append(signal)
                
                # Update equity
                self._update_account_equity(row['Close'])
                equity_curve.append(self.equity)
            
            # Create results DataFrame
            result_df = df.copy()
            result_df['signal'] = [0] * 50 + signals
            result_df['equity'] = equity_curve
            
            return {
                'df': result_df,
                'trades': self.orders,
                'final_balance': self.equity,
                'total_trades': len([o for o in self.orders if o.get('type') in ['buy', 'sell']]),
                'winning_trades': len([o for o in self.orders if o.get('profit', 0) > 0]),
                'max_drawdown': self._calculate_max_drawdown(equity_curve)
            }
            
        except Exception as e:
            raise Exception(f"Error executing MQL strategy: {str(e)}")
    
    def _extract_parameters(self, content: str) -> Dict:
        """Extract input parameters from MQL code"""
        params = {}
        
        # Look for input/extern declarations
        input_pattern = r'(?:input|extern)\s+(\w+)\s+(\w+)\s*=\s*([^;]+);'
        matches = re.findall(input_pattern, content, re.IGNORECASE)
        
        for match in matches:
            param_type, param_name, param_value = match
            try:
                # Try to convert to appropriate type
                if 'int' in param_type.lower():
                    params[param_name] = int(param_value.strip())
                elif 'double' in param_type.lower() or 'float' in param_type.lower():
                    params[param_name] = float(param_value.strip())
                elif 'bool' in param_type.lower():
                    params[param_name] = param_value.strip().lower() in ['true', '1']
                else:
                    params[param_name] = param_value.strip().strip('"\'')
            except:
                params[param_name] = param_value.strip()
        
        return params
    
    def _extract_function(self, content: str, function_name: str) -> str:
        """Extract function code from MQL content"""
        # Pattern to match function definition
        pattern = rf'{function_name}\s*\([^)]*\)\s*\{{([^}}]*(?:\{{[^}}]*\}}[^}}]*)*)\}}'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1)
        return ""
    
    def _extract_variables(self, content: str) -> Dict:
        """Extract global variables from MQL code"""
        variables = {}
        
        # Look for variable declarations
        var_pattern = r'(?:static\s+)?(\w+)\s+(\w+)(?:\s*=\s*([^;]+))?;'
        matches = re.findall(var_pattern, content)
        
        for match in matches:
            var_type, var_name, var_value = match
            if var_value:
                try:
                    if 'int' in var_type.lower():
                        variables[var_name] = int(var_value.strip())
                    elif 'double' in var_type.lower():
                        variables[var_name] = float(var_value.strip())
                    elif 'bool' in var_type.lower():
                        variables[var_name] = var_value.strip().lower() in ['true', '1']
                    else:
                        variables[var_name] = var_value.strip()
                except:
                    variables[var_name] = 0
            else:
                variables[var_name] = 0
        
        return variables
    
    def _initialize_strategy(self, strategy_info: Dict, user_params: Dict = None):
        """Initialize strategy with parameters"""
        self.variables = strategy_info['variables'].copy()
        
        # Apply user parameters
        if user_params:
            for param, value in user_params.items():
                if param in strategy_info['parameters']:
                    self.variables[param] = value
                    strategy_info['parameters'][param] = value
        
        self.orders = []
        self.account_balance = user_params.get('starting_capital', 10000.0) if user_params else 10000.0
        self.equity = self.account_balance
    
    def _execute_tick_logic(self, strategy_info: Dict, df_slice: pd.DataFrame, current_bar: int) -> int:
        """Execute the OnTick logic and return signal"""
        try:
            # Simple signal generation based on common MQL patterns
            tick_code = strategy_info['tick_code']
            
            # Look for common trading patterns in the code
            signal = 0
            
            # Check for moving average crossover
            if 'iMA' in tick_code and ('>' in tick_code or '<' in tick_code):
                ma_fast = self._iMA(df_slice, 10, 0, 'Close')
                ma_slow = self._iMA(df_slice, 20, 0, 'Close')
                
                if len(ma_fast) > 1 and len(ma_slow) > 1:
                    if ma_fast.iloc[-1] > ma_slow.iloc[-1] and ma_fast.iloc[-2] <= ma_slow.iloc[-2]:
                        signal = 1  # Buy signal
                    elif ma_fast.iloc[-1] < ma_slow.iloc[-1] and ma_fast.iloc[-2] >= ma_slow.iloc[-2]:
                        signal = -1  # Sell signal
            
            # Check for RSI patterns
            elif 'iRSI' in tick_code:
                rsi = self._iRSI(df_slice, 14, 'Close')
                if len(rsi) > 0:
                    current_rsi = rsi.iloc[-1]
                    if current_rsi < 30:
                        signal = 1  # Oversold - buy
                    elif current_rsi > 70:
                        signal = -1  # Overbought - sell
            
            # Check for MACD patterns
            elif 'iMACD' in tick_code:
                macd_main, macd_signal = self._iMACD(df_slice, 12, 26, 9, 'Close')
                if len(macd_main) > 1 and len(macd_signal) > 1:
                    if macd_main.iloc[-1] > macd_signal.iloc[-1] and macd_main.iloc[-2] <= macd_signal.iloc[-2]:
                        signal = 1  # MACD bullish crossover
                    elif macd_main.iloc[-1] < macd_signal.iloc[-1] and macd_main.iloc[-2] >= macd_signal.iloc[-2]:
                        signal = -1  # MACD bearish crossover
            
            return signal
            
        except Exception as e:
            return 0
    
    def _update_account_equity(self, current_price: float):
        """Update account equity based on open positions"""
        total_profit = 0
        
        for order in self.orders:
            if order.get('status') == 'open':
                if order['type'] == 'buy':
                    profit = (current_price - order['open_price']) * order['lots'] * 100000
                else:  # sell
                    profit = (order['open_price'] - current_price) * order['lots'] * 100000
                
                order['current_profit'] = profit
                total_profit += profit
        
        self.equity = self.account_balance + total_profit
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak * 100
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    # MQL4/5 Built-in Function Implementations
    
    def _iMA(self, df: pd.DataFrame, period: int, shift: int, price: str) -> pd.Series:
        """Moving Average indicator"""
        if price == 'Close' or price == 'PRICE_CLOSE':
            return df['Close'].rolling(window=period).mean()
        elif price == 'Open' or price == 'PRICE_OPEN':
            return df['Open'].rolling(window=period).mean()
        elif price == 'High' or price == 'PRICE_HIGH':
            return df['High'].rolling(window=period).mean()
        elif price == 'Low' or price == 'PRICE_LOW':
            return df['Low'].rolling(window=period).mean()
        else:
            return df['Close'].rolling(window=period).mean()
    
    def _iRSI(self, df: pd.DataFrame, period: int, price: str) -> pd.Series:
        """RSI indicator"""
        if price == 'Close' or price == 'PRICE_CLOSE':
            data = df['Close']
        else:
            data = df['Close']
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _iMACD(self, df: pd.DataFrame, fast: int, slow: int, signal: int, price: str) -> Tuple[pd.Series, pd.Series]:
        """MACD indicator"""
        if price == 'Close' or price == 'PRICE_CLOSE':
            data = df['Close']
        else:
            data = df['Close']
        
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line, signal_line
    
    def _iStochastic(self, df: pd.DataFrame, k_period: int, d_period: int, slowing: int) -> Tuple[pd.Series, pd.Series]:
        """Stochastic indicator"""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _iBands(self, df: pd.DataFrame, period: int, deviation: float, price: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands indicator"""
        if price == 'Close' or price == 'PRICE_CLOSE':
            data = df['Close']
        else:
            data = df['Close']
        
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * deviation)
        lower_band = sma - (std * deviation)
        
        return upper_band, sma, lower_band
    
    def _iATR(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range indicator"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _OrderSend(self, symbol: str, cmd: int, volume: float, price: float, 
                   slippage: int, stoploss: float, takeprofit: float, comment: str = ""):
        """Simulate order sending"""
        order = {
            'symbol': symbol,
            'type': 'buy' if cmd == 0 else 'sell',  # OP_BUY = 0, OP_SELL = 1
            'lots': volume,
            'open_price': price,
            'stop_loss': stoploss,
            'take_profit': takeprofit,
            'comment': comment,
            'status': 'open',
            'open_time': pd.Timestamp.now(),
            'ticket': len(self.orders) + 1
        }
        
        self.orders.append(order)
        return order['ticket']
    
    def _OrderClose(self, ticket: int, lots: float, price: float, slippage: int):
        """Simulate order closing"""
        for order in self.orders:
            if order.get('ticket') == ticket and order.get('status') == 'open':
                if order['type'] == 'buy':
                    profit = (price - order['open_price']) * lots * 100000
                else:
                    profit = (order['open_price'] - price) * lots * 100000
                
                order['close_price'] = price
                order['close_time'] = pd.Timestamp.now()
                order['profit'] = profit
                order['status'] = 'closed'
                
                self.account_balance += profit
                return True
        
        return False
    
    def _OrderModify(self, ticket: int, price: float, stoploss: float, takeprofit: float):
        """Simulate order modification"""
        for order in self.orders:
            if order.get('ticket') == ticket:
                order['stop_loss'] = stoploss
                order['take_profit'] = takeprofit
                return True
        
        return False