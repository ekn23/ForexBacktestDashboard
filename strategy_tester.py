"""
Strategy Testing Engine for Forex Trading
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
from strategies.base_strategy import BaseStrategy
from strategies.example_strategies import MovingAverageCrossover, MACDStrategy

class StrategyTester:
    """
    Main engine for testing trading strategies
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.results = {}
    
    def load_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load market data for backtesting
        """
        filename = f"{symbol}_Candlestick_{timeframe}_BID_26.04.2023-26.04.2025.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        df = pd.read_csv(filepath)
        df['Local time'] = pd.to_datetime(df['Local time'], utc=True)
        df = df.sort_values('Local time')
        return df
    
    def run_backtest(self, strategy: BaseStrategy, symbol: str, timeframe: str, 
                     start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest on a strategy
        """
        # Load market data
        data = self.load_market_data(symbol, timeframe)
        
        # Filter by date range if provided
        if start_date:
            start_date_parsed = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
            # Handle timezone compatibility
            try:
                if hasattr(data['Local time'].dtype, 'tz') and data['Local time'].dt.tz is not None:
                    if start_date_parsed.tz is None:
                        start_date_parsed = start_date_parsed.tz_localize('UTC')
            except:
                pass  # If timezone handling fails, proceed without it
            data = data[data['Local time'] >= start_date_parsed]
        if end_date:
            end_date_parsed = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
            # Handle timezone compatibility
            try:
                if hasattr(data['Local time'].dtype, 'tz') and data['Local time'].dt.tz is not None:
                    if end_date_parsed.tz is None:
                        end_date_parsed = end_date_parsed.tz_localize('UTC')
            except:
                pass  # If timezone handling fails, proceed without it
            data = data[data['Local time'] <= end_date_parsed]
        
        if len(data) < 50:
            return {"error": "Not enough data for backtesting"}
        
        # Initialize strategy
        strategy.symbol = symbol
        strategy.timeframe = timeframe
        
        # Run simulation
        for i in range(50, len(data)):  # Start after 50 bars for indicators
            current_data = data.iloc[:i+1]
            current_row = data.iloc[i]
            
            # Generate signal
            signal = strategy.on_tick(current_data)
            
            if signal:
                strategy.execute_trade(signal, current_row['Close'], str(current_row['Local time']))
        
        # Close any remaining open positions
        if strategy.open_positions:
            last_price = data.iloc[-1]['Close']
            last_time = str(data.iloc[-1]['Local time'])
            strategy.execute_trade({'action': 'CLOSE'}, last_price, last_time)
        
        # Return results
        return {
            'strategy_name': strategy.__class__.__name__,
            'symbol': symbol,
            'timeframe': timeframe,
            'parameters': strategy.get_parameters(),
            'performance': strategy.get_performance_metrics(),
            'trades': strategy.trades
        }
    
    def optimize_strategy(self, strategy_class, symbol: str, timeframe: str, 
                         param_ranges: Dict[str, List], max_iterations: int = 50) -> List[Dict]:
        """
        Optimize strategy parameters
        """
        results = []
        iterations = 0
        
        # Generate parameter combinations (simplified grid search)
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations[:max_iterations]:
            try:
                # Create strategy instance with parameters
                strategy = strategy_class(symbol, timeframe, **params)
                
                # Run backtest
                result = self.run_backtest(strategy, symbol, timeframe)
                
                if 'error' not in result:
                    result['optimization_params'] = params
                    results.append(result)
                    
                iterations += 1
                
            except Exception as e:
                continue
        
        # Sort by profit factor (or other metric)
        results.sort(key=lambda x: x['performance'].get('profit_factor', 0), reverse=True)
        return results
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """
        Generate all parameter combinations for optimization
        """
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def compare_strategies(self, symbol: str, timeframe: str) -> Dict:
        """
        Compare performance of different strategies
        """
        strategies = [
            MovingAverageCrossover(symbol, timeframe),
            RSIStrategy(symbol, timeframe),
            MACDStrategy(symbol, timeframe)
        ]
        
        results = []
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, symbol, timeframe)
                if 'error' not in result:
                    results.append(result)
            except Exception as e:
                continue
        
        return {
            'comparison_results': results,
            'best_strategy': max(results, key=lambda x: x['performance'].get('profit_factor', 0)) if results else None
        }

# Example usage functions
def test_moving_average_strategy(symbol: str = "EURUSD", timeframe: str = "30_M"):
    """Test Moving Average Crossover Strategy"""
    tester = StrategyTester()
    strategy = MovingAverageCrossover(symbol, timeframe, fast_period=10, slow_period=20)
    return tester.run_backtest(strategy, symbol, timeframe)



def optimize_ma_strategy(symbol: str = "EURUSD", timeframe: str = "30_M"):
    """Optimize Moving Average Strategy"""
    tester = StrategyTester()
    param_ranges = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 50]
    }
    return tester.optimize_strategy(MovingAverageCrossover, symbol, timeframe, param_ranges)