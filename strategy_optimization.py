"""
Strategy Optimization & Robustness Testing
Parameter sweeps, walk-forward analysis, and out-of-sample validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import itertools
from strategy_tester import StrategyTester
from strategies.example_strategies import RSIStrategy, MovingAverageCrossStrategy
from trading_config import TradingConfig
import matplotlib.pyplot as plt
import seaborn as sns

class StrategyOptimizer:
    """
    Advanced strategy optimization with parameter sweeps and walk-forward testing
    """
    
    def __init__(self, data_dir: str = "data"):
        self.tester = StrategyTester(data_dir)
        self.optimization_results = []
        
    def parameter_sweep(self, strategy_class, symbol: str, timeframe: str, 
                       param_ranges: Dict[str, List], 
                       optimization_period: Tuple[str, str],
                       validation_period: Tuple[str, str]) -> Dict:
        """
        Comprehensive parameter sweep with in-sample and out-of-sample testing
        
        Args:
            strategy_class: Strategy class to optimize
            symbol: Currency pair (e.g., 'EURUSD')
            timeframe: '5_M' or '30_M'
            param_ranges: Dict of parameter names and their test ranges
            optimization_period: (start_date, end_date) for parameter optimization
            validation_period: (start_date, end_date) for out-of-sample validation
        """
        print(f"🔍 Parameter Sweep: {symbol} {timeframe}")
        print(f"📊 Optimization: {optimization_period[0]} to {optimization_period[1]}")
        print(f"✅ Validation: {validation_period[0]} to {validation_period[1]}")
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        print(f"🎯 Testing {len(param_combinations)} parameter combinations")
        
        results = {
            'optimization_results': [],
            'validation_results': [],
            'best_params': None,
            'robustness_score': 0
        }
        
        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(param_combinations)} combinations tested")
            
            # In-sample optimization
            opt_result = self._test_parameters(
                strategy_class, symbol, timeframe, params, 
                optimization_period[0], optimization_period[1]
            )
            
            if opt_result and opt_result.get('total_return', 0) > 0:
                # Out-of-sample validation
                val_result = self._test_parameters(
                    strategy_class, symbol, timeframe, params,
                    validation_period[0], validation_period[1]
                )
                
                if val_result:
                    combined_result = {
                        'parameters': params,
                        'optimization': opt_result,
                        'validation': val_result,
                        'robustness_score': self._calculate_robustness_score(opt_result, val_result)
                    }
                    
                    results['optimization_results'].append(opt_result)
                    results['validation_results'].append(val_result)
                    self.optimization_results.append(combined_result)
        
        # Find best parameters based on robustness
        if self.optimization_results:
            best_result = max(self.optimization_results, key=lambda x: x['robustness_score'])
            results['best_params'] = best_result['parameters']
            results['robustness_score'] = best_result['robustness_score']
            
            print(f"\n🏆 Best Parameters Found:")
            for param, value in best_result['parameters'].items():
                print(f"   {param}: {value}")
            print(f"🎯 Robustness Score: {best_result['robustness_score']:.3f}")
        
        return results
    
    def walk_forward_analysis(self, strategy_class, symbol: str, timeframe: str,
                            best_params: Dict, start_date: str, end_date: str,
                            optimization_window: int = 60, validation_window: int = 30) -> Dict:
        """
        Walk-forward analysis to test strategy robustness over time
        
        Args:
            optimization_window: Days for optimization window
            validation_window: Days for validation window
        """
        print(f"\n🚶 Walk-Forward Analysis: {symbol} {timeframe}")
        print(f"📊 Optimization Window: {optimization_window} days")
        print(f"✅ Validation Window: {validation_window} days")
        
        # Load full dataset
        data = self.tester.load_market_data(symbol, timeframe)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        walk_forward_results = []
        current_start = start_dt
        
        while current_start + timedelta(days=optimization_window + validation_window) <= end_dt:
            opt_end = current_start + timedelta(days=optimization_window)
            val_start = opt_end
            val_end = val_start + timedelta(days=validation_window)
            
            print(f"Period: {current_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
            
            # Test with best parameters on this validation period
            val_result = self._test_parameters(
                strategy_class, symbol, timeframe, best_params,
                val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d')
            )
            
            if val_result:
                walk_forward_results.append({
                    'period_start': val_start,
                    'period_end': val_end,
                    'return': val_result.get('total_return', 0),
                    'sharpe': val_result.get('sharpe_ratio', 0),
                    'max_drawdown': val_result.get('max_drawdown', 0),
                    'win_rate': val_result.get('win_rate', 0)
                })
            
            current_start += timedelta(days=validation_window)
        
        # Calculate walk-forward statistics
        if walk_forward_results:
            returns = [r['return'] for r in walk_forward_results]
            sharpes = [r['sharpe'] for r in walk_forward_results if r['sharpe'] is not None]
            
            wf_stats = {
                'periods_tested': len(walk_forward_results),
                'avg_return': np.mean(returns),
                'return_std': np.std(returns),
                'avg_sharpe': np.mean(sharpes) if sharpes else 0,
                'positive_periods': sum(1 for r in returns if r > 0),
                'consistency_score': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
                'results': walk_forward_results
            }
            
            print(f"\n📈 Walk-Forward Results:")
            print(f"   Periods Tested: {wf_stats['periods_tested']}")
            print(f"   Average Return: {wf_stats['avg_return']:.2f}%")
            print(f"   Return Volatility: {wf_stats['return_std']:.2f}%")
            print(f"   Positive Periods: {wf_stats['positive_periods']}/{wf_stats['periods_tested']}")
            print(f"   Consistency Score: {wf_stats['consistency_score']:.2f}")
            
            return wf_stats
        
        return {'error': 'No valid walk-forward periods found'}
    
    def robustness_report(self, symbol: str, timeframe: str) -> Dict:
        """
        Generate comprehensive robustness report for a strategy
        """
        if not self.optimization_results:
            return {'error': 'No optimization results available. Run parameter_sweep first.'}
        
        # Analyze parameter sensitivity
        param_sensitivity = self._analyze_parameter_sensitivity()
        
        # Performance consistency analysis
        consistency_analysis = self._analyze_performance_consistency()
        
        # Risk-adjusted metrics
        risk_analysis = self._analyze_risk_metrics()
        
        report = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_combinations_tested': len(self.optimization_results),
            'parameter_sensitivity': param_sensitivity,
            'performance_consistency': consistency_analysis,
            'risk_analysis': risk_analysis,
            'recommendations': self._generate_recommendations()
        }
        
        print(f"\n📋 Robustness Report for {symbol} {timeframe}")
        print(f"🔬 Combinations Tested: {report['total_combinations_tested']}")
        print(f"📊 Parameter Sensitivity: {param_sensitivity.get('most_sensitive', 'N/A')}")
        print(f"⚖️ Performance Consistency: {consistency_analysis.get('consistency_score', 0):.2f}")
        
        return report
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all possible parameter combinations"""
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _test_parameters(self, strategy_class, symbol: str, timeframe: str, 
                        params: Dict, start_date: str, end_date: str) -> Dict:
        """Test a specific parameter combination"""
        try:
            # Create strategy with parameters
            config = TradingConfig()
            strategy = strategy_class(symbol, timeframe, config)
            
            # Apply parameters
            for param, value in params.items():
                if hasattr(strategy, param):
                    setattr(strategy, param, value)
            
            # Run backtest
            result = self.tester.run_backtest(strategy, symbol, timeframe, start_date, end_date)
            
            if result and 'total_return' in result:
                # Add parameter info to result
                result['parameters'] = params
                return result
            
        except Exception as e:
            pass
        
        return None
    
    def _calculate_robustness_score(self, opt_result: Dict, val_result: Dict) -> float:
        """Calculate robustness score comparing optimization vs validation performance"""
        try:
            opt_return = opt_result.get('total_return', 0)
            val_return = val_result.get('total_return', 0)
            opt_sharpe = opt_result.get('sharpe_ratio', 0) or 0
            val_sharpe = val_result.get('sharpe_ratio', 0) or 0
            
            # Robustness = validation performance / optimization performance
            # Penalize large degradation in out-of-sample
            return_ratio = val_return / max(opt_return, 0.01)
            sharpe_ratio = val_sharpe / max(opt_sharpe, 0.01) if opt_sharpe > 0 else 0
            
            # Combined score (higher is better)
            robustness = (return_ratio + sharpe_ratio) / 2
            
            # Bonus for consistent positive performance
            if val_return > 0 and opt_return > 0:
                robustness *= 1.2
            
            return max(0, robustness)
        except:
            return 0
    
    def _analyze_parameter_sensitivity(self) -> Dict:
        """Analyze which parameters have the most impact on performance"""
        if not self.optimization_results:
            return {}
        
        param_impact = {}
        
        # Get all unique parameter names
        all_params = set()
        for result in self.optimization_results:
            all_params.update(result['parameters'].keys())
        
        for param in all_params:
            values = []
            scores = []
            
            for result in self.optimization_results:
                if param in result['parameters']:
                    values.append(result['parameters'][param])
                    scores.append(result['robustness_score'])
            
            if len(set(values)) > 1:  # Only analyze if parameter varies
                correlation = np.corrcoef(values, scores)[0, 1] if len(values) > 1 else 0
                param_impact[param] = abs(correlation) if not np.isnan(correlation) else 0
        
        most_sensitive = max(param_impact.items(), key=lambda x: x[1]) if param_impact else ('None', 0)
        
        return {
            'parameter_impacts': param_impact,
            'most_sensitive': most_sensitive[0],
            'sensitivity_score': most_sensitive[1]
        }
    
    def _analyze_performance_consistency(self) -> Dict:
        """Analyze consistency of performance across parameter combinations"""
        if not self.optimization_results:
            return {}
        
        robustness_scores = [r['robustness_score'] for r in self.optimization_results]
        validation_returns = [r['validation'].get('total_return', 0) for r in self.optimization_results]
        
        return {
            'consistency_score': 1 - (np.std(robustness_scores) / max(np.mean(robustness_scores), 0.01)),
            'avg_robustness': np.mean(robustness_scores),
            'robustness_std': np.std(robustness_scores),
            'positive_validation_rate': sum(1 for r in validation_returns if r > 0) / len(validation_returns)
        }
    
    def _analyze_risk_metrics(self) -> Dict:
        """Analyze risk-adjusted performance metrics"""
        if not self.optimization_results:
            return {}
        
        max_drawdowns = []
        sharpe_ratios = []
        
        for result in self.optimization_results:
            val_result = result['validation']
            max_drawdowns.append(val_result.get('max_drawdown', 0))
            if val_result.get('sharpe_ratio'):
                sharpe_ratios.append(val_result['sharpe_ratio'])
        
        return {
            'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            'best_sharpe': max(sharpe_ratios) if sharpe_ratios else 0,
            'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'risk_consistency': 1 - (np.std(max_drawdowns) / max(np.mean(max_drawdowns), 0.01)) if max_drawdowns else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate strategy recommendations based on optimization results"""
        recommendations = []
        
        if not self.optimization_results:
            return ["Run parameter optimization first to get recommendations"]
        
        # Analyze results
        best_result = max(self.optimization_results, key=lambda x: x['robustness_score'])
        avg_robustness = np.mean([r['robustness_score'] for r in self.optimization_results])
        
        if best_result['robustness_score'] > 1.0:
            recommendations.append("✅ Strategy shows good out-of-sample performance")
        else:
            recommendations.append("⚠️ Strategy may be overfitted - validation performance is weak")
        
        if avg_robustness > 0.5:
            recommendations.append("✅ Generally robust across parameter ranges")
        else:
            recommendations.append("⚠️ Strategy is sensitive to parameter choices")
        
        # Parameter-specific recommendations
        param_sensitivity = self._analyze_parameter_sensitivity()
        if param_sensitivity.get('most_sensitive'):
            recommendations.append(f"🎯 Focus on optimizing: {param_sensitivity['most_sensitive']}")
        
        return recommendations

# Example usage and testing functions
def optimize_rsi_strategy():
    """Example: Optimize RSI strategy parameters"""
    optimizer = StrategyOptimizer()
    
    # Define parameter ranges for RSI strategy
    param_ranges = {
        'rsi_period': [10, 12, 14, 16, 18, 20],
        'oversold': [20, 25, 30, 35],
        'overbought': [65, 70, 75, 80]
    }
    
    # Define time periods
    optimization_period = ('2024-01-01', '2024-06-30')
    validation_period = ('2024-07-01', '2024-12-31')
    
    # Run parameter sweep
    results = optimizer.parameter_sweep(
        RSIStrategy, 'EURUSD', '30_M',
        param_ranges, optimization_period, validation_period
    )
    
    return results

def run_comprehensive_analysis():
    """Run complete optimization and robustness analysis"""
    print("🚀 Starting Comprehensive Strategy Analysis")
    print("=" * 50)
    
    # Run RSI optimization
    rsi_results = optimize_rsi_strategy()
    
    if rsi_results.get('best_params'):
        optimizer = StrategyOptimizer()
        
        # Run walk-forward analysis
        wf_results = optimizer.walk_forward_analysis(
            RSIStrategy, 'EURUSD', '30_M',
            rsi_results['best_params'],
            '2024-01-01', '2024-12-31'
        )
        
        # Generate robustness report
        robustness_report = optimizer.robustness_report('EURUSD', '30_M')
        
        print("\n" + "=" * 50)
        print("🎯 ANALYSIS COMPLETE")
        print("=" * 50)
        
        return {
            'parameter_optimization': rsi_results,
            'walk_forward_analysis': wf_results,
            'robustness_report': robustness_report
        }

if __name__ == "__main__":
    # Run the comprehensive analysis
    results = run_comprehensive_analysis()