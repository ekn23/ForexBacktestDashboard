"""
Test Strategy with OANDA-Compliant Settings
Demonstrates realistic backtesting with professional parameters
"""
from trading_config import TradingConfig, OANDATradeExecutor
from strategies.example_strategies import MovingAverageCrossover, RSIStrategy, MACDStrategy
from strategies.multi_indicator_strategy import SuperTrendStrategy, IchimokuStrategy, ScalpingStrategy
from strategy_tester import StrategyTester
import pandas as pd

def create_custom_config(starting_capital=10000, leverage=50, max_risk_per_trade=2.0, slippage_pips=2.0):
    """Create a custom trading configuration"""
    config = TradingConfig(
        starting_capital=starting_capital,
        leverage=leverage,
        max_risk_per_trade=max_risk_per_trade,
        slippage_pips=slippage_pips,
        max_daily_loss=5.0,  # 5% daily loss limit
        max_drawdown=20.0    # 20% max drawdown
    )
    return config

def test_strategy_with_realistic_settings():
    """Test a strategy with OANDA-compliant settings"""
    
    # Your custom settings
    config = create_custom_config(
        starting_capital=10000,  # Start with $10K
        leverage=50,             # 50:1 leverage
        max_risk_per_trade=1.5,  # Risk 1.5% per trade
        slippage_pips=2.0        # 2 pip slippage
    )
    
    # Create strategy with realistic config
    strategy = SuperTrendStrategy("EURUSD", "30_M", config)
    
    # Run backtest
    tester = StrategyTester("data")
    result = tester.run_backtest(strategy, "EURUSD", "30_M")
    
    return result

def compare_realistic_vs_unrealistic():
    """Compare results with and without realistic trading costs"""
    
    # Unrealistic settings (perfect world)
    unrealistic_config = TradingConfig(
        starting_capital=10000,
        leverage=500,            # Unrealistic high leverage
        max_risk_per_trade=10.0, # Unrealistic high risk
        slippage_pips=0.0,       # No slippage
        spread_pips={'EURUSD': 0.0}  # No spread
    )
    
    # Realistic OANDA settings
    realistic_config = create_custom_config()
    
    # Test same strategy with both configs
    unrealistic_strategy = MovingAverageCrossover("EURUSD", "30_M", unrealistic_config)
    realistic_strategy = MovingAverageCrossover("EURUSD", "30_M", realistic_config)
    
    tester = StrategyTester("data")
    
    unrealistic_result = tester.run_backtest(unrealistic_strategy, "EURUSD", "30_M")
    realistic_result = tester.run_backtest(realistic_strategy, "EURUSD", "30_M")
    
    return {
        'unrealistic': unrealistic_result,
        'realistic': realistic_result
    }

def run_professional_strategy_suite():
    """Run multiple strategies with professional settings"""
    
    config = create_custom_config(
        starting_capital=25000,  # Higher starting capital
        leverage=30,             # Conservative leverage
        max_risk_per_trade=1.0,  # Conservative risk
        slippage_pips=1.5        # Tight slippage
    )
    
    strategies = [
        ('Moving Average', MovingAverageCrossover("EURUSD", "30_M", config)),
        ('RSI Strategy', RSIStrategy("EURUSD", "30_M", config)),
        ('MACD Strategy', MACDStrategy("EURUSD", "30_M", config)),
        ('SuperTrend', SuperTrendStrategy("EURUSD", "30_M", config)),
        ('Ichimoku', IchimokuStrategy("EURUSD", "30_M", config)),
        ('Scalping', ScalpingStrategy("EURUSD", "5_M", config))  # 5-min for scalping
    ]
    
    results = {}
    tester = StrategyTester("data")
    
    for name, strategy in strategies:
        try:
            timeframe = "5_M" if "Scalping" in name else "30_M"
            result = tester.run_backtest(strategy, "EURUSD", timeframe)
            results[name] = result
            print(f"✅ {name}: {result['performance']['total_pnl']:.2f} USD")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            
    return results

def analyze_trading_costs():
    """Analyze the impact of different trading costs"""
    
    cost_scenarios = {
        'Low Cost Broker': {
            'spread_pips': {'EURUSD': 0.5},
            'slippage_pips': 0.5
        },
        'Average Broker': {
            'spread_pips': {'EURUSD': 1.0},
            'slippage_pips': 1.0
        },
        'High Cost Broker': {
            'spread_pips': {'EURUSD': 2.0},
            'slippage_pips': 2.0
        },
        'OANDA Standard': {
            'spread_pips': {'EURUSD': 1.0},
            'slippage_pips': 2.0
        }
    }
    
    results = {}
    
    for scenario_name, costs in cost_scenarios.items():
        config = TradingConfig(
            starting_capital=10000,
            leverage=50,
            spread_pips=costs['spread_pips'],
            slippage_pips=costs['slippage_pips']
        )
        
        strategy = MovingAverageCrossover("EURUSD", "30_M", config)
        tester = StrategyTester("data")
        result = tester.run_backtest(strategy, "EURUSD", "30_M")
        
        results[scenario_name] = {
            'net_pnl': result['performance']['total_pnl'],
            'total_trades': result['performance']['total_trades'],
            'win_rate': result['performance']['win_rate']
        }
    
    return results

if __name__ == "__main__":
    print("🚀 Testing Strategy with Professional OANDA Settings")
    print("=" * 60)
    
    # Test single strategy
    print("\n📊 Testing SuperTrend Strategy...")
    result = test_strategy_with_realistic_settings()
    if result and 'performance' in result:
        perf = result['performance']
        print(f"Net P&L: ${perf['total_pnl']:.2f}")
        print(f"Win Rate: {perf['win_rate']:.1f}%")
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Profit Factor: {perf['profit_factor']:.2f}")
    
    # Compare realistic vs unrealistic
    print("\n⚖️ Comparing Realistic vs Unrealistic Settings...")
    comparison = compare_realistic_vs_unrealistic()
    if comparison['realistic'] and comparison['unrealistic']:
        real_pnl = comparison['realistic']['performance']['total_pnl']
        unreal_pnl = comparison['unrealistic']['performance']['total_pnl']
        difference = unreal_pnl - real_pnl
        print(f"Unrealistic P&L: ${unreal_pnl:.2f}")
        print(f"Realistic P&L: ${real_pnl:.2f}")
        print(f"Cost of Reality: ${difference:.2f} ({difference/unreal_pnl*100:.1f}%)")
    
    # Test multiple strategies
    print("\n🎯 Professional Strategy Suite Results...")
    suite_results = run_professional_strategy_suite()
    
    # Analyze trading costs
    print("\n💰 Trading Cost Analysis...")
    cost_analysis = analyze_trading_costs()
    for scenario, metrics in cost_analysis.items():
        print(f"{scenario}: ${metrics['net_pnl']:.2f} PnL, {metrics['win_rate']:.1f}% Win Rate")