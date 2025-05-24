"""
Quick Strategy Testing with Custom Settings
Simple interface to test strategies with your preferred parameters
"""
from trading_config import TradingConfig
from strategies.example_strategies import MovingAverageCrossover, RSIStrategy
from strategies.multi_indicator_strategy import SuperTrendStrategy
from strategy_tester import StrategyTester

def quick_test(
    starting_capital=10000,
    leverage=50,
    risk_per_trade=2.0,
    slippage_pips=2.0,
    strategy_type="RSI",
    currency_pair="EURUSD",
    timeframe="30_M"
):
    """Quick test with your custom settings"""
    
    # Create your trading configuration
    config = TradingConfig(
        starting_capital=starting_capital,
        leverage=leverage,
        max_risk_per_trade=risk_per_trade,
        slippage_pips=slippage_pips
    )
    
    # Choose strategy
    if strategy_type == "RSI":
        strategy = RSIStrategy(currency_pair, timeframe, config)
    elif strategy_type == "MovingAverage":
        strategy = MovingAverageCrossover(currency_pair, timeframe, config)
    elif strategy_type == "SuperTrend":
        strategy = SuperTrendStrategy(currency_pair, timeframe, config)
    else:
        strategy = RSIStrategy(currency_pair, timeframe, config)
    
    # Run backtest
    tester = StrategyTester("data")
    result = tester.run_backtest(strategy, currency_pair, timeframe)
    
    if result and 'performance' in result:
        perf = result['performance']
        
        print(f"\n🎯 {strategy_type} Strategy Results on {currency_pair}")
        print("=" * 50)
        print(f"💰 Starting Capital: ${starting_capital:,.2f}")
        print(f"📊 Leverage: {leverage}:1")
        print(f"⚠️  Risk per Trade: {risk_per_trade}%")
        print(f"📈 Net P&L: ${perf['total_pnl']:,.2f}")
        print(f"🎲 Win Rate: {perf['win_rate']:.1f}%")
        print(f"📋 Total Trades: {perf['total_trades']}")
        print(f"💪 Profit Factor: {perf['profit_factor']:.2f}")
        print(f"💵 Final Balance: ${perf['balance']:,.2f}")
        
        # Calculate return percentage
        return_pct = (perf['balance'] - starting_capital) / starting_capital * 100
        print(f"📊 Total Return: {return_pct:+.1f}%")
        
        return result
    else:
        print("❌ Strategy test failed - check your data files")
        return None

if __name__ == "__main__":
    print("🚀 Quick Strategy Tester with Professional Settings")
    print("Testing RSI Strategy with OANDA-compliant parameters...")
    
    # Test with conservative settings
    result = quick_test(
        starting_capital=15000,  # Your starting amount
        leverage=30,             # Conservative leverage
        risk_per_trade=1.5,      # Conservative risk
        slippage_pips=2.0,       # Realistic slippage
        strategy_type="RSI",     # Strategy type
        currency_pair="EURUSD",  # Currency pair
        timeframe="30_M"         # Timeframe
    )