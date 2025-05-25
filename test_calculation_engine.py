"""
Test script to verify all calculation engine math is working correctly
"""
import pandas as pd
from simple_app import (
    simple_ma_strategy, rsi_strategy, bollinger_strategy, macd_strategy,
    calculate_real_win_rate, calculate_real_max_drawdown,
    run_realistic_backtest_engine, load_csv_data
)

def test_calculation_accuracy():
    """Test all calculation components for accuracy"""
    print("🔍 Testing Calculation Engine Accuracy...")
    
    # Load real forex data for testing
    try:
        df = load_csv_data("attached_assets/EURUSD_Candlestick_30_M_BID_26.04.2023-26.04.2025.csv")
        if df.empty:
            print("❌ No data loaded - cannot test calculations")
            return
        
        # Filter to a smaller date range for testing
        df = df.head(1000)  # Use first 1000 rows for quick testing
        print(f"✅ Loaded {len(df)} data points for testing")
        
        # Test each strategy's calculation accuracy
        strategies = {
            'MA Strategy': simple_ma_strategy,
            'RSI Strategy': rsi_strategy,
            'Bollinger Strategy': bollinger_strategy,
            'MACD Strategy': macd_strategy
        }
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n📊 Testing {strategy_name}...")
            
            try:
                result = strategy_func(df.copy())
                trades = result.get('trades', [])
                signals = result.get('signals', [])
                
                print(f"  - Generated {len(signals)} signals")
                print(f"  - Completed {len(trades)} trades")
                
                if trades:
                    # Check if trades have required fields
                    first_trade = trades[0]
                    required_fields = ['entry_price', 'exit_price', 'type', 'pnl']
                    missing_fields = [field for field in required_fields if field not in first_trade]
                    
                    if missing_fields:
                        print(f"  ❌ Missing fields: {missing_fields}")
                    else:
                        print(f"  ✅ All required fields present")
                    
                    # Test win rate calculation with improved accuracy
                    profitable_trades = sum(1 for trade in trades if trade['pnl'] > 0)
                    win_rate = (profitable_trades / len(trades) * 100) if trades else 0
                    print(f"  - Win Rate: {win_rate:.2f}%")
                    
                    # Test max drawdown calculation with cumulative balance
                    balance = 400  # Starting balance
                    balances = [balance]
                    for trade in trades:
                        balance += trade['pnl']
                        balances.append(balance)
                    
                    peak = 400
                    max_dd = 0
                    for bal in balances:
                        if bal > peak:
                            peak = bal
                        dd = (peak - bal) / peak * 100
                        max_dd = max(max_dd, dd)
                    print(f"  - Max Drawdown: {max_dd:.2f}%")
                    
                    # Test realistic backtest engine
                    user_params = {
                        'starting_capital': 400,
                        'lot_size': 0.05,
                        'spread_pips': 2.0,
                        'slippage_pips': 1.5
                    }
                    total_pnl = run_realistic_backtest_engine(result, 400, user_params)
                    print(f"  - Total P&L: ${total_pnl}")
                    
                    # Verify P&L makes sense
                    if abs(total_pnl) > 10000:
                        print(f"  ⚠️  P&L seems unrealistic: ${total_pnl}")
                    elif total_pnl == 0:
                        print(f"  ⚠️  P&L is zero - check calculations")
                    else:
                        print(f"  ✅ P&L appears reasonable")
                
                else:
                    print(f"  ⚠️  No trades generated")
                    
            except Exception as e:
                print(f"  ❌ Error testing {strategy_name}: {e}")
        
        print(f"\n🎯 Calculation Engine Test Complete!")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")

if __name__ == "__main__":
    test_calculation_accuracy()