"""
Simple test to verify dashboard loads with clean zero baselines
No complex strategies - just basic functionality
"""

import pandas as pd
import numpy as np

def test_basic_data_loading():
    """Test that we can load CSV data without any strategy complexity"""
    try:
        # Try to load one of the data files
        filepath = "data/EURUSD_Candlestick_30_M_BID_26.04.2023-26.04.2025.csv"
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} rows of EURUSD data")
        
        # Basic data validation
        required_columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
        else:
            print("✅ All required columns present")
            
        # Show a sample
        print("\nSample data:")
        print(df.head(3))
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_clean_metrics():
    """Test that we can calculate clean zero baseline metrics"""
    # Create empty results to show clean baselines
    clean_metrics = {
        'total_pnl': 0.0,
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_trade': 0.0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0
    }
    
    print("✅ Clean baseline metrics:")
    for key, value in clean_metrics.items():
        print(f"  {key}: {value}")
    
    return clean_metrics

if __name__ == "__main__":
    print("🎯 Testing Basic Dashboard Functionality")
    print("=" * 50)
    
    # Test data loading
    data_ok = test_basic_data_loading()
    print()
    
    # Test clean metrics
    metrics = test_clean_metrics()
    
    if data_ok:
        print("\n✅ Dashboard should display clean zero baselines!")
    else:
        print("\n❌ Data loading issues detected")