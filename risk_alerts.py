"""
Enhanced Risk Management with Custom Alerts and Position Sizing
Answers user questions about protective messages and lot size control
"""

def check_account_health(current_balance: float, starting_capital: float = 10000, trades_count: int = 0) -> dict:
    """
    Check if your account is healthy and provide protective messages
    """
    balance_percent = (current_balance / starting_capital) * 100
    loss_percent = 100 - balance_percent
    
    # Risk levels and messages
    if current_balance <= 200:
        return {
            'status': 'CRITICAL_RISK',
            'message': '🚨 CRITICAL: Account below $200! Strategy performance is poor. Consider stopping.',
            'recommendation': 'Stop trading immediately and review strategy parameters',
            'allow_trading': False,
            'risk_level': 'EXTREME'
        }
    elif loss_percent >= 20:
        return {
            'status': 'HIGH_RISK', 
            'message': f'⚠️ HIGH RISK: Account down {loss_percent:.1f}% (${current_balance:.2f})',
            'recommendation': 'Reduce position sizes or pause trading',
            'allow_trading': True,
            'risk_level': 'HIGH'
        }
    elif loss_percent >= 10:
        return {
            'status': 'MODERATE_RISK',
            'message': f'⚡ CAUTION: Account down {loss_percent:.1f}% (${current_balance:.2f})',
            'recommendation': 'Monitor closely and consider reducing risk',
            'allow_trading': True,
            'risk_level': 'MODERATE'
        }
    else:
        return {
            'status': 'HEALTHY',
            'message': f'✅ HEALTHY: Account at ${current_balance:.2f} ({balance_percent:.1f}%)',
            'recommendation': 'Continue with current strategy',
            'allow_trading': True,
            'risk_level': 'LOW'
        }

def calculate_custom_lot_size(account_balance: float, risk_percent: float = 2.0, 
                            stop_loss_pips: float = 20, custom_lot: float = None) -> dict:
    """
    Calculate position size with custom lot size options
    User can override automatic calculation
    """
    # Automatic calculation based on risk
    pip_value = 10  # For EUR/USD standard lot
    risk_amount = account_balance * (risk_percent / 100)
    auto_lot_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Apply OANDA limits
    min_lot = 0.01
    max_lot = 100.0
    auto_lot_size = max(min_lot, min(auto_lot_size, max_lot))
    
    # Use custom lot if provided
    final_lot_size = custom_lot if custom_lot else auto_lot_size
    final_lot_size = max(min_lot, min(final_lot_size, max_lot))
    
    return {
        'recommended_lot': round(auto_lot_size, 2),
        'final_lot': round(final_lot_size, 2),
        'is_custom': custom_lot is not None,
        'risk_amount': risk_amount,
        'message': f'💰 Position: {final_lot_size:.2f} lots (Risk: ${risk_amount:.2f})',
        'custom_override': custom_lot is not None
    }

def trading_session_summary(trades: list, starting_capital: float = 10000) -> dict:
    """
    Provide session summary with risk warnings
    """
    if not trades:
        return {'message': '📊 No trades executed yet', 'status': 'READY'}
    
    total_pnl = sum(trade.get('pnl', 0) for trade in trades)
    current_balance = starting_capital + total_pnl
    win_trades = len([t for t in trades if t.get('pnl', 0) > 0])
    total_trades = len(trades)
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    health = check_account_health(current_balance, starting_capital, total_trades)
    
    return {
        'current_balance': current_balance,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'health_status': health,
        'summary_message': f'📈 Session: {total_trades} trades, {win_rate:.1f}% wins, P&L: ${total_pnl:.2f}'
    }

# Example usage for your questions:
def demo_risk_scenarios():
    """
    Examples showing protective messages when account goes below $200
    """
    print("🎯 Risk Management Examples:")
    print()
    
    # Scenario 1: Account drops to $150
    result1 = check_account_health(150, 10000)
    print(f"Account at $150: {result1['message']}")
    print(f"Trading allowed: {result1['allow_trading']}")
    print()
    
    # Scenario 2: Custom lot size selection  
    lot_calc = calculate_custom_lot_size(5000, risk_percent=2.0, custom_lot=0.05)
    print(f"Custom lot sizing: {lot_calc['message']}")
    print(f"Recommended: {lot_calc['recommended_lot']}, Using: {lot_calc['final_lot']}")
    print()
    
    # Scenario 3: High risk warning
    result3 = check_account_health(7500, 10000)  # 25% loss
    print(f"High loss scenario: {result3['message']}")
    print(f"Recommendation: {result3['recommendation']}")

if __name__ == "__main__":
    demo_risk_scenarios()