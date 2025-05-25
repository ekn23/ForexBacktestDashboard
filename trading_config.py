"""
Trading Configuration and Risk Management
Following OANDA and industry-standard trading rules
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class TradingConfig:
    """Trading configuration following OANDA rules"""
    
    # Account Settings
    starting_capital: float = 10000.0
    currency: str = "USD"
    leverage: float = 50.0  # OANDA standard leverage
    
    # Risk Management
    max_risk_per_trade: float = 2.0  # Maximum 2% per trade
    max_daily_loss: float = 5.0  # Maximum 5% daily loss
    max_drawdown: float = 20.0  # Maximum 20% drawdown
    
    # Trading Costs (OANDA typical spreads and costs)
    spread_pips: Dict[str, float] = None
    commission_per_lot: float = 0.0  # OANDA no commission model
    overnight_swap_long: float = 0.0  # Will be calculated per pair
    overnight_swap_short: float = 0.0
    
    # Slippage and Execution
    slippage_pips: float = 2.0  # 2 pip slippage
    max_slippage_pips: float = 5.0  # Maximum acceptable slippage
    
    # Position Sizing
    min_lot_size: float = 0.01  # OANDA minimum
    max_lot_size: float = 100.0  # OANDA maximum for retail
    lot_step: float = 0.01  # OANDA lot step
    
    # Margin Requirements (OANDA standards)
    margin_requirements: Dict[str, float] = None
    
    # Trading Hours and Sessions
    trading_sessions: Dict[str, Dict] = None
    
    def __post_init__(self):
        """Initialize default values after creation"""
        if self.spread_pips is None:
            self.spread_pips = {
                'EURUSD': 1.0,
                'GBPUSD': 1.5,
                'USDJPY': 1.0,
                'USDCHF': 1.5,
                'USDCAD': 1.5,
                'AUDUSD': 1.5,
                'NZDUSD': 2.0,
                'EURJPY': 2.0,
                'GBPJPY': 3.0,
                'EURGBP': 1.5
            }
        
        if self.margin_requirements is None:
            # OANDA margin requirements (as percentage)
            self.margin_requirements = {
                'EURUSD': 2.0,  # 50:1 leverage = 2% margin
                'GBPUSD': 2.0,
                'USDJPY': 2.0,
                'USDCHF': 2.0,
                'USDCAD': 2.0,
                'AUDUSD': 2.0,
                'NZDUSD': 2.0,
                'EURJPY': 3.33,  # 30:1 leverage = 3.33% margin
                'GBPJPY': 3.33,
                'EURGBP': 2.0
            }
        
        if self.trading_sessions is None:
            self.trading_sessions = {
                'sydney': {'start': '21:00', 'end': '06:00'},
                'tokyo': {'start': '00:00', 'end': '09:00'},
                'london': {'start': '08:00', 'end': '17:00'},
                'new_york': {'start': '13:00', 'end': '22:00'}
            }

class RiskManager:
    """Advanced risk management following OANDA rules"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.max_drawdown_reached = 0.0
        self.open_positions_value = 0.0
    
    def calculate_position_size(self, account_balance: float, stop_loss_pips: float, 
                              risk_percent: float = None) -> float:
        """Calculate position size based on risk management"""
        if risk_percent is None:
            risk_percent = self.config.max_risk_per_trade
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate pip value for standard lot (100,000 units)
        pip_value = 10.0  # USD for most major pairs
        
        # Calculate position size
        if stop_loss_pips > 0:
            position_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            position_size = self.config.min_lot_size
        
        # Apply OANDA constraints
        position_size = max(self.config.min_lot_size, position_size)
        position_size = min(self.config.max_lot_size, position_size)
        
        # Round to OANDA lot step
        position_size = round(position_size / self.config.lot_step) * self.config.lot_step
        
        return position_size
    
    def calculate_margin_required(self, symbol: str, lot_size: float, current_price: float) -> float:
        """Calculate margin required for position"""
        margin_percent = self.config.margin_requirements.get(symbol, 3.33)
        position_value = lot_size * 100000 * current_price  # Standard lot value
        
        return position_value * (margin_percent / 100)
    
    def check_margin_availability(self, account_balance: float, symbol: str, 
                                 lot_size: float, current_price: float) -> bool:
        """Check if sufficient margin is available"""
        required_margin = self.calculate_margin_required(symbol, lot_size, current_price)
        available_margin = account_balance - self.open_positions_value
        
        return available_margin >= required_margin
    
    def apply_slippage(self, intended_price: float, symbol: str, action: str) -> float:
        """Apply realistic slippage to order execution"""
        slippage_pips = self.config.slippage_pips
        
        # Convert pips to price
        if 'JPY' in symbol:
            pip_size = 0.01  # JPY pairs
        else:
            pip_size = 0.0001  # Other major pairs
        
        slippage_amount = slippage_pips * pip_size
        
        # Apply slippage based on order direction
        if action == 'BUY':
            return intended_price + slippage_amount
        else:  # SELL
            return intended_price - slippage_amount
    
    def calculate_spread_cost(self, symbol: str, lot_size: float) -> float:
        """Calculate spread cost for position"""
        spread_pips = self.config.spread_pips.get(symbol, 2.0)
        pip_value = 10.0 * lot_size  # USD value per pip for lot size
        
        if 'JPY' in symbol:
            pip_value = pip_value / 100  # Adjust for JPY pairs
        
        return spread_pips * pip_value
    
    def calculate_swap(self, symbol: str, lot_size: float, action: str, days: int = 1) -> float:
        """Calculate overnight swap/rollover costs"""
        # Simplified swap calculation - in practice, get from OANDA API
        swap_rates = {
            'EURUSD': {'long': -0.5, 'short': 0.1},
            'GBPUSD': {'long': -0.8, 'short': 0.3},
            'USDJPY': {'long': 0.2, 'short': -0.7},
            'USDCHF': {'long': 0.1, 'short': -0.6},
            'USDCAD': {'long': 0.0, 'short': -0.3},
            'AUDUSD': {'long': -0.3, 'short': -0.2},
            'NZDUSD': {'long': -0.4, 'short': -0.1}
        }
        
        pair_swaps = swap_rates.get(symbol, {'long': 0.0, 'short': 0.0})
        swap_rate = pair_swaps['long'] if action == 'BUY' else pair_swaps['short']
        
        return swap_rate * lot_size * days
    
    def check_trading_hours(self, symbol: str, timestamp: pd.Timestamp) -> bool:
        """Check if trading is allowed during current time"""
        # Forex markets are open 24/5, closed on weekends
        weekday = timestamp.weekday()
        
        # Market closed on Saturday (5) and Sunday (6)
        if weekday >= 5:
            return False
        
        # Friday close at 22:00 UTC, Sunday open at 22:00 UTC
        if weekday == 4 and timestamp.hour >= 22:  # Friday evening
            return False
        
        if weekday == 6 and timestamp.hour < 22:  # Sunday before opening
            return False
        
        return True
    
    def check_daily_loss_limit(self, current_pnl: float) -> bool:
        """Check if daily loss limit is exceeded"""
        daily_loss_limit = self.config.starting_capital * (self.config.max_daily_loss / 100)
        return abs(current_pnl) < daily_loss_limit
    
    def check_max_drawdown(self, current_balance: float, peak_balance: float) -> bool:
        """Check if maximum drawdown limit is exceeded"""
        if peak_balance > 0:
            drawdown_percent = ((peak_balance - current_balance) / peak_balance) * 100
            return drawdown_percent < self.config.max_drawdown
        return True

class OANDATradeExecutor:
    """Simulates OANDA-compliant trade execution"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
    
    def execute_order(self, symbol: str, action: str, lot_size: float, 
                     current_price: float, stop_loss: float = None, 
                     take_profit: float = None, account_balance: float = 10000) -> Dict:
        """Execute order with OANDA-compliant validation and costs"""
        
        # Validate lot size
        if lot_size < self.config.min_lot_size:
            return {'success': False, 'error': 'Lot size below minimum'}
        
        if lot_size > self.config.max_lot_size:
            return {'success': False, 'error': 'Lot size above maximum'}
        
        # Check margin availability
        if not self.risk_manager.check_margin_availability(account_balance, symbol, lot_size, current_price):
            return {'success': False, 'error': 'Insufficient margin'}
        
        # Apply slippage
        execution_price = self.risk_manager.apply_slippage(current_price, symbol, action)
        
        # Calculate costs
        spread_cost = self.risk_manager.calculate_spread_cost(symbol, lot_size)
        margin_required = self.risk_manager.calculate_margin_required(symbol, lot_size, execution_price)
        
        # Validate stop loss and take profit distances (OANDA minimum distances)
        min_distance_pips = 5  # Minimum 5 pips distance
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        min_distance = min_distance_pips * pip_size
        
        if stop_loss and abs(execution_price - stop_loss) < min_distance:
            return {'success': False, 'error': 'Stop loss too close to market price'}
        
        if take_profit and abs(execution_price - take_profit) < min_distance:
            return {'success': False, 'error': 'Take profit too close to market price'}
        
        return {
            'success': True,
            'execution_price': execution_price,
            'lot_size': lot_size,
            'spread_cost': spread_cost,
            'margin_required': margin_required,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'slippage': execution_price - current_price
        }
    
    def close_position(self, symbol: str, action: str, lot_size: float, 
                      open_price: float, close_price: float, days_held: int = 0) -> Dict:
        """Close position with all costs calculated"""
        
        # Apply slippage to closing price
        execution_price = self.risk_manager.apply_slippage(close_price, symbol, 
                                                          'SELL' if action == 'BUY' else 'BUY')
        
        # Calculate P&L
        if action == 'BUY':
            pnl = (execution_price - open_price) * lot_size * 100000
        else:  # SELL
            pnl = (open_price - execution_price) * lot_size * 100000
        
        # Subtract costs
        spread_cost = self.risk_manager.calculate_spread_cost(symbol, lot_size)
        swap_cost = self.risk_manager.calculate_swap(symbol, lot_size, action, days_held)
        
        net_pnl = pnl - spread_cost - swap_cost
        
        return {
            'gross_pnl': pnl,
            'spread_cost': spread_cost,
            'swap_cost': swap_cost,
            'net_pnl': net_pnl,
            'execution_price': execution_price,
            'days_held': days_held
        }