# trading/risk_manager.py
from config import Config

class RiskManager:
    def __init__(self, total_capital: float, backtest_results: dict = None):
        self.total_capital = total_capital
        self.backtest_results = backtest_results or {}
        
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                              volatility_index: float, leverage: int = 1) -> float:
        """Calculate position size with Kelly Criterion integration"""
        # Calculate risk amount
        risk_fraction = self.get_optimal_risk_fraction()
        risk_amount = self.total_capital * risk_fraction
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return 0
            
        # Calculate position size
        position_size = (risk_amount / risk_per_unit) * leverage
        
        # Volatility adjustment (reduce size in high volatility)
        position_size *= min(1.0, 1.5 - (volatility_index / 100))
        
        return position_size
    
    def get_optimal_risk_fraction(self) -> float:
        """Get optimal risk fraction using Kelly Criterion"""
        win_rate = self.backtest_results.get('win_rate', 0.55)
        avg_win = self.backtest_results.get('avg_win', 1)
        avg_loss = self.backtest_results.get('avg_loss', 1)
        
        if avg_loss <= 0:
            return Config.RISK_PERCENTAGE / 100  # Use default risk percentage
            
        win_loss_ratio = avg_win / avg_loss
        kelly_f = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Constrain between 0.5% and 5%
        return max(0.005, min(kelly_f, 0.05))