# risk_manager.py
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

    # ———————————————————————
    # 🚀 Advanced Strategy Additions
    # ———————————————————————

    def kelly_position_size(self, win_rate: float, win_loss_ratio: float) -> float:
        """Kelly Criterion-based position sizing formula."""
        return win_rate - (1 - win_rate) / win_loss_ratio

    def multi_target_strategy(self, entry_price: float, atr: float,
                             fib_levels: dict, liquidation_clusters: list) -> list:
        """Generates a list of profit targets based on ATR, Fibonacci levels, and liquidation clusters."""
        return [
            {'price': entry_price + 1 * atr, 'weight': 0.4},
            {'price': fib_levels.get('161.8%', entry_price + 1.618 * atr), 'weight': 0.3},
            {'price': self._nearest_liquidation_cluster(liquidation_clusters, entry_price), 'weight': 0.3}
        ]

    def dynamic_stoploss(self, entry_price: float, swing_point: float,
                         atr: float, trend_direction: str) -> float:
        """Calculates a volatility-adjusted stoploss based on swing points and ATR."""
        buffer = 0.5 * atr
        if trend_direction == 'long':
            return swing_point - buffer
        elif trend_direction == 'short':
            return swing_point + buffer
        else:
            raise ValueError("trend_direction must be 'long' or 'short'")

    def liquidation_risk_score(self, price: float, clusters: list) -> float:
        """Calculates the risk score based on proximity to liquidation clusters."""
        if not clusters:
            return 1.0  # No clusters = low risk
        cluster_distances = [abs(c['price'] - price) for c in clusters]
        return min(cluster_distances)

    def _nearest_liquidation_cluster(self, clusters: list, current_price: float) -> float:
        """Finds the nearest liquidation cluster to the current price."""
        if not clusters:
            return float('inf')  # No clusters = no impact
        return min(clusters, key=lambda c: abs(c['price'] - current_price))['price']