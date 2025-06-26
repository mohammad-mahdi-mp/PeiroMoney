# risk_manager.py
from config import Config

class RiskManager:
    def __init__(self, total_capital: float, backtest_results: dict = None):
        self.total_capital = total_capital
        self.backtest_results = backtest_results or {}

    def calculate_position_size(self, entry_price: float, stop_loss_price: float,
                                volatility_index: float, leverage: int = 1) -> float:
        """Calculate position size with Kelly Criterion integration"""
        risk_fraction = self.get_optimal_risk_fraction()
        risk_amount = self.total_capital * risk_fraction

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 0:
            return 0

        position_size = (risk_amount / risk_per_unit) * leverage
        position_size *= min(1.0, 1.5 - (volatility_index / 100))

        return position_size

    def get_optimal_risk_fraction(self) -> float:
        """Get optimal risk fraction using Kelly Criterion"""
        win_rate = self.backtest_results.get('win_rate', 0.55)
        avg_win = self.backtest_results.get('avg_win', 1)
        avg_loss = self.backtest_results.get('avg_loss', 1)

        if avg_loss <= 0:
            return Config.RISK_PERCENTAGE / 100

        win_loss_ratio = avg_win / avg_loss
        kelly_f = win_rate - (1 - win_rate) / win_loss_ratio

        return max(0.005, min(kelly_f, 0.05))

    def kelly_position_size(self, win_rate: float, win_loss_ratio: float) -> float:
        """Kelly Criterion-based position sizing formula."""
        return win_rate - (1 - win_rate) / win_loss_ratio

    def multi_target_strategy(self, entry_price: float, atr: float,
                             fib_levels: dict, liquidation_clusters: list,
                             trade_direction: str, bridge_strength: int) -> list:
        """Generates profit targets with dynamic weighting based on bridge strength and volatility"""
        # Base weights
        weights = [0.4, 0.3, 0.3]

        # Adjust weights based on bridge strength
        if bridge_strength >= 8:
            weights = [0.3, 0.2, 0.5]  # Favor liquidity targets
        elif bridge_strength <= 5:
            weights = [0.5, 0.3, 0.2]  # Favor conservative targets

        # Adjust weights based on volatility
        if atr > entry_price * 0.03:  # High volatility
            weights[0] *= 1.2  # Increase conservative target weight

        # Calculate targets
        targets = [
            {'price': self._calc_target(entry_price, 1 * atr, trade_direction, 'add'),
             'type': 'conservative'},
            {'price': self._calc_target(entry_price, fib_levels.get('161.8%', 1.618 * atr), trade_direction, 'level'),
             'type': 'fibonacci'},
            {'price': self._nearest_liquidation_cluster(liquidation_clusters, entry_price, trade_direction),
             'type': 'liquidity'}
        ]

        return [{'price': t['price'], 'weight': w, 'type': t['type']} 
                for t, w in zip(targets, weights)]

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
            return 1.0
        cluster_distances = [abs(c['price'] - price) for c in clusters]
        return min(cluster_distances)

    def _calc_target(self, entry_price, delta, trade_direction, operation_type):
        """Helper to calculate target price based on direction and operation type"""
        if trade_direction not in ['long', 'short']:
            raise ValueError("trade_direction must be 'long' or 'short'")
        if operation_type == 'add':
            return entry_price + delta if trade_direction == 'long' else entry_price - delta
        elif operation_type == 'level':
            return delta
        else:
            raise ValueError("operation_type must be 'add' or 'level'")

    def _nearest_liquidation_cluster(self, clusters: list, current_price: float, trade_direction: str) -> float:
        """Finds the nearest liquidation cluster in the direction of the trade."""
        if not clusters:
            return float('inf')

        if trade_direction == 'long':
            valid_clusters = [c for c in clusters if c['price'] > current_price]
        elif trade_direction == 'short':
            valid_clusters = [c for c in clusters if c['price'] < current_price]
        else:
            raise ValueError("trade_direction must be 'long' or 'short'")

        if not valid_clusters:
            return float('inf')

        return min(valid_clusters, key=lambda c: abs(c['price'] - current_price))['price']