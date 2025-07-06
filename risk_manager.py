# risk_manager.py
from typing import Optional
import ccxt
import logging
from config import Config

logger = logging.getLogger("CryptoAnalyzer")

class RiskManager:
    def __init__(self, total_capital: float, backtest_results: dict = None):
        self.total_capital = total_capital
        self.backtest_results = backtest_results or {}

    def calculate_bridge_position(self, 
                                  entry_price: float, 
                                  stop_loss_price: float,
                                  volatility_index: float,
                                  bridge_strength: float,
                                  liquidation_risk: float,
                                  exchange: str = "kucoin",
                                  symbol: str = "BTC/USDT") -> float:
        """
        Calculate position size for bridge strategy with exchange minimums
        
        Parameters:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        volatility_index: Market volatility (0-100)
        bridge_strength: Bridge strength score (0-10)
        liquidation_risk: Liquidation risk score
        exchange: Exchange name
        symbol: Trading pair symbol
        
        Returns: Position size in base currency
        """
        try:
            # Input validation
            if not all([entry_price > 0, stop_loss_price > 0, 0 <= volatility_index <= 100]):
                raise ValueError("Invalid input values")
            if not 0 <= bridge_strength <= 10:
                raise ValueError("Bridge strength must be 0-10")
            if liquidation_risk < 0:
                raise ValueError("Liquidation risk cannot be negative")

            # 1. Calculate base risk using Kelly Criterion
            risk_fraction = self.get_optimal_risk_fraction()
            base_risk = self.total_capital * risk_fraction

            # 2. Apply bridge strength multiplier (1.0x to 2.0x)
            bridge_multiplier = 1 + (bridge_strength / 10)
            adjusted_risk = base_risk * bridge_multiplier

            # 3. Apply volatility dampening
            volatility_dampening = min(1.0, 1.5 - (volatility_index / 100))
            adjusted_risk *= volatility_dampening

            # 4. Apply liquidation risk scaling
            liquidation_scaling = max(0.01, 1 - (liquidation_risk / 200))
            adjusted_risk *= liquidation_scaling

            # 5. Calculate position size
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit <= 0:
                logger.warning("Risk per unit <= 0")
                return 0.0

            position_size = adjusted_risk / risk_per_unit

            # 6. Apply exchange minimums
            min_size = self.get_exchange_min_size(exchange, symbol)
            if position_size < min_size:
                logger.warning(f"Position size {position_size:.6f} < min {min_size:.6f}")
                return 0.0

            # 7. Cap position size (max 10% of total capital)
            max_position = self.total_capital * 0.10
            return min(position_size, max_position)

        except Exception as e:
            logger.error(f"Bridge position error: {e}")
            # Fallback to standard calculation
            return self.calculate_position_size(
                entry_price, 
                stop_loss_price, 
                volatility_index
            )

    def get_exchange_min_size(self, exchange: str, symbol: str) -> float:
        """Get minimum trade size for exchange/symbol"""
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange.lower())
            exch = exchange_class({'enableRateLimit': True})
            exch.load_markets()
            
            # Get market data
            market = exch.market(symbol)
            return float(market['limits']['amount']['min'])
        
        except Exception as e:
            logger.warning(f"Min size error: {e}, using default")
            # Default minimums for major exchanges
            defaults = {
                'binance': 0.001,
                'kucoin': 0.001,
                'bybit': 0.001,
                'okx': 0.001
            }
            return defaults.get(exchange.lower(), 0.001)

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