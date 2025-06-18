# trading/engine.py
from typing import List, Dict, Callable, Tuple, Optional, Any
import numpy as np
from config import Config
from utils import setup_logger

logger = setup_logger()

class IntelligentTradingEngine:
    def __init__(self, tech_analyzer):
        self.ta = tech_analyzer
        self.df = tech_analyzer.df
        self.latest = self.df.iloc[-1]
        self.context = self._determine_market_context()
    def _determine_market_context(self) -> Dict[str, Any]:
        """Determine market context with strength assessment"""
        context = {
            'type': 'Unknown',
            'strength': 0.0,
            'confidence': 0.0
        }
        
        # Use market structure analysis
        if hasattr(self.ta, 'market_structure'):
            ms = self.ta.market_structure
            context['type'] = ms['trend']
            context['strength'] = ms['trend_strength']
            
            # Confidence based on ADX and sequence length
            adx_score = min(1.0, self.latest.get('adx', 0) / 100)
            sequence_length = len(ms['sequence'])
            sequence_score = min(1.0, sequence_length / 5)  # Max 5 swings
            context['confidence'] = (adx_score * 0.6) + (sequence_score * 0.4)
        
        # Override with ranging market if ADX is low
        if context['type'] != 'Ranging' and self.latest.get('adx', 0) < 25:
            context['type'] = 'Ranging'
            context['strength'] = 0.5
            context['confidence'] = 0.7  # High confidence for ranging markets with low ADX
            
        return context
    
    def generate_trade_idea(self) -> Optional[Dict[str, Any]]:
        """Generate trade idea based on probabilistic scoring"""
        if len(self.df) < 100:
            return None
            
        # Initialize trade setup
        trade_setup = {
            'score': 0,
            'reasoning_log': [],
            'entry_price': None,
            'stop_loss': None,
            'targets': []
        }
        
        # Evaluate scenarios based on market context
        if self.context['type'] in ['Uptrend', 'Downtrend']:
            # Evaluate trend continuation first
            trend_continuation = self._evaluate_trend_continuation_setup()
            if trend_continuation['score'] >= Config.MIN_CONFIDENCE_SCORE:
                trade_setup = trend_continuation
            else:
                # Evaluate trend reversal if continuation not found
                trend_reversal = self._evaluate_trend_reversal_setup()
                if trend_reversal['score'] >= Config.MIN_CONFIDENCE_SCORE:
                    trade_setup = trend_reversal
        else:  # Ranging market
            range_reversal = self._evaluate_range_reversal_setup()
            if range_reversal['score'] >= Config.MIN_CONFIDENCE_SCORE:
                trade_setup = range_reversal
        
        # Format and return if valid setup found
        if trade_setup['score'] >= Config.MIN_CONFIDENCE_SCORE:
            return self._format_trade_idea(trade_setup)
        return None
    
    def _evaluate_trend_continuation_setup(self) -> Dict[str, Any]:
        """Evaluate trend continuation setup with scoring"""
        setup = {
            'score': 0,
            'reasoning_log': [],
            'entry_price': self.latest['close'],
            'stop_loss': None,
            'targets': [],
            'signal': 'BUY' if self.context['type'] == 'Uptrend' else 'SELL'
        }
        
        # Base score for trend context
        setup['score'] += int(self.context['confidence'] * 30)
        setup['reasoning_log'].append(
            f"(+{int(self.context['confidence']*30)}) Context: {self.context['type']} with {self.context['confidence']*100:.1f}% confidence"
        )
        
        # Pullback location scoring
        zone_type = 'demand' if self.context['type'] == 'Uptrend' else 'supply'
        zones = getattr(self.ta, f'{zone_type}_zones', [])
        
        if zones:
            # Find nearest zone
            current_price = self.latest['close']
            nearest_zone = min(
                zones, 
                key=lambda z: abs(z['price'] - current_price),
                default=None
            )
            if nearest_zone:
                # Check if price is near the zone
                price_diff = abs(nearest_zone['price'] - current_price)
                if price_diff / current_price < 0.015:  # Within 1.5%
                    # Zone strength scoring
                    strength_bonus = nearest_zone['strength_score'] * 2.5
                    setup['score'] += strength_bonus
                    setup['reasoning_log'].append(
                        f"(+{strength_bonus:.0f}) Location: Pullback to {zone_type} zone (Strength: {nearest_zone['strength_score']}/10)"
                    )
                    
                    # MA confluence scoring
                    if 'ma_50' in self.latest:
                        ma_diff = abs(nearest_zone['price'] - self.latest['ma_50'])
                        if ma_diff / current_price < 0.01:  # Within 1%
                            setup['score'] += 15
                            setup['reasoning_log'].append("(+15) Confluence: Zone aligns with 50 MA")
                    
                    # Entry trigger scoring
                    bullish_pattern = self.latest.get('hammer') or self.latest.get('bullish_engulfing')
                    bearish_pattern = self.latest.get('shooting_star') or self.latest.get('bearish_engulfing')
                    
                    if (self.context['type'] == 'Uptrend' and bullish_pattern) or \
                       (self.context['type'] == 'Downtrend' and bearish_pattern):
                        setup['score'] += 20
                        setup['reasoning_log'].append(
                            f"(+20) Trigger: {'Bullish' if self.context['type'] == 'Uptrend' else 'Bearish'} pattern confirmed"
                        )
        
        # Momentum scoring
        if self.context['type'] == 'Uptrend':
            if self.latest['rsi'] > 45 and self.latest['macd_diff'] > 0:
                setup['score'] += 10
                setup['reasoning_log'].append("(+10) Momentum: RSI and MACD bullish")
        else:  # Downtrend
            if self.latest['rsi'] < 55 and self.latest['macd_diff'] < 0:
                setup['score'] += 10
                setup['reasoning_log'].append("(+10) Momentum: RSI and MACD bearish")
        
        # Risk/Reward scoring
        if setup['score'] > 0:
            stop_loss = self._calculate_structural_stop_loss(setup['signal'])
            setup['stop_loss'] = stop_loss
            
            # Calculate risk
            risk = abs(setup['entry_price'] - stop_loss)
            
            # Find nearest liquidity pool for TP
            pool_type = 'highs' if self.context['type'] == 'Uptrend' else 'lows'
            pools = self.ta.liquidity_pools.get(pool_type, [])
            
            if pools:
                # Find nearest pool above current price for uptrend, below for downtrend
                if self.context['type'] == 'Uptrend':
                    valid_pools = [p for p in pools if p['price'] > setup['entry_price']]
                    if valid_pools:
                        nearest_pool = min(valid_pools, key=lambda p: p['price'] - setup['entry_price'])
                        reward = nearest_pool['price'] - setup['entry_price']
                        rr_ratio = reward / risk
                        
                        if rr_ratio >= 2:
                            setup['score'] += 10
                            setup['reasoning_log'].append(f"(+10) Risk/Reward: Favorable {rr_ratio:.1f}:1 R/R")
                            setup['targets'].append({
                                'price': nearest_pool['price'],
                                'reason': f"Nearest liquidity pool ({nearest_pool['count']} clusters)"
                            })
                        else:
                            setup['score'] -= 5
                            setup['reasoning_log'].append(f"(-5) Risk/Reward: Poor {rr_ratio:.1f}:1 R/R")
                else:  # Downtrend
                    valid_pools = [p for p in pools if p['price'] < setup['entry_price']]
                    if valid_pools:
                        nearest_pool = max(valid_pools, key=lambda p: setup['entry_price'] - p['price'])
                        reward = setup['entry_price'] - nearest_pool['price']
                        rr_ratio = reward / risk
                        
                        if rr_ratio >= 2:
                            setup['score'] += 10
                            setup['reasoning_log'].append(f"(+10) Risk/Reward: Favorable {rr_ratio:.1f}:1 R/R")
                            setup['targets'].append({
                                'price': nearest_pool['price'],
                                'reason': f"Nearest liquidity pool ({nearest_pool['count']} clusters)"
                            })
                        else:
                            setup['score'] -= 5
                            setup['reasoning_log'].append(f"(-5) Risk/Reward: Poor {rr_ratio:.1f}:1 R/R")
        
        # Negative factors - ChoCH warning
        if hasattr(self.ta, 'market_structure') and self.ta.market_structure.get('choch', False):
            setup['score'] -= 40
            setup['reasoning_log'].append("(-40) Warning: Recent character change detected")
        return setup
    
    def _evaluate_range_reversal_setup(self) -> Dict[str, Any]:
        """Evaluate range reversal setup with scoring"""
        setup = {
            'score': 0,
            'reasoning_log': [],
            'entry_price': self.latest['close'],
            'stop_loss': None,
            'targets': [],
            'signal': None
        }
        
        # Base score for ranging context
        setup['score'] += int(self.context['confidence'] * 30)
        setup['reasoning_log'].append(
            f"(+{int(self.context['confidence']*30)}) Context: Ranging market with {self.context['confidence']*100:.1f}% confidence"
        )
        
        # Check supply zones for short signals
        if self.ta.supply_zones:
            nearest_supply = min(
                self.ta.supply_zones, 
                key=lambda z: abs(z['price'] - self.latest['close']),
                default=None
            )
            
            if nearest_supply and abs(nearest_supply['price'] - self.latest['close']) / self.latest['close'] < 0.015:
                # Zone strength scoring
                strength_bonus = nearest_supply['strength_score'] * 2.5
                setup['score'] += strength_bonus
                setup['reasoning_log'].append(
                    f"(+{strength_bonus:.0f}) Location: Approach to supply zone (Strength: {nearest_supply['strength_score']}/10)"
                )
                
                # Bearish confirmation
                if self.latest['rsi'] > 65 or self.latest['macd_diff'] < 0 or \
                   self.latest.get('shooting_star') or self.latest.get('evening_star'):
                    setup['score'] += 20
                    setup['reasoning_log'].append("(+20) Confirmation: Bearish signals present")
                    setup['signal'] = 'SELL'
        
        # Check demand zones for long signals
        if not setup['signal'] and self.ta.demand_zones:
            nearest_demand = min(
                self.ta.demand_zones, 
                key=lambda z: abs(z['price'] - self.latest['close']),
                default=None
            )
            
            if nearest_demand and abs(nearest_demand['price'] - self.latest['close']) / self.latest['close'] < 0.015:
                # Zone strength scoring
                strength_bonus = nearest_demand['strength_score'] * 2.5
                setup['score'] += strength_bonus
                setup['reasoning_log'].append(
                    f"(+{strength_bonus:.0f}) Location: Approach to demand zone (Strength: {nearest_demand['strength_score']}/10)"
                )
                
                # Bullish confirmation
                if self.latest['rsi'] < 35 or self.latest['macd_diff'] > 0 or \
                   self.latest.get('hammer') or self.latest.get('morning_star'):
                    setup['score'] += 20
                    setup['reasoning_log'].append("(+20) Confirmation: Bullish signals present")
                    setup['signal'] = 'BUY'
        
        # Risk/Reward assessment
        if setup['signal']:
            stop_loss = self._calculate_structural_stop_loss(setup['signal'])
            setup['stop_loss'] = stop_loss
            risk = abs(setup['entry_price'] - stop_loss)
            
            # Find opposite boundary for TP
            if setup['signal'] == 'SELL':
                # For shorts, target demand zone
                if self.ta.demand_zones:
                    nearest_demand = min(
                        self.ta.demand_zones, 
                        key=lambda z: abs(z['price'] - setup['entry_price']),
                        default=None
                    )
                    if nearest_demand:
                        reward = setup['entry_price'] - nearest_demand['price']
                        rr_ratio = reward / risk
                        
                        if rr_ratio >= 2:
                            setup['score'] += 10
                            setup['reasoning_log'].append(f"(+10) Risk/Reward: Favorable {rr_ratio:.1f}:1 R/R")
                            setup['targets'].append({
                                'price': nearest_demand['price'],
                                'reason': "Nearest demand zone"
                            })
            else:  # BUY
                if self.ta.supply_zones:
                    nearest_supply = min(
                        self.ta.supply_zones, 
                        key=lambda z: abs(z['price'] - setup['entry_price']),
                        default=None
                    )
                    if nearest_supply:
                        reward = nearest_supply['price'] - setup['entry_price']
                        rr_ratio = reward / risk
                        
                        if rr_ratio >= 2:
                            setup['score'] += 10
                            setup['reasoning_log'].append(f"(+10) Risk/Reward: Favorable {rr_ratio:.1f}:1 R/R")
                            setup['targets'].append({
                                'price': nearest_supply['price'],
                                'reason': "Nearest supply zone"
                            })
        
        return setup
    
    def _evaluate_trend_reversal_setup(self) -> Dict[str, Any]:
        """Evaluate trend reversal setup with scoring"""
        setup = {
            'score': 0,
            'reasoning_log': [],
            'entry_price': self.latest['close'],
            'stop_loss': None,
            'targets': [],
            'signal': 'BUY' if self.context['type'] == 'Downtrend' else 'SELL'  # Reversal signal
        }
        
        # Require recent ChoCH
        if hasattr(self.ta, 'market_structure') and self.ta.market_structure.get('choch', False):
            setup['score'] += 30
            setup['reasoning_log'].append("(+30) Context: Recent character change detected")
        else:
            return setup  # No reversal without ChoCH
        
        # Check for confirmation patterns
        if setup['signal'] == 'BUY':  # Reversal from downtrend to uptrend
            if self.latest.get('hammer') or self.latest.get('morning_star'):
                setup['score'] += 20
                setup['reasoning_log'].append("(+20) Confirmation: Bullish reversal pattern")
            
            # Check if RSI shows bullish divergence
            if self._detect_bullish_divergence():
                setup['score'] += 15
                setup['reasoning_log'].append("(+15) Confirmation: Bullish RSI divergence")
        else:  # SELL - reversal from uptrend to downtrend
            if self.latest.get('shooting_star') or self.latest.get('evening_star'):
                setup['score'] += 20
                setup['reasoning_log'].append("(+20) Confirmation: Bearish reversal pattern")
            
            # Check if RSI shows bearish divergence
            if self._detect_bearish_divergence():
                setup['score'] += 15
                setup['reasoning_log'].append("(+15) Confirmation: Bearish RSI divergence")
        
        # Structural stop loss
        if setup['score'] > 50:
            setup['stop_loss'] = self._calculate_structural_stop_loss(setup['signal'])
            
            # Find liquidity pools for targets
            pool_type = 'highs' if setup['signal'] == 'SELL' else 'lows'
            pools = self.ta.liquidity_pools.get(pool_type, [])
            
            if pools:
                if setup['signal'] == 'SELL':
                    valid_pools = [p for p in pools if p['price'] > setup['entry_price']]
                    if valid_pools:
                        nearest_pool = min(valid_pools, key=lambda p: p['price'] - setup['entry_price'])
                        setup['targets'].append({
                            'price': nearest_pool['price'],
                            'reason': f"Nearest liquidity pool ({nearest_pool['count']} clusters)"
                        })
                else:  # BUY
                    valid_pools = [p for p in pools if p['price'] < setup['entry_price']]
                    if valid_pools:
                        nearest_pool = max(valid_pools, key=lambda p: setup['entry_price'] - p['price'])
                        setup['targets'].append({
                            'price': nearest_pool['price'],
                            'reason': f"Nearest liquidity pool ({nearest_pool['count']} clusters)"
                        })
        
        return setup
    
    def _detect_bullish_divergence(self) -> bool:
        """Detect bullish RSI divergence"""
        if len(self.df) < 20:
            return False
            
        # Find last two swing lows in price and RSI
        price_lows = []
        rsi_lows = []
        
        for i in range(len(self.df)-1, max(0, len(self.df)-30), -1):
            if not np.isnan(self.df.iloc[i]['swing_low']):
                price_lows.append((i, self.df.iloc[i]['swing_low']))
                rsi_lows.append((i, self.df.iloc[i]['rsi']))
                if len(price_lows) >= 2:
                    break
        
        if len(price_lows) < 2:
            return False
            
        # Check for divergence: lower price lows but higher RSI lows
        if price_lows[0][1] < price_lows[1][1] and rsi_lows[0][1] > rsi_lows[1][1]:
            return True
            
        return False
    
    def _detect_bearish_divergence(self) -> bool:
        """Detect bearish RSI divergence"""
        if len(self.df) < 20:
            return False
            
        # Find last two swing highs in price and RSI
        price_highs = []
        rsi_highs = []
        
        for i in range(len(self.df)-1, max(0, len(self.df)-30), -1):
            if not np.isnan(self.df.iloc[i]['swing_high']):
                price_highs.append((i, self.df.iloc[i]['swing_high']))
                rsi_highs.append((i, self.df.iloc[i]['rsi']))
                if len(price_highs) >= 2:
                    break
        
        if len(price_highs) < 2:
            return False
            
        # Check for divergence: higher price highs but lower RSI highs
        if price_highs[0][1] > price_highs[1][1] and rsi_highs[0][1] < rsi_highs[1][1]:
            return True
            
        return False    
    def _calculate_structural_stop_loss(self, signal: str) -> float:
        """Calculate structural stop loss with ATR buffer"""
        buffer = self.latest['atr'] * 0.5
        if signal == 'BUY':
            if hasattr(self.ta, 'demand_zones') and self.ta.demand_zones:
                nearest_zone = min(self.ta.demand_zones, key=lambda z: z['price'])
                return nearest_zone['price'] - buffer
            elif not np.isnan(self.ta.market_structure['last_swing_low']):
                return self.ta.market_structure['last_swing_low'] - buffer
            else:
                return self.latest['low'] - buffer
        else:  # SELL
            if hasattr(self.ta, 'supply_zones') and self.ta.supply_zones:
                nearest_zone = max(self.ta.supply_zones, key=lambda z: z['price'])
                return nearest_zone['price'] + buffer
            elif not np.isnan(self.ta.market_structure['last_swing_high']):
                return self.ta.market_structure['last_swing_high'] + buffer
            else:
                return self.latest['high'] + buffer
    
    def _format_trade_idea(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        """Format trade idea for final output"""
        return {
            'signal': setup['signal'],
            'confidence_score': min(100, max(0, setup['score'])),
            'entry_price': setup['entry_price'],
            'stop_loss': setup['stop_loss'],
            'targets': setup['targets'],
            'reasoning_log': setup['reasoning_log'],
            'summary': f"{setup['signal']} signal with {min(100, max(0, setup['score']))}% confidence. " +
                       f"Entry: {setup['entry_price']:.2f}, Stop: {setup['stop_loss']:.2f}"
        }
    