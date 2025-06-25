import pandas as pd
import numpy as np
from scipy.stats import zscore
import ta
from scipy.signal import find_peaks
from config import Config
from utils import setup_logger

logger = setup_logger()

class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame, timeframes: dict = None):
        if df.empty:
            raise ValueError("Empty dataframe provided for technical analysis")
        if len(df) < max(Config.MA_PERIODS):
            logger.warning(f"Insufficient data for technical analysis. Got {len(df)} points")
        
        self.df = df.copy()
        self.timeframes = timeframes.copy() if timeframes else {}
        self.trendlines = []
        self._initialize_attributes()
        
        self.calculate_all_indicators()
        self.detect_candlestick_patterns()
        self.calculate_volume_profile()
        self.identify_swing_points()
        self.identify_supply_demand_zones()
        self.identify_liquidity_pools()
        self.detect_market_structure()
        self.auto_draw_trendlines()

    def _initialize_attributes(self):
        """Initialize all analysis result attributes"""
        self.supply_zones = []
        self.demand_zones = []
        self.liquidity_pools = {'highs': [], 'lows': []}
        self.significant_levels = []
        self.market_structure = {}
        self.pivot_point = None
        self.r3 = None
        self.s3 = None
        self.price_bridges = []
        self.liquidation_clusters = []
        self.fib_levels = {}
#.
    def calculate_pivot_points(self):
        """Calculate weekly/monthly pivot points"""
        # Weekly pivots
        weekly_df = self.df.resample('W', on='timestamp').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        weekly_df['pivot'] = (weekly_df['high'] + weekly_df['low'] + weekly_df['close']) / 3
        weekly_df['r3'] = weekly_df['pivot'] + (weekly_df['high'] - weekly_df['low']) * 1.5
        weekly_df['s3'] = weekly_df['pivot'] - (weekly_df['high'] - weekly_df['low']) * 1.5
        
        # Monthly pivots
        monthly_df = self.df.resample('M', on='timestamp').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        monthly_df['pivot_monthly'] = (monthly_df['high'] + monthly_df['low'] + monthly_df['close']) / 3
        monthly_df['r3_monthly'] = monthly_df['pivot_monthly'] + (monthly_df['high'] - monthly_df['low']) * 1.5
        monthly_df['s3_monthly'] = monthly_df['pivot_monthly'] - (monthly_df['high'] - monthly_df['low']) * 1.5

        # Merge with main dataframe
        weekly_df = weekly_df.reset_index()
        monthly_df = monthly_df.reset_index()
        
        self.df = self.df.merge(
            weekly_df[['timestamp', 'pivot', 'r3', 's3']], 
            left_on=self.df['timestamp'].dt.to_period('W').dt.start_time,
            right_on='timestamp',
            how='left',
            suffixes=('', '_weekly')
        ).merge(
            monthly_df[['timestamp', 'pivot_monthly', 'r3_monthly', 's3_monthly']],
            left_on=self.df['timestamp'].dt.to_period('M').dt.start_time,
            right_on='timestamp',
            how='left',
            suffixes=('', '_monthly')
        ).ffill().drop(columns=['timestamp_weekly', 'timestamp_monthly'], errors='ignore')
#.
    def identify_price_bridges(self):
        """Identify price bridges with confluence scoring"""
        # Collect all relevant levels
        supply_prices = [z['price'] for z in self.supply_zones]
        demand_prices = [z['price'] for z in self.demand_zones]
        volume_levels = self.significant_levels
        fib_levels = list(self.fib_levels.values()) if self.fib_levels else []
        
        # Create price clusters
        all_levels = np.array(supply_prices + demand_prices + volume_levels + fib_levels)
        if len(all_levels) < 2:
            self.price_bridges = []
            return
            
        z_scores = np.abs(zscore(all_levels))
        filtered_levels = all_levels[z_scores < 2]  # Remove outliers
        
        # Cluster analysis
        clusters = []
        atr = self.df['atr'].mean() or 0.01
        cluster_threshold = atr * 0.5
        
        for level in filtered_levels:
            found = False
            for cluster in clusters:
                if abs(level - cluster['mean']) < cluster_threshold:
                    cluster['values'].append(level)
                    cluster['mean'] = np.mean(cluster['values'])
                    found = True
                    break
            if not found:
                clusters.append({'values': [level], 'mean': level})
        
        # Score clusters
        self.price_bridges = []
        for cluster in clusters:
            cluster_values = np.array(cluster['values'])
            score = 0
            
            # Confluence checks
            supply_demand = any(np.isin(cluster_values, supply_prices + demand_prices))
            fib_confluence = any(np.isin(cluster_values, fib_levels)) if fib_levels else False
            volume_confluence = any(np.isin(cluster_values, volume_levels)) if volume_levels else False
            
            # Scoring
            score += 3 if supply_demand else 0
            score += 3 if fib_confluence else 0
            score += 2 if volume_confluence else 0
            
            # Determine bridge type
            bridge_type = 'bullish' if any(np.isin(cluster_values, demand_prices)) else 'bearish'
            
            self.price_bridges.append({
                'price_level': cluster['mean'],
                'type': bridge_type,
                'strength_score': min(score, 10),
                'supply_demand': supply_demand,
                'fib_confluence': fib_confluence,
                'volume_profile': volume_confluence
            })
#.
    def detect_liquidation_clusters(self, threshold=0.95):
        """Detect liquidation clusters using Bybit data"""
        if 'long_liquidations' not in self.df.columns:
            return []
        
        # Calculate liquidation density
        price_bins = pd.cut(self.df['close'], bins=50)
        liquidation_density = self.df.groupby(price_bins)[['long_liquidations', 'short_liquidations']].sum()
        liquidation_density['total'] = liquidation_density.sum(axis=1)
        liquidation_density['zscore'] = zscore(liquidation_density['total'])
        
        # Identify clusters above threshold
        threshold_value = liquidation_density['total'].quantile(threshold)
        clusters = liquidation_density[liquidation_density['total'] >= threshold_value]
        
        self.liquidation_clusters = [{
            'price_level': row.name.mid,
            'liquidation_score': row['total']
        } for idx, row in clusters.iterrows()]
#.
    def calculate_fibonacci_bridge(self):
        """Calculate Fibonacci retracement levels"""
        # Get last swing points
        last_high = self.df['swing_high'].last_valid_index()
        last_low = self.df['swing_low'].last_valid_index()
        
        if last_high is None or last_low is None:
            return
        
        high_price = self.df.loc[last_high, 'swing_high']
        low_price = self.df.loc[last_low, 'swing_low']
        price_range = high_price - low_price
        
        self.fib_levels = {
            '23.6%': low_price + price_range * 0.236,
            '38.2%': low_price + price_range * 0.382,
            '50%': low_price + price_range * 0.5,
            '61.8%': low_price + price_range * 0.618,
            '78.6%': low_price + price_range * 0.786
        }
        
        # Merge with nearest price bridges
        for level in self.fib_levels.values():
            closest_bridge = min(self.price_bridges, 
                               key=lambda x: abs(x['price_level'] - level),
                               default=None)
            if closest_bridge and abs(closest_bridge['price_level'] - level) < self.df['atr'].mean():
                closest_bridge['fib_confluence'] = True
#.
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        # Moving Averages
        for period in Config.MA_PERIODS:
            self.df[f'ma_{period}'] = ta.trend.sma_indicator(self.df['close'], window=period)
        
        # RSI
        self.df['rsi'] = ta.momentum.rsi(self.df['close'], window=Config.RSI_WINDOW)
        
        # MACD
        macd = ta.trend.MACD(
            self.df['close'], 
            window_slow=Config.MACD_SLOW,
            window_fast=Config.MACD_FAST,
            window_sign=Config.MACD_SIGNAL
        )
        self.df['macd'] = macd.macd()
        self.df['signal'] = macd.macd_signal()
        self.df['macd_diff'] = self.df['macd'] - self.df['signal']
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(self.df['close'])
        self.df['bb_upper'] = bollinger.bollinger_hband()
        self.df['bb_middle'] = bollinger.bollinger_mavg()
        self.df['bb_lower'] = bollinger.bollinger_lband()
        
        # ADX
        self.df['adx'] = ta.trend.adx(
            self.df['high'], self.df['low'], self.df['close'], 
            window=Config.ADX_WINDOW
        )
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=Config.STOCH_WINDOW,
            smooth_window=Config.STOCH_SMOOTH
        )
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
        # ATR
        self.df['atr'] = ta.volatility.average_true_range(
            self.df['high'], self.df['low'], self.df['close'], 
            window=Config.ATR_WINDOW
        )
        
        # Volume Profile
        self.calculate_volume_profile()
        
        # OBV
        self.df['obv'] = ta.volume.on_balance_volume(self.df['close'], self.df['volume'])
        
        # Ichimoku Cloud
        self._add_ichimoku()
        
        # Supertrend
        self._add_supertrend()
        
        # VWAP
        self._add_vwap()
        
        logger.info("Calculated all technical indicators")
#.
    def _add_ichimoku(self):
        """Add Ichimoku Cloud indicators"""
        if len(self.df) < max(Config.ICHIMOKU_PERIODS):
            return
            
        tenkan_period, kijun_period, senkou_period = Config.ICHIMOKU_PERIODS
        
        # Tenkan-sen (Conversion Line)
        high_9 = self.df['high'].rolling(tenkan_period, min_periods=1).max()
        low_9 = self.df['low'].rolling(tenkan_period, min_periods=1).min()
        self.df['tenkan'] = (high_9 + low_9) / 2

        # Kijun-sen (Base Line)
        high_26 = self.df['high'].rolling(kijun_period, min_periods=1).max()
        low_26 = self.df['low'].rolling(kijun_period, min_periods=1).min()
        self.df['kijun'] = (high_26 + low_26) / 2

        # Senkou Span A (Leading Span A)
        self.df['senkou_a'] = ((self.df['tenkan'] + self.df['kijun']) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        high_52 = self.df['high'].rolling(senkou_period, min_periods=1).max()
        low_52 = self.df['low'].rolling(senkou_period, min_periods=1).min()
        self.df['senkou_b'] = ((high_52 + low_52) / 2).shift(kijun_period)

        # Chikou Span (Lagging Span)
        self.df['chikou'] = self.df['close'].shift(-kijun_period)
        
        # Add cloud status
        self.df['cloud_green'] = self.df['senkou_a'] > self.df['senkou_b']
        self.df['price_above_cloud'] = self.df['close'] > self.df[['senkou_a', 'senkou_b']].max(axis=1)
#.    
    def _add_supertrend(self):
        """Vectorized Supertrend implementation"""
        if len(self.df) < Config.SUPERTREND_PERIOD:
            return
            
        atr = ta.volatility.AverageTrueRange(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=Config.SUPERTREND_PERIOD
        ).average_true_range()
        
        hl2 = (self.df['high'] + self.df['low']) / 2
        upper_band = hl2 + (Config.SUPERTREND_MULTIPLIER * atr)
        lower_band = hl2 - (Config.SUPERTREND_MULTIPLIER * atr)
        
        # Vectorized calculation
        supertrend = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        direction = pd.Series(np.ones(len(self.df)), index=self.df.index)
        
        # Initialize first value
        supertrend.iloc[0] = upper_band.iloc[0]
        
        for i in range(1, len(self.df)):
            close_prev = self.df['close'].iloc[i-1]
            supertrend_prev = supertrend.iloc[i-1]
            
            if self.df['close'].iloc[i] > supertrend_prev:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
        
        self.df['supertrend'] = supertrend
        self.df['supertrend_direction'] = direction
#.
    def _add_vwap(self):
        """Add Volume Weighted Average Price"""
        # Only for intraday timeframes
        if self.df['timestamp'].dt.floor('d').nunique() > 1:  # Check if intraday
            vwap = (self.df['volume'] * (self.df['high'] + self.df['low'] + self.df['close']) / 3).cumsum() / self.df['volume'].cumsum()
            self.df['vwap'] = vwap
#.
    def calculate_volume_profile(self):
        """Robust volume profile calculation"""
        try:
            if len(self.df) < 5:  # Insufficient data
                self.significant_levels = []
                return
                
            # Use price range for binning
            min_price = self.df['low'].min()
            max_price = self.df['high'].max()
            price_range = max_price - min_price
            
            if price_range == 0:  # Handle flat market
                self.significant_levels = [self.df['close'].mean()]
                return
                
            # Create bins based on price range
            num_bins = min(20, len(self.df) // 5)  # Adaptive bin count
            bins = np.linspace(min_price, max_price, num_bins + 1)
            
            # Assign each candle to bins based on high-low range
            vol_profile = pd.Series(0, index=(bins[:-1] + bins[1:]) / 2)
            
            for _, row in self.df.iterrows():
                # Find bins within candle's high-low range
                mask = (bins[:-1] >= row['low']) & (bins[1:] <= row['high'])
                valid_bins = bins[:-1][mask]
                
                if len(valid_bins) > 0:
                    # Distribute volume equally across bins
                    vol_per_bin = row['volume'] / len(valid_bins)
                    vol_profile.loc[valid_bins] += vol_per_bin
            
            # Find significant levels (top 3 volume bins)
            significant_bins = vol_profile.nlargest(3).index
            self.significant_levels = significant_bins.tolist()
            
            logger.debug(f"Volume profile calculated with levels: {self.significant_levels}")
        except Exception as e:
            logger.error(f"Volume profile calculation error: {str(e)}")
            self.significant_levels = []
#.
    def identify_swing_points(self, lookback: int = 5):
        """Efficient swing point identification"""
        if len(self.df) < lookback * 2:
            return
            
        # Initialize columns
        self.df['swing_high'] = np.nan
        self.df['swing_low'] = np.nan
        
        # Find swing highs
        high_peaks, _ = find_peaks(
            self.df['high'].values, 
            distance=lookback,
            prominence=(self.df['atr'].mean() or 0.01)
        )
        
        # Find swing lows
        low_peaks, _ = find_peaks(
            -self.df['low'].values, 
            distance=lookback,
            prominence=(self.df['atr'].mean() or 0.01)
        )
        
        # Update DataFrame in vectorized manner
        self.df.loc[self.df.index[high_peaks], 'swing_high'] = self.df.loc[self.df.index[high_peaks], 'high']
        self.df.loc[self.df.index[low_peaks], 'swing_low'] = self.df.loc[self.df.index[low_peaks], 'low']
#.
    def identify_supply_demand_zones(self, consolidation_threshold: float = 0.02):
        """Optimized supply/demand zone identification"""
        # Get swing points indices
        swing_high_idxs = self.df[self.df['swing_high'].notna()].index
        swing_low_idxs = self.df[self.df['swing_low'].notna()].index
        
        # Process swing highs (supply zones)
        supply_zones = []
        for idx in swing_high_idxs:
            price = self.df.at[idx, 'swing_high']
            
            # Check consolidation window
            start_idx = max(0, idx - 5)
            end_idx = min(len(self.df) - 1, idx + 5)
            window = self.df.iloc[start_idx:end_idx + 1]
            
            if len(window) < 3:
                continue
                
            price_range = window['high'].max() - window['low'].min()
            if price_range / price < consolidation_threshold:
                strength_score = self._calculate_zone_strength(price, idx)
                supply_zones.append({
                    'price': price,
                    'start': window.index[0],
                    'end': window.index[-1],
                    'strength_score': strength_score,
                    'test_count': 0
                })
        
        # Process swing lows (demand zones)
        demand_zones = []
        for idx in swing_low_idxs:
            price = self.df.at[idx, 'swing_low']
            
            # Check consolidation window
            start_idx = max(0, idx - 5)
            end_idx = min(len(self.df) - 1, idx + 5)
            window = self.df.iloc[start_idx:end_idx + 1]
            
            if len(window) < 3:
                continue
                
            price_range = window['high'].max() - window['low'].min()
            if price_range / price < consolidation_threshold:
                strength_score = self._calculate_zone_strength(price, idx)
                demand_zones.append({
                    'price': price,
                    'start': window.index[0],
                    'end': window.index[-1],
                    'strength_score': strength_score,
                    'test_count': 0
                })
        
        # Update test counts
        self._update_zone_test_counts(supply_zones, demand_zones)
        
        self.supply_zones = supply_zones
        self.demand_zones = demand_zones
#.
    def detect_candlestick_patterns(self):
        """Fixed candlestick pattern detection"""
        df = self.df
        
        # Hammer
        body_size = (df['open'] - df['close']).abs()
        candle_range = df['high'] - df['low']
        lower_wick = df['close'] - df['low']
        upper_wick = df['high'] - df['close']
        
        df['hammer'] = (
            (candle_range > 3 * body_size) &
            (lower_wick / candle_range > 0.6) &
            (upper_wick / candle_range < 0.2)
        )
        # Shooting Star
        df['shooting_star'] = (
            (candle_range > 3 * body_size) &
            (upper_wick / candle_range > 0.6) &
            (lower_wick / candle_range < 0.2)
        )
        # Bullish Engulfing
        prev_bearish = df['close'].shift(1) < df['open'].shift(1)
        current_bullish = df['close'] > df['open']
        engulf_condition =  (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
        df['bullish_engulfing'] = prev_bearish & current_bullish & engulf_condition
        
        # Bearish Engulfing
        prev_bullish = df['close'].shift(1) > df['open'].shift(1)
        current_bearish = df['close'] < df['open']
        engulf_condition = (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
        df['bearish_engulfing'] = prev_bullish & current_bearish & engulf_condition
        
        # Morning Star
        prev_red = df['close'].shift(2) < df['open'].shift(2)
        small_body = (df['close'].shift(1) - df['open'].shift(1)).abs() < 0.1 * candle_range.shift(1)
        green_candle = df['close'] > df['open']
        above_first_open = df['close'] > df['open'].shift(2)
        df['morning_star'] = prev_red & small_body & green_candle & above_first_open
        
        # Evening Star
        prev_green = df['close'].shift(2) > df['open'].shift(2)
        red_candle = df['close'] < df['open']
        below_first_open = df['close'] < df['open'].shift(2)
        df['evening_star'] = prev_green & small_body & red_candle & below_first_open
#.
    def auto_draw_trendlines(self, order=5):
        """Efficient trendline identification"""
        try:
            # Find significant peaks and troughs
            min_prominence = (self.df['atr'].mean() or 0.01) / 2
            
            peak_indices, _ = find_peaks(
                self.df['high'].values, 
                distance=order, 
                prominence=min_prominence
            )
            
            trough_indices, _ = find_peaks(
                -self.df['low'].values, 
                distance=order, 
                prominence=min_prominence
            )
            
            # Create resistance lines
            if len(peak_indices) >= 2:
                for i in range(1, len(peak_indices)):
                    idx1, idx2 = peak_indices[i-1], peak_indices[i]
                    self.trendlines.append({
                        'type': 'resistance',
                        'start_index': self.df.index[idx1],
                        'start_price': self.df['high'].iloc[idx1],
                        'end_index': self.df.index[idx2],
                        'end_price': self.df['high'].iloc[idx2]
                    })
            
            # Create support lines
            if len(trough_indices) >= 2:
                for i in range(1, len(trough_indices)):
                    idx1, idx2 = trough_indices[i-1], trough_indices[i]
                    self.trendlines.append({
                        'type': 'support',
                        'start_index': self.df.index[idx1],
                        'start_price': self.df['low'].iloc[idx1],
                        'end_index': self.df.index[idx2],
                        'end_price': self.df['low'].iloc[idx2]
                    })
                
        except Exception as e:
            logger.error(f"Error in auto_draw_trendlines: {e}")
            self.trendlines = []
#.
    def _calculate_zone_strength(self, price: float, zone_idx: int) -> int:
        """Calculate strength score (1-10) for a supply/demand zone"""
        score = 5  # Base score
        
        # 1. Velocity factor - how aggressively price left the zone
        if zone_idx + 1 < len(self.df):
            exit_candle = self.df.iloc[zone_idx + 1]
            body_size = abs(exit_candle['close'] - exit_candle['open'])
            candle_range = exit_candle['high'] - exit_candle['low']
            
            if candle_range > 0:
                velocity_factor = body_size / candle_range
                if velocity_factor > 0.7:
                    score += 3  # Strong exit
                elif velocity_factor > 0.4:
                    score += 1  # Moderate exit
                else:
                    score -= 1  # Weak exit
        
        # 2. MA confluence
        ma_200 = self.df['ma_200'].iloc[zone_idx] if 'ma_200' in self.df else None
        if ma_200 and abs(price - ma_200) / price < 0.005:  # Within 0.5%
            score += 2
        
        # 3. Volume confluence
        if 'volume' in self.df:
            zone_volume = self.df['volume'].iloc[zone_idx]
            avg_volume = self.df['volume'].rolling(20).mean().iloc[zone_idx]
            if zone_volume > avg_volume * 1.5:
                score += 1
        
        # Constrain score between 1-10
        return max(1, min(10, score))
#.
    def _update_zone_test_counts(self, supply_zones: list, demand_zones: list):
        """Update the test count for each zone based on price revisits"""
        all_zones = supply_zones + demand_zones
        if not all_zones:
            return
            
        # Sort zones by price
        all_zones.sort(key=lambda z: z['price'])
        
        # Create price levels array
        zone_prices = [z['price'] for z in all_zones]
        
        # Check each candle to see if it tests a zone
        for idx, row in self.df.iterrows():
            for zone in all_zones:
                # Skip if this candle is within the zone formation period
                if zone['start'] <= idx <= zone['end']:
                    continue
                    
                # Check if price tested the zone
                if row['low'] <= zone['price'] <= row['high']:
                    zone['test_count'] += 1
        
        # Adjust strength score based on test count
        for zone in all_zones:
            # Freshness: fewer tests = higher score
            if zone['test_count'] == 0:
                zone['strength_score'] = min(10, zone['strength_score'] + 3)
            elif zone['test_count'] == 1:
                zone['strength_score'] = min(10, zone['strength_score'] + 1)
            elif zone['test_count'] >= 3:
                zone['strength_score'] = max(1, zone['strength_score'] - 2)
#.
    def identify_liquidity_pools(self):
        """Identify liquidity pools (untapped swing highs/lows)"""
        self.liquidity_pools = {
            'highs': [],
            'lows': []
        }
        
        # Find recent swing highs that haven't been broken
        swing_highs = self.df.dropna(subset=['swing_high'])
        if len(swing_highs) > 0:
            last_high = swing_highs.iloc[-1]
            current_price = self.df['close'].iloc[-1]
            
            if current_price < last_high['swing_high']:
                # Check if there's a cluster of swing highs at this level
                similar_highs = swing_highs[
                    (swing_highs['swing_high'] > last_high['swing_high'] * 0.99) &
                    (swing_highs['swing_high'] < last_high['swing_high'] * 1.01)
                ]
                
                self.liquidity_pools['highs'].append({
                    'price': last_high['swing_high'],
                    'count': len(similar_highs),
                    'timestamp': last_high['timestamp']
                })
        
        # Find recent swing lows that haven't been broken
        swing_lows = self.df.dropna(subset=['swing_low'])
        if len(swing_lows) > 0:
            last_low = swing_lows.iloc[-1]
            current_price = self.df['close'].iloc[-1]
            
            if current_price > last_low['swing_low']:
                # Check if there's a cluster of swing lows at this level
                similar_lows = swing_lows[
                    (swing_lows['swing_low'] > last_low['swing_low'] * 0.99) &
                    (swing_lows['swing_low'] < last_low['swing_low'] * 1.01)
                ]
                
                self.liquidity_pools['lows'].append({
                    'price': last_low['swing_low'],
                    'count': len(similar_lows),
                    'timestamp': last_low['timestamp']
                })
        
        # Find equal highs and equal lows
        self._find_equal_highs_lows()    
#.
    def _find_equal_highs_lows(self):
        """Find clusters of equal highs and equal lows"""
        # Equal highs
        high_clusters = []
        swing_highs = self.df.dropna(subset=['swing_high'])
        if len(swing_highs) > 1:
            # Group similar swing highs
            swing_highs = swing_highs.sort_values('swing_high')
            current_cluster = []
            
            for idx, row in swing_highs.iterrows():
                if not current_cluster:
                    current_cluster.append(row)
                else:
                    last_price = current_cluster[-1]['swing_high']
                    if abs(row['swing_high'] - last_price) / last_price < 0.005:  # Within 0.5%
                        current_cluster.append(row)
                    else:
                        if len(current_cluster) > 1:
                            avg_price = sum(h['swing_high'] for h in current_cluster) / len(current_cluster)
                            high_clusters.append({
                                'price': avg_price,
                                'count': len(current_cluster),
                                'timestamps': [h['timestamp'] for h in current_cluster]
                            })
                        current_cluster = [row]
            
            # Add last cluster
            if len(current_cluster) > 1:
                avg_price = sum(h['swing_high'] for h in current_cluster) / len(current_cluster)
                high_clusters.append({
                    'price': avg_price,
                    'count': len(current_cluster),
                    'timestamps': [h['timestamp'] for h in current_cluster]
                })
        
        # Add to liquidity pools if not already present
        for cluster in high_clusters:
            existing = any(abs(p['price'] - cluster['price']) / cluster['price'] < 0.005 
                          for p in self.liquidity_pools['highs'])
            if not existing:
                self.liquidity_pools['highs'].append({
                    'price': cluster['price'],
                    'count': cluster['count'],
                    'timestamp': cluster['timestamps'][-1]  # Most recent
                })
        
        # Equal lows (similar approach)
        low_clusters = []
        swing_lows = self.df.dropna(subset=['swing_low'])
        if len(swing_lows) > 1:
            swing_lows = swing_lows.sort_values('swing_low')
            current_cluster = []
            
            for idx, row in swing_lows.iterrows():
                if not current_cluster:
                    current_cluster.append(row)
                else:
                    last_price = current_cluster[-1]['swing_low']
                    if abs(row['swing_low'] - last_price) / last_price < 0.005:  # Within 0.5%
                        current_cluster.append(row)
                    else:
                        if len(current_cluster) > 1:
                            avg_price = sum(l['swing_low'] for l in current_cluster) / len(current_cluster)
                            low_clusters.append({
                                'price': avg_price,
                                'count': len(current_cluster),
                                'timestamps': [l['timestamp'] for l in current_cluster]
                            })
                        current_cluster = [row]
            
            if len(current_cluster) > 1:
                avg_price = sum(l['swing_low'] for l in current_cluster) / len(current_cluster)
                low_clusters.append({
                    'price': avg_price,
                    'count': len(current_cluster),
                    'timestamps': [l['timestamp'] for l in current_cluster]
                })        
        for cluster in low_clusters:
            existing = any(abs(p['price'] - cluster['price']) / cluster['price'] < 0.005 
                          for p in self.liquidity_pools['lows'])
            if not existing:
                self.liquidity_pools['lows'].append({
                    'price': cluster['price'],
                    'count': cluster['count'],
                    'timestamp': cluster['timestamps'][-1]
                })
#.
    def detect_market_structure(self):
        """Detect market structure with sequence analysis"""
        if 'swing_high' not in self.df or 'swing_low' not in self.df:
            return
            
        # Get all swing points with their types and prices
        swing_points = []
        for idx, row in self.df.iterrows():
            if not np.isnan(row['swing_high']):
                swing_points.append(('high', row['swing_high'], idx))
            if not np.isnan(row['swing_low']):
                swing_points.append(('low', row['swing_low'], idx))
        
        # Sort by index (time)
        swing_points.sort(key=lambda x: x[2])
        
        # Keep only the last 5 swings for analysis
        recent_swings = swing_points[-5:]
        
        # Analyze the sequence
        structure_sequence = []
        for i in range(1, len(recent_swings)):
            prev_type, prev_price, _ = recent_swings[i-1]
            curr_type, curr_price, _ = recent_swings[i]
            
            if prev_type == 'high' and curr_type == 'high':
                structure_sequence.append('HH' if curr_price > prev_price else 'LH')
            elif prev_type == 'low' and curr_type == 'low':
                structure_sequence.append('HL' if curr_price > prev_price else 'LL')
            elif prev_type == 'high' and curr_type == 'low':
                structure_sequence.append('HL' if curr_price > prev_price else 'LL')
            elif prev_type == 'low' and curr_type == 'high':
                structure_sequence.append('HH' if curr_price > prev_price else 'LH')
        
        # Determine overall trend based on sequence
        uptrend_count = sum(1 for s in structure_sequence if s in ['HH', 'HL'])
        downtrend_count = sum(1 for s in structure_sequence if s in ['LH', 'LL'])
        
        if uptrend_count > downtrend_count:
            trend_strength = uptrend_count / len(structure_sequence)
            market_structure = "Uptrend"
        elif downtrend_count > uptrend_count:
            trend_strength = downtrend_count / len(structure_sequence)
            market_structure = "Downtrend"
        else:
            trend_strength = 0.5
            market_structure = "Ranging"
        
        # Get last swing points
        last_high_idx = self.df['swing_high'].last_valid_index()
        last_low_idx = self.df['swing_low'].last_valid_index()
        
        if last_high_idx is None or last_low_idx is None:
            return
            
        last_high = self.df.at[last_high_idx, 'swing_high']
        last_low = self.df.at[last_low_idx, 'swing_low']
        
        # Check for Break of Structure (BOS)
        bos = False
        if last_high_idx > last_low_idx:  # Uptrend
            # Break above previous high
            if self.df['close'].iloc[-1] > last_high:
                bos = True
        else:  # Downtrend
            # Break below previous low
            if self.df['close'].iloc[-1] < last_low:
                bos = True
                
        # Check for Change of Character (ChoCH)
        choch = False
        if last_high_idx > last_low_idx:  # Uptrend
            # Lower low formed
            if self.df['low'].iloc[-1] < last_low:
                choch = True
        else:  # Downtrend
            # Higher high formed
            if self.df['high'].iloc[-1] > last_high:
                choch = True
                
        self.market_structure = {
            'sequence': structure_sequence,
            'trend': market_structure,
            'trend_strength': trend_strength,
            'bos': bos,
            'choch': choch,
            'last_swing_high': last_high,
            'last_swing_low': last_low
        }
        
        # Add to DataFrame for strategy use
        self.df.loc[:, 'bos'] = False
        self.df.loc[:, 'choch'] = False
        if bos:
            self.df.loc[self.df.index[-1], 'bos'] = True
        if choch:
            self.df.loc[self.df.index[-1], 'choch'] = True
#.
