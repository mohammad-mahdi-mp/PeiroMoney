# analysis/technical_analyzer.py
import pandas as pd
import numpy as np
import ta
from scipy.signal import find_peaks
from config import Config
from utils import setup_logger

logger = setup_logger()

class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Empty dataframe provided for technical analysis")
            
        if len(df) < max(Config.MA_PERIODS):
            logger.warning(f"Insufficient data for technical analysis. Got {len(df)} points")
            
        self.df = df.copy()
        self.calculate_all_indicators()
        self.detect_candlestick_patterns()
        self.calculate_volume_profile()
        self.identify_swing_points()
        self.identify_supply_demand_zones()
        self.identify_liquidity_pools()
        self.detect_market_structure()
        self.auto_draw_trendlines()
        self.trendlines = []
    
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
    
    def _add_supertrend(self):
        """Add Supertrend indicator"""
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
        
        supertrend = np.zeros(len(self.df))
        direction = np.zeros(len(self.df))
        
        supertrend[0] = upper_band.iloc[0] if not upper_band.empty else 0
        direction[0] = 1
        
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > supertrend[i-1]:
                supertrend[i] = lower_band.iloc[i]
                direction[i] = -1
            else:
                supertrend[i] = upper_band.iloc[i]
                direction[i] = 1
        
        self.df['supertrend'] = supertrend
        self.df['supertrend_direction'] = direction
        
    def _add_vwap(self):
        """Add Volume Weighted Average Price"""
        # Only for intraday timeframes
        if self.df['timestamp'].dt.floor('d').nunique() > 1:  # Check if intraday
            vwap = (self.df['volume'] * (self.df['high'] + self.df['low'] + self.df['close']) / 3).cumsum() / self.df['volume'].cumsum()
            self.df['vwap'] = vwap
            
    def calculate_volume_profile(self):
        """Calculate volume profile for significant price levels"""
        try:
            # Create price bins
            price_range = self.df['high'].max() - self.df['low'].min()
            bin_size = price_range / 20  # 20 bins
            
            # Calculate volume at price levels
            self.df['price_bin'] = pd.cut(
                self.df['close'],
                bins=20,
                labels=False
            )
            
            volume_profile = self.df.groupby('price_bin')['volume'].sum()
            
            # Find significant levels
            high_volume_bins = volume_profile.nlargest(3).index
            self.significant_levels = []
            
            for bin_idx in high_volume_bins:
                bin_data = self.df[self.df['price_bin'] == bin_idx]
                price_level = bin_data['close'].mean()
                self.significant_levels.append(price_level)
            
            logger.debug(f"Volume profile calculated with levels: {self.significant_levels}")
        except Exception as e:
            logger.error(f"Volume profile calculation error: {str(e)}")
            self.significant_levels = []

    def identify_swing_points(self, lookback: int = 5):
        """Identify swing highs and lows using scipy's find_peaks"""
        if len(self.df) < lookback * 2:
            return
            
        # Find swing highs
        highs = self.df['high'].values
        high_peaks, _ = find_peaks(highs, distance=lookback)
        
        # Find swing lows
        lows = self.df['low'].values
        low_peaks, _ = find_peaks(-lows, distance=lookback)
        
        # Create new columns
        self.df['swing_high'] = np.nan
        self.df['swing_low'] = np.nan
        
        # Mark swing highs
        for idx in high_peaks:
            self.df.at[self.df.index[idx], 'swing_high'] = highs[idx]
            
        # Mark swing lows
        for idx in low_peaks:
            self.df.at[self.df.index[idx], 'swing_low'] = lows[idx]
    
    def identify_supply_demand_zones(self, consolidation_threshold: float = 0.02):
        """Identify supply and demand zones based on swing points"""
        if 'swing_high' not in self.df or 'swing_low' not in self.df:
            return
            
        # Find significant swing highs (supply zones)
        supply_zones = []
        for idx, row in self.df.iterrows():
            if not np.isnan(row['swing_high']):
                # Check if price consolidated near this high
                start_idx = max(0, idx-5)
                end_idx = min(len(self.df)-1, idx+5)
                window = self.df.iloc[start_idx:end_idx+1]
                
                if len(window) < 3:
                    continue
                    
                price_range = window['high'].max() - window['low'].min()
                
                if price_range / row['swing_high'] < consolidation_threshold:
                    # Calculate zone strength
                    strength_score = self._calculate_zone_strength(row['swing_high'], idx)
                    supply_zones.append({
                        'price': row['swing_high'],
                        'start': window.index[0],
                        'end': window.index[-1],
                        'strength_score': strength_score,
                        'test_count': 0  # Will be updated later
                    })
        
        # Find significant swing lows (demand zones)
        demand_zones = []
        for idx, row in self.df.iterrows():
            if not np.isnan(row['swing_low']):
                # Check if price consolidated near this low
                start_idx = max(0, idx-5)
                end_idx = min(len(self.df)-1, idx+5)
                window = self.df.iloc[start_idx:end_idx+1]
                
                if len(window) < 3:
                    continue
                    
                price_range = window['high'].max() - window['low'].min()
                
                if price_range / row['swing_low'] < consolidation_threshold:
                    # Calculate zone strength
                    strength_score = self._calculate_zone_strength(row['swing_low'], idx)
                    
                    demand_zones.append({
                        'price': row['swing_low'],
                        'start': window.index[0],
                        'end': window.index[-1],
                        'strength_score': strength_score,
                        'test_count': 0  # Will be updated later
                    })
        
        # Update test counts for all zones
        self._update_zone_test_counts(supply_zones, demand_zones)
        
        # Store zones
        self.supply_zones = supply_zones
        self.demand_zones = demand_zones    
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
        
    def detect_candlestick_patterns(self):
        """Detect common candlestick patterns"""
        df = self.df
        
        # Hammer: Small body, long lower wick, little/no upper wick
        df['hammer'] = (
            ((df['high'] - df['low']) > 3 * (df['open'] - df['close']).abs()) &
            ((df['close'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6) &
            ((df['high'] - df['close']) / (0.001 + df['high'] - df['low']) < 0.2)
        )

        # Shooting Star: Small body, long upper wick, little/no lower wick
        df['shooting_star'] = (
            ((df['high'] - df['low']) > 3 * (df['open'] - df['close']).abs()) &
            ((df['high'] - df['close']) / (0.001 + df['high'] - df['low']) > 0.6) &
            ((df['close'] - df['low']) / (0.001 + df['high'] - df['low']) < 0.2)
        )

        # Bullish Engulfing: Current green candle fully engulfs previous red candle
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open']) &                  # Current candle is bullish
            (df['open'] < df['close'].shift(1)) &         # Current open is lower than previous close
            (df['close'] > df['open'].shift(1))           # Current close is higher than previous open
        )
        
        # Bearish Engulfing: Current red candle fully engulfs previous green candle
        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['close'] < df['open']) &                   # Current candle is bearish
            (df['open'] > df['close'].shift(1)) &          # Current open is higher than previous close
            (df['close'] < df['open'].shift(1))            # Current close is lower than previous open
        )
        # Morning Star: Downtrend, long red, small candle, long green
        df['morning_star'] = (
            (df['close'].shift(2) < df['open'].shift(2)) &
            (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * (df['high'].shift(1) - df['low'].shift(1))) & # <-- FIX: Parenthesis added            (df['close'] > df['open']) &
            (df['close'] > df['open'].shift(2))
        )

        # Evening Star: Uptrend, long green, small candle, long red
        df['evening_star'] = (
            (df['close'].shift(2) > df['open'].shift(2)) &
            (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * (df['high'].shift(1) - df['low'].shift(1))) & # <-- FIX: Parenthesis added
            (df['close'] < df['open']) &
            (df['close'] < df['open'].shift(2))
        )

    def auto_draw_trendlines(self, order=5):
        """
        Automatically identifies and stores major trendlines based on swing highs and lows.
        A simple implementation connecting the last two major pivots.
        """
        try:
            # پیدا کردن سقف‌های کلیدی (نقاط مقاومت)
            # پارامتر order به این معنی است که یک قله باید از هر طرف با 'order' نقطه پایین‌تر احاطه شده باشد
            peak_indices, _ = find_peaks(self.df['high'], distance=order, prominence=self.df['atr'].mean() / 2)
        
            # پیدا کردن کف‌های کلیدی (نقاط حمایت) با پیدا کردن قله در سری معکوس قیمت 'low'
            trough_indices, _ = find_peaks(-self.df['low'], distance=order, prominence=self.df['atr'].mean() / 2)

            # ایجاد خط روند مقاومت با استفاده از دو سقف آخر
            if len(peak_indices) >= 2:
                last_two_peaks_indices = peak_indices[-2:]
                p1_idx = self.df.index[last_two_peaks_indices[0]]
                p2_idx = self.df.index[last_two_peaks_indices[1]]
                p1_price = self.df['high'].iloc[last_two_peaks_indices[0]]
                p2_price = self.df['high'].iloc[last_two_peaks_indices[1]]
                
                resistance_line = {
                    'type': 'resistance',
                    'start_index': p1_idx,
                    'start_price': p1_price,
                    'end_index': p2_idx,
                    'end_price': p2_price
                }
                self.trendlines.append(resistance_line)

            # ایجاد خط روند حمایت با استفاده از دو کف آخر
            if len(trough_indices) >= 2:
                last_two_troughs_indices = trough_indices[-2:]
                t1_idx = self.df.index[last_two_troughs_indices[0]]
                t2_idx = self.df.index[last_two_troughs_indices[1]]
                t1_price = self.df['low'].iloc[last_two_troughs_indices[0]]
                t2_price = self.df['low'].iloc[last_two_troughs_indices[1]]

                support_line = {
                    'type': 'support',
                    'start_index': t1_idx,
                    'start_price': t1_price,
                    'end_index': t2_idx,
                    'end_price': t2_price
                }
                self.trendlines.append(support_line)
        
            # برای دسترسی آسان‌تر، لیست خطوط روند را در یک ستون جدید در دیتافریم قرار می‌دهیم
            # این کار ممکن است حافظه زیادی مصرف کند؛ در آینده می‌توان آن را بهینه‌تر کرد
            if self.trendlines:
                # Create a column that contains the list of trendlines for each row
                self.df['trendlines'] = [self.trendlines for _ in range(len(self.df))]
            else:
                self.df['trendlines'] = [[] for _ in range(len(self.df))]

        except Exception as e:
            print(f"Error in auto_draw_trendlines: {e}")
            # در صورت خطا، یک لیست خالی ایجاد می‌کنیم تا برنامه متوقف نشود
            self.trendlines = []
            self.df['trendlines'] = [[] for _ in range(len(self.df))]
            