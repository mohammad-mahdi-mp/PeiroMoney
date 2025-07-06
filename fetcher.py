# data/fetcher.py
import os
import asyncio
import aiohttp
import ccxt
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config
from utils import setup_logger

logger = setup_logger()

class DataFetcher:
    def __init__(self, symbol: str, timeframe: str, exchange: str = "kucoin", timezone: str = "UTC"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.historical_days = Config.TIMEFRAME_DAYS_MAP.get(timeframe, 7)
        self.timezone = pytz.timezone(timezone)
        self.exchange_name = exchange.lower()
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self):
        exchange_class = getattr(ccxt, self.exchange_name)
        return exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_data_async(self, endpoint: str, params: dict = None) -> dict:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(endpoint, params=params, timeout=15) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {str(e)}")
                raise
            except asyncio.TimeoutError:
                logger.error("Request timed out")
                raise
            except Exception as e:
                logger.error(f"Unexpected API error: {str(e)}")
                raise

    # ====================
    # LIQUIDATION DATA
    # ====================
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_liquidation_data(self) -> dict:
        """Fetch liquidation data from Bybit (supports cross-exchange)"""
        endpoint = "https://api.bybit.com/v2/public/liq-records"
        params = {'symbol': self.symbol.replace('/', ''), 'limit': 200}  # Format symbol
        
        try:
            logger.info(f"Fetching liquidation data for {self.symbol}")
            return await self.fetch_data_async(endpoint, params)
        except Exception as e:
            logger.error(f"Liquidation data fetch failed: {str(e)}")
            return {'ret_code': -1, 'ret_msg': str(e), 'result': []}

    # ====================
    # ORDER FLOW DATA
    # ====================
    async def fetch_orderflow_data(self) -> dict:
        """Mock order flow data (replace with actual API implementation)"""
        try:
            logger.info("Generating mock order flow data")
            # Placeholder logic - replace with actual API integration
            return {
                'imbalance_levels': [
                    {'price': 100.0, 'strength': 0.8},
                    {'price': 102.5, 'strength': 0.6}
                ],
                'delta': np.random.uniform(-1, 1)
            }
        except Exception as e:
            logger.error(f"Order flow simulation failed: {str(e)}")
            return {'imbalance_levels': []}

    # ====================
    # MULTI-TIMEFRAME DATA
    # ====================
    def get_candles(self) -> pd.DataFrame:  # تغییر نوع بازگشتی
        """Fetch candles for THIS specific timeframe"""
        try:
            # حذف کد مربوط به multi-timeframe
            days = Config.TIMEFRAME_DAYS_MAP.get(self.timeframe, 7)
            return self._fetch_timeframe_via_rest(self.timeframe, days)
        except Exception as e:
            logger.error(f"Error fetching {self.timeframe} data: {str(e)}")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No data for {self.symbol}/{self.timeframe}")
        return pd.DataFrame()  # Return empty instead of None

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _fetch_timeframe_via_rest(self, timeframe: str, historical_days: int) -> pd.DataFrame:
        """Fetch OHLCV data with enhanced error handling"""
        start_time = datetime.now() - timedelta(days=historical_days)
        try:
            # Attempt to fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=1000
            )
            
            if not ohlcv:
                logger.warning(f"No {timeframe} data for {self.symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return self._process_dataframe(df)
            
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(f"Exchange error ({timeframe}): {str(e)}")
            # Attempt fallback to smaller timeframe
            if timeframe != '1m':
                logger.info(f"Attempting fallback to 1m for {self.symbol}")
                return self._fetch_timeframe_via_rest('1m', historical_days)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Critical error ({timeframe}): {str(e)}")
            return pd.DataFrame()

    # ====================
    # DATA PROCESSING
    # ====================
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        # Timezone conversion
        df['timestamp'] = df['timestamp'].dt.tz_localize("UTC").dt.tz_convert(self.timezone)
        
        # Basic candle metrics
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['candle_type'] = np.where(df['close'] > df['open'], 'bull', 'bear')
        
        # Strategy-specific features
        df = self._detect_fvg(df)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                 'range', 'body', 'candle_type', 'fvg']]
    
    def _detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:

        """Detect Fair Value Gaps (FVG) in price series"""
        try:
            if len(df) < 3:
                df['fvg'] = False
                return df
                
            # FVG detection logic
            condition1 = df['high'].shift(2) < df['low'].shift(1)
            condition2 = df['low'] > df['high'].shift(1)
            df['fvg'] = condition1 & condition2
            
            # Convert bool to int for better serialization
            df['fvg'] = df['fvg'].astype(int)
            
        except Exception as e:
            logger.error(f"FVG detection failed: {str(e)}")
            df['fvg'] = 0
            
        return df
    
    
class MultiTimeframeFetcher:
    def __init__(self, symbol: str, timeframes: dict, exchange: str):
        self.symbol = symbol
        self.timeframes = timeframes
        self.exchange = exchange
        
    def get_candles(self) -> dict:
        """Fetch data for multiple timeframes with prioritization"""
        results = {}
        priorities = {
            'primary': 1,  # Highest priority
            'confirm': 2,
            'entry': 3     # Lowest priority
        }
        
        # Fetch in priority order
        for tf_key in sorted(self.timeframes.keys(), key=lambda x: priorities.get(x, 10)):
            try:
                tf_value = self.timeframes[tf_key]
                fetcher = DataFetcher(self.symbol, tf_value, self.exchange)
                df = fetcher.get_candles()
                
                if df.empty:
                    logger.warning(f"Empty data for {tf_key} ({tf_value})")
                    # If primary fails, abort entire process
                    if tf_key == 'primary':
                        return {}
                else:
                    results[tf_key] = df
            except Exception as e:
                logger.error(f"Error fetching {tf_key} ({tf_value}): {str(e)}")
                if tf_key == 'primary':
                    return {}
        
        return results    