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
            except Exception as e:
                logger.error(f"API Error: {str(e)}")
                raise

    def get_candles(self) -> pd.DataFrame:
        try:
            return self._fetch_via_rest()
        except Exception as e:
            logger.error(f"Data fetch error: {str(e)}")
            return pd.DataFrame()  
      
    def _fetch_via_rest(self) -> pd.DataFrame:
        start_time = datetime.now() - timedelta(days=self.historical_days)
        end_time = datetime.now()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=1000
            )
            
            if not ohlcv:
                logger.warning(f"No data returned for {self.symbol} on {self.exchange_name}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = self._process_dataframe(df)
            return df
        except Exception as e:
            logger.error(f"REST API Error: {str(e)}")
            return pd.DataFrame()

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df['timestamp'] = df['timestamp'].dt.tz_localize("UTC").dt.tz_convert(self.timezone)
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['candle_type'] = np.where(df['close'] > df['open'], 'bull', 'bear')
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'range', 'body', 'candle_type']]