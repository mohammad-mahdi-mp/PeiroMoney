import ccxt
# config.py
class Config:
    DEEPSEEK_API_KEY = "sk-5b8d8f710d134a3bac674c3b83995fdf"
    ALPHA_VANTAGE_API_KEY = "G53IOI85VWVD2819"
    DEEPSEEK_API_URL = "https://api.deepseek.ai/v1/analyze"
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MA_PERIODS = [9, 20, 50, 200]
    ADX_WINDOW = 14
    ATR_WINDOW = 14
    STOCH_WINDOW = 14
    STOCH_SMOOTH = 3
    ICHIMOKU_PERIODS = (9, 26, 52)
    SUPERTREND_PERIOD = 10
    SUPERTREND_MULTIPLIER = 3.0
    RISK_PERCENTAGE = 1.0
    MIN_CONFIDENCE_SCORE = 25
    MIN_DATA_POINTS = 100
    EXCHANGE_MAP = {
        'kucoin': 'ccxt.kucoin',
        'bybit': 'ccxt.bybit',
        'okx': 'ccxt.okx',
        'binance': 'ccxt.binance'
    }
    TIMEFRAME_DAYS_MAP = {
        '1m': 1, '5m': 2, '15m': 3, '30m': 5,
        '1h': 7, '4h': 14, '1d': 90, '1w': 180
    }