# core/system.py
import asyncio
import numpy as np
from datetime import datetime
from data.fetcher import DataFetcher
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.fundamental_analyzer import FundamentalAnalyzer
from integrations.deepseek_ai import DeepSeekAI
from ml.forecaster import PriceForecaster
from trading.engine import IntelligentTradingEngine
from trading.risk_manager import RiskManager
from config import Config
from utils import setup_logger

logger = setup_logger()

class CryptoAnalysisSystem:
    def __init__(self, symbol: str, timeframe: str, exchange: str = "kucoin", total_capital: float = 10000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = exchange
        self.total_capital = total_capital
        self.data_fetcher = DataFetcher(symbol, timeframe, exchange)
        self.deepseek_ai = DeepSeekAI()
        self.forecaster = PriceForecaster(model_path="lstm_model.h5")
        
    def run_live_analysis(self):
        """Run complete live analysis workflow"""
        # Fetch data
        df = self.data_fetcher.get_candles()
        if df.empty:
            logger.error("No data retrieved for analysis")
            return None, None
            
        # Technical analysis
        tech_analyzer = TechnicalAnalyzer(df)
        
        # Fundamental analysis (async)
        fundamental_analyzer = FundamentalAnalyzer(self.symbol, self.exchange)
        whale_data = asyncio.run(fundamental_analyzer.whale_volume_analysis())
        
        # AI analysis
        last_row = tech_analyzer.df.iloc[-1]
        tech_data = {
            'rsi': last_row.get('rsi', 50),
            'macd_diff': last_row.get('macd_diff', 0),
            'ma_50': last_row.get('ma_50', 0),
            'ma_200': last_row.get('ma_200', 0),
            'adx': last_row.get('adx', 25)
        }
        ai_recommendation = self.deepseek_ai.analyze(tech_data)
        
        # Price forecasting
        try:
            prices = tech_analyzer.df['close'].values
            forecast = self.forecaster.forecast(prices)
        except Exception as e:
            logger.error(f"Forecasting error: {str(e)}")
            forecast = []
        
        # Generate trade signal using IntelligentTradingEngine
        trading_engine = IntelligentTradingEngine(tech_analyzer)
        trade_idea = trading_engine.generate_trade_idea()
        
        # Get volatility index
        volatility_index = tech_analyzer.df['atr'].iloc[-1] / tech_analyzer.df['close'].iloc[-1] * 100
        
        # Calculate position size if trade idea exists
        position_size = 0
        if trade_idea:
            risk_manager = RiskManager(self.total_capital)
            position_size = risk_manager.calculate_position_size(
                trade_idea['entry_price'],
                trade_idea['stop_loss'],
                volatility_index
            )
        
        # Prepare report
        report = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'exchange': self.exchange,
            'timestamp': datetime.utcnow().isoformat(),
            'price': tech_analyzer.df['close'].iloc[-1],
            'ai_recommendation': ai_recommendation,
            'whale_activity': whale_data,
            'price_forecast': forecast,
            'market_structure': getattr(tech_analyzer, 'market_structure', {}),
            'trade_idea': trade_idea,
            'position_size': position_size,
            'volatility_index': volatility_index
        }
        
        return report, tech_analyzer
