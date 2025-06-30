# core/system.py
import asyncio
import numpy as np
from datetime import datetime
from typing import Union, Dict, Any  # Added for type hints
from fetcher import DataFetcher , MultiTimeframeFetcher
from technical_analyzer import TechnicalAnalyzer
from fundamental_analyzer import FundamentalAnalyzer
from deepseek_ai import DeepSeekAI
from forecaster import PriceForecaster
from engine import IntelligentTradingEngine
from risk_manager import RiskManager
from config import Config
from utils import setup_logger

logger = setup_logger()

class CryptoAnalysisSystem:
    def __init__(
        self, 
        symbol: str, 
        timeframe: Union[str, Dict[str, str]] = None,  # Accept both string and dict
        exchange: str = "kucoin", 
        total_capital: float = 10000
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.total_capital = total_capital
        
        # Handle both string (backward compat) and dict timeframe inputs
        if isinstance(timeframe, dict):
            self.timeframes = timeframe
        else:
            self.timeframes = {
                'primary': timeframe or '4h',
                'confirm': '1d',
                'entry': '15m'
            }
        
        # Initialize data fetchers for each timeframe
        self.data_fetcher = MultiTimeframeFetcher(
            symbol=self.symbol,
            timeframes=self.timeframes,
            exchange=self.exchange
        )
        
        self.deepseek_ai = DeepSeekAI()
        self.forecaster = PriceForecaster(model_path="lstm_model.h5")
        
    def run_live_analysis(self):
        """Run analysis with comprehensive error handling"""
        try:
            # Fetch data with timeout
            fetched_data = asyncio.run(self._fetch_data_with_timeout(30))
            
            # Validate primary data
            if not fetched_data or 'primary' not in fetched_data:
                logger.error("Primary data unavailable")
                # Attempt emergency fallback
                return self._emergency_fallback_analysis()
                
            # Initialize analyzers
            self.analyzers = {
                tf_key: TechnicalAnalyzer(df) 
                for tf_key, df in fetched_data.items()
            }
        
        # Fundamental analysis (async)
            fundamental_analyzer = FundamentalAnalyzer(self.symbol, self.exchange)
            whale_data = asyncio.run(fundamental_analyzer.whale_volume_analysis())
            
            # AI analysis (using primary timeframe)
            primary_analyzer = self.analyzers['primary']
            last_row = primary_analyzer.df.iloc[-1]
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
                prices = primary_analyzer.df['close'].values
                forecast = self.forecaster.forecast(prices)
            except Exception as e:
                logger.error(f"Forecasting error: {str(e)}")
                forecast = []
            
            # Bridge analysis
            bridge_report = {
                'pivots': self._calculate_dynamic_pivots(),
                'bridges': primary_analyzer.identify_price_bridges(),
                'liquidation_clusters': primary_analyzer.detect_liquidation_clusters(),
                'multi_timeframe': self._check_timeframe_alignment()
            }
            
            # Generate trade signal using PriceBridgeEngine
            trading_engine = IntelligentTradingEngine(
                primary_analyzer=self.analyzers['primary'],
                confirm_analyzer=self.analyzers['confirm'],
                entry_analyzer=self.analyzers['entry']
            )
            trade_idea = trading_engine.generate_trade_idea()
            
            # Get volatility index
            volatility_index = primary_analyzer.df['atr'].iloc[-1] / primary_analyzer.df['close'].iloc[-1] * 100
            
            # Calculate position size with bridge-specific parameters
            position_size = 0
            if trade_idea:
                risk_manager = RiskManager(self.total_capital)
                position_size = risk_manager.calculate_bridge_position(
                    entry_price=trade_idea['entry_price'],
                    stop_loss=trade_idea['stop_loss'],
                    volatility=volatility_index,
                    bridge_strength=bridge_report['bridges']['strength_score'],
                    liquidation_risk=bridge_report['liquidation_clusters']['risk_score']
                )
            
            # Prepare report (maintain backward compatibility)
            report = {
                'symbol': self.symbol,
                'timeframe': self.timeframes['primary'],  # Maintain single timeframe field
                'exchange': self.exchange,
                'timestamp': datetime.utcnow().isoformat(),
                'price': primary_analyzer.df['close'].iloc[-1],
                'ai_recommendation': ai_recommendation,
                'whale_activity': whale_data,
                'price_forecast': forecast,
                'market_structure': getattr(primary_analyzer, 'market_structure', {}),
                'trade_idea': trade_idea,
                'position_size': position_size,
                'volatility_index': volatility_index,
                'bridge_report': bridge_report  # New field
            }
            
            return report, primary_analyzer
        
        except Exception as e:
            logger.critical(f"Analysis failed: {str(e)}")
            return None, None

    async def _fetch_data_with_timeout(self, timeout: int):
        """Fetch data with timeout protection"""
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.data_fetcher.get_candles
                ), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error("Data fetching timed out")
            return {}
            
    def _emergency_fallback_analysis(self):
        """Fallback to single timeframe when multi fails"""
        logger.warning("Using emergency fallback to primary timeframe")
        try:
            fetcher = DataFetcher(self.symbol, self.timeframes['primary'], self.exchange)
            df = fetcher.get_candles()
            
            if df.empty:
                logger.error("Fallback failed: No primary data")
                return None, None
                
            self.analyzers = {'primary': TechnicalAnalyzer(df)}
            # [Rest of analysis without multi-timeframe features]
            # ... (متناسب با نیازتان این بخش را کامل کنید)
            
        except Exception as e:
            logger.error(f"Fallback failed: {str(e)}")
            return None, None        

    def _calculate_dynamic_pivots(self):
        """Calculate dynamic support/resistance pivots"""
        # Implementation logic here
        return {"pivot": self.analyzers['primary'].calculate_pivots()}
    
    def _check_timeframe_alignment(self):
        """Check alignment across multiple timeframes"""
        alignment = {}
        for tf_key, analyzer in self.analyzers.items():
            alignment[tf_key] = {
                'trend': analyzer.current_trend(),
                'momentum': analyzer.current_momentum()
            }
        return alignment