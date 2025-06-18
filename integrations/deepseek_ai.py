# integrations/deepseek_ai.py
import requests
from datetime import datetime
from config import Config
from utils import setup_logger, convert_to_native

logger = setup_logger()

class DeepSeekAI:
    """Enhanced AI integration with market sentiment analysis"""
    def analyze(self, data: dict, market_sentiment: float = 0.5) -> dict:
        """
        Enhanced analysis with technical indicators and market sentiment
        :param data: Technical indicators data
        :param market_sentiment: Market sentiment score (0-1)
        :return: Analysis results
        """
        if not Config.DEEPSEEK_API_KEY:
            logger.warning("DeepSeek API Key is missing. Using local fallback.")
            return self.local_analysis(data, market_sentiment)
        payload = {
            "technical_data": data,
            "market_sentiment": market_sentiment,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_type": "advanced_ta"
        }
        payload = convert_to_native(payload)

        headers = {
            "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(Config.DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            logger.error(f"DeepSeek AI Error: {str(e)}")
            return self.local_analysis(data, market_sentiment)
    
    def local_analysis(self, data: dict, sentiment: float) -> dict:
        rsi = data.get("rsi", 50)
        macd_diff = data.get("macd_diff", 0)
        ma_50 = data.get("ma_50", 0)
        ma_200 = data.get("ma_200", 0)
        adx = data.get("adx", 25)
        
        # Enhanced logic with ADX
        if adx > 25:  # Strong trend
            if rsi < 35 and macd_diff > 0 and ma_50 > ma_200:
                return {"recommendation": "strong buy", "confidence": 0.8, "reason": "Oversold with bullish divergence in strong trend"}
            elif rsi > 65 and macd_diff < 0 and ma_50 < ma_200:
                return {"recommendation": "strong sell", "confidence": 0.8, "reason": "Overbought with bearish divergence in strong trend"}
        else:  # Weak trend
            if rsi < 30 and macd_diff > 0:
                return {"recommendation": "buy", "confidence": 0.65, "reason": "Oversold with bullish momentum"}
            elif rsi > 70 and macd_diff < 0:
                return {"recommendation": "sell", "confidence": 0.65, "reason": "Overbought with bearish momentum"}
        
        # Neutral conditions
        if ma_50 > ma_200 and macd_diff > 0:
            return {"recommendation": "buy", "confidence": 0.6, "reason": "Bullish trend"}
        elif ma_50 < ma_200 and macd_diff < 0:
            return {"recommendation": "sell", "confidence": 0.6, "reason": "Bearish trend"}
        else:
            return {"recommendation": "hold", "confidence": 0.5, "reason": "No clear signal"}
