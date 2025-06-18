# ml/forecaster.py
import os
import numpy as np
import tensorflow
from typing import List, Dict, Callable, Tuple, Optional, Any
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from utils import setup_logger

logger = setup_logger()

class PriceForecaster:
    def __init__(self, model_path: str = None, lookback: int = 60, forecast_horizon: int = 10):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = self._build_lstm_model()    
    def _build_lstm_model(self) -> Sequential:
        """Build LSTM model for price forecasting"""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.lookback, 1)))
        model.add(LSTM(50))
        model.add(Dense(self.forecast_horizon))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_data(self, prices: np.array) -> Tuple[np.array, np.array]:
        """Prepare data for LSTM training"""
        X, y = [], []
        for i in range(len(prices) - self.lookback - self.forecast_horizon):
            X.append(prices[i:i+self.lookback])
            y.append(prices[i+self.lookback:i+self.lookback+self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def train(self, prices: np.array, epochs: int = 50, batch_size: int = 32, save_path: str = None):
        """Train the forecasting model"""
        if len(prices) < self.lookback + self.forecast_horizon + 100:
            logger.warning("Insufficient data for training")
            return
            
        # Prepare data
        X, y = self.prepare_data(prices)
        
        # Train-test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Train model
        self.model.fit(X_train, y_train, 
                      epochs=epochs, batch_size=batch_size,
                      validation_data=(X_test, y_test),
                      verbose=0)
        
        # Save model if requested
        if save_path:
            self.model.save(save_path)
    
    def forecast(self, recent_prices: np.array) -> np.array:
        """Generate price forecast"""
        if len(recent_prices) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} prices for forecasting")
            
        # Prepare input
        input_data = recent_prices[-self.lookback:].reshape(1, self.lookback, 1)
        
        # Generate forecast
        forecast = self.model.predict(input_data, verbose=0)
        return forecast[0]
