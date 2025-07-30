# Logic/TradingSignal.py

from datetime import datetime

class TradingSignal:
    """Represents a trading signal"""
    def __init__(self, symbol: str, action: str, confidence: float, price: float,
                 timestamp: datetime, model_name: str, features: dict = None):
        self.symbol = symbol
        self.action = action  # 'BUY', 'SELL', 'HOLD'
        self.confidence = confidence  # 0.0 to 1.0
        self.price = price
        self.timestamp = timestamp
        self.model_name = model_name
        self.features = features or {}
