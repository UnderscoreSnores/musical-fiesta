# Logic/FeatureEngine.py

import numpy as np
from datetime import datetime
from collections import deque, defaultdict
import logging

class FeatureEngine:
    """Real-time feature calculation"""
    def __init__(self, lookback_periods = [5, 10, 20, 30]):
        self.lookback_periods = lookback_periods
        self.price_history = defaultdict(deque)
        self.volume_history = defaultdict(deque)
        self.max_history = max(lookback_periods) + 50

    def update_history(self, symbol: str, price: float, volume: int):
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        if len(self.price_history[symbol]) > self.max_history:
            self.price_history[symbol].popleft()
            self.volume_history[symbol].popleft()

    def calculate_features(self, symbol: str):
        try:
            prices = np.array(self.price_history[symbol])
            volumes = np.array(self.volume_history[symbol])
            if len(prices) < max(self.lookback_periods):
                return None
            features = []
            current_price = prices[-1]
            for period in self.lookback_periods:
                if len(prices) >= period:
                    returns = (current_price - prices[-period]) / prices[-period]
                    features.append(returns)
                    sma = np.mean(prices[-period:])
                    features.append(current_price / sma - 1)
                    if period > 1:
                        volatility = np.std(prices[-period:]) / np.mean(prices[-period:])
                        features.append(volatility)
                    else:
                        features.append(0.0)
            if len(prices) >= 14:
                deltas = np.diff(prices[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss != 0:
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                else:
                    rsi = 100
                features.append(rsi / 100)
            else:
                features.append(0.5)
            if len(volumes) >= 5:
                volume_sma = np.mean(volumes[-5:])
                volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
                features.append(min(volume_ratio, 10.0))
            else:
                features.append(1.0)
            if len(prices) >= 20:
                high_20 = np.max(prices[-20:])
                low_20 = np.min(prices[-20:])
                if high_20 != low_20:
                    price_position = (current_price - low_20) / (high_20 - low_20)
                else:
                    price_position = 0.5
                features.append(price_position)
            else:
                features.append(0.5)
            now = datetime.now()
            hour_sin = np.sin(2 * np.pi * now.hour / 24)
            hour_cos = np.cos(2 * np.pi * now.hour / 24)
            features.extend([hour_sin, hour_cos])
            is_market_hours = 1.0 if 9 <= now.hour < 16 else 0.0
            features.append(is_market_hours)
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logging.error(f"Error calculating features for {symbol}: {e}")
            return None
