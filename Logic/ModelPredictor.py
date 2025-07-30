# Logic/ModelPredictor.py

import os
import json
import logging
import joblib
import numpy as np

class ModelPredictor:
    """Handles loading models and making predictions"""
    def __init__(self, model_dir: str = "Models"):
        self.model_dir = model_dir
        self.loaded_models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_names = {}
        self.last_predictions = {}

    def load_model_for_symbol(self, symbol: str, model_name: str = "xgb"):
        try:
            symbol_dir = os.path.join(self.model_dir, symbol)
            model_path = os.path.join(symbol_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                logging.warning(f"Model not found: {model_path}")
                return False
            model = joblib.load(model_path)
            scaler_path = os.path.join(symbol_dir, "scaler.joblib")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            selector_path = os.path.join(symbol_dir, "feature_selector.joblib")
            selector = joblib.load(selector_path) if os.path.exists(selector_path) else None
            metadata_path = os.path.join(symbol_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    feature_names = metadata.get('feature_columns', [])
            else:
                feature_names = []
            model_key = f"{symbol}_{model_name}"
            self.loaded_models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_selectors[model_key] = selector
            self.feature_names[model_key] = feature_names
            logging.info(f"Loaded model {model_name} for {symbol}")
            return True
        except Exception as e:
            logging.error(f"Error loading model for {symbol}: {e}")
            return False

    def predict(self, symbol: str, features: np.ndarray, model_name: str = "xgb"):
        try:
            model_key = f"{symbol}_{model_name}"
            if model_key not in self.loaded_models:
                if not self.load_model_for_symbol(symbol, model_name):
                    return 0, 0.0
            model = self.loaded_models[model_key]
            scaler = self.scalers.get(model_key)
            selector = self.feature_selectors.get(model_key)
            processed_features = features.copy()
            if scaler is not None:
                processed_features = scaler.transform(processed_features)
            if selector is not None:
                processed_features = selector.transform(processed_features)
            prediction = model.predict(processed_features)[0]
            confidence = model.predict_proba(processed_features)[0].max() if hasattr(model, 'predict_proba') else 0.5
            return int(prediction), float(confidence)
        except Exception as e:
            logging.error(f"Error making prediction for {symbol}: {e}")
            return 0, 0.0
