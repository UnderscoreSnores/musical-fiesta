import os
import pandas as pd
import numpy as np
import joblib
import logging
import asyncpg
import asyncio
import warnings
warnings.filterwarnings("ignore", message="Protobuf gencode version")
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from tqdm import tqdm
import time

from Utils.Config_Loader import load_config

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, TimeSeriesSplit,
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models will be skipped.")

warnings.filterwarnings('ignore')

class AdvancedTrainerPipeline:
    class Config:
        def __init__(self, pg_dsn, model_dir, min_data_points=1000, test_size=0.2,
                     validation_size=0.2, cv_folds=5, random_state=42, n_jobs=-1,
                     enable_deep_learning=TENSORFLOW_AVAILABLE, enable_hyperparameter_tuning=True,
                     feature_selection_k=25, lookback_periods=None):
            self.pg_dsn = pg_dsn
            self.model_dir = model_dir
            self.min_data_points = min_data_points
            self.test_size = test_size
            self.validation_size = validation_size
            self.cv_folds = cv_folds
            self.random_state = random_state
            self.n_jobs = n_jobs
            self.enable_deep_learning = enable_deep_learning
            self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
            self.feature_selection_k = feature_selection_k
            self.lookback_periods = lookback_periods or [5, 10, 20, 30, 60]

    def __init__(self, pg_dsn=None, model_dir=None):
        self.config = self.Config(
            pg_dsn,
            model_dir or os.path.join(os.path.dirname(__file__), '..', 'Models')
        )
        os.makedirs(self.config.model_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.model_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.feature_names = []
        self.scalers = {}
        self.feature_selectors = {}

    # --- FEATURE ENGINEERING (Optimized) ---
    def create_technical_indicators(self, df):
        df = df.copy()
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = SMAIndicator(df['close'], window=period).sma_indicator()
            df[f'ema_{period}'] = EMAIndicator(df['close'], window=period).ema_indicator()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
            df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}']
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        bb = BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        return df

    def create_price_features(self, df):
        df = df.copy()
        for period in [5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        df['high_low_ratio'] = df['high'] / df['low']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        for period in [10, 20]:
            df[f'high_{period}'] = df['high'].rolling(period).max()
            df[f'low_{period}'] = df['low'].rolling(period).min()
            df[f'price_position_{period}'] = (df['close'] - df[f'low_{period}']) / (
                        df[f'high_{period}'] - df[f'low_{period}'])
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        return df

    def create_time_features(self, df):
        df = df.copy()
        if 'ts' in df.columns:
            df['hour'] = df['ts'].dt.hour
            df['day_of_week'] = df['ts'].dt.dayofweek
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df

    def create_lag_features(self, df):
        df = df.copy()
        base_features = ['close', 'volume', 'rsi_14', 'macd', 'bb_position']
        for feature in base_features:
            if feature in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        return df

    def create_all_features(self, df):
        df = self.create_technical_indicators(df)
        df = self.create_price_features(df)
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df['target_1'] = (df['close'].shift(-1) > df['close']).astype(int)
        df['target_5'] = (df['close'].shift(-5) > df['close']).astype(int)
        df['target_return'] = df['close'].pct_change(-1)
        exclude_cols = ['ts', 'open', 'high', 'low', 'close', 'volume', 'target_1', 'target_5', 'target_return']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        return df

    def get_base_models(self):
        c = self.config
        models = {
            'xgb': XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=c.random_state,
                eval_metric='logloss', n_jobs=c.n_jobs
            ),
            'lgbm': LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=c.random_state,
                n_jobs=c.n_jobs, verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=200, depth=8, learning_rate=0.1,
                random_state=c.random_state, verbose=False,
                thread_count=c.n_jobs if c.n_jobs > 0 else None
            ),
            'rf': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=c.random_state, n_jobs=c.n_jobs
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=c.random_state, n_jobs=c.n_jobs
            ),
            'lr': LogisticRegression(
                max_iter=1000, random_state=c.random_state, n_jobs=c.n_jobs
            )
        }
        return models

    def get_ensemble_models(self, base_models):
        c = self.config
        ensemble_base = {k: v for k, v in base_models.items()
                         if k in ['xgb', 'lgbm', 'catboost', 'rf', 'extra_trees']}
        ensembles = {
            'voting_soft': VotingClassifier(
                estimators=list(ensemble_base.items()),
                voting='soft', n_jobs=c.n_jobs
            ),
            'stacking': StackingClassifier(
                estimators=list(ensemble_base.items()),
                final_estimator=LogisticRegression(random_state=c.random_state),
                cv=3, n_jobs=c.n_jobs
            )
        }
        return ensembles

    def create_lstm_model(self, input_shape):
        if not TENSORFLOW_AVAILABLE:
            return None
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_param_grids(self):
        return {
            'xgb': {
                'n_estimators': [150, 200, 250],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'lgbm': {
                'n_estimators': [150, 200, 250],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'catboost': {
                'iterations': [150, 200, 250],
                'depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        }

    def optimize_model(self, model_name, model, X, y):
        param_grids = self.get_param_grids()
        if model_name not in param_grids:
            return model
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = RandomizedSearchCV(
            model,
            param_grids[model_name],
            cv=tscv,
            scoring='accuracy',
            n_iter=15,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        grid_search.fit(X, y)
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logging.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def load_trained_model(self, symbol, model_name):
        import os
        import joblib
        symbol_dir = os.path.join(self.config.model_dir, symbol)
        if model_name == "lstm":
            # LSTM models are saved as .h5 files
            model_path = os.path.join(symbol_dir, "lstm.h5")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            try:
                from tensorflow.keras.models import load_model
            except ImportError:
                raise ImportError("TensorFlow/Keras is not installed.")
            model = load_model(model_path)
            scaler_path = os.path.join(symbol_dir, "scaler.joblib")
            selector_path = os.path.join(symbol_dir, "feature_selector.joblib")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            selector = joblib.load(selector_path) if os.path.exists(selector_path) else None
            return model, scaler, selector
        else:
            model_path = os.path.join(symbol_dir, f"{model_name}.joblib")
            scaler_path = os.path.join(symbol_dir, "scaler.joblib")
            selector_path = os.path.join(symbol_dir, "feature_selector.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            selector = joblib.load(selector_path) if os.path.exists(selector_path) else None
            return model, scaler, selector

    async def load_data(self, symbol):
        query = """
                SELECT ts, open, high, low, close, volume
                FROM stock_bars_minute
                WHERE symbol = $1
                ORDER BY ts \
                """
        try:
            conn = await asyncpg.connect(self.config.pg_dsn)
            rows = await conn.fetch(query, symbol)
            await conn.close()
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logging.error(f"Error loading data for {symbol}: {e}")
            return None

    def prepare_data(self, df, target_col='target_1'):
        df_features = self.create_all_features(df)
        df_clean = df_features.dropna()
        if len(df_clean) < self.config.min_data_points:
            raise ValueError(f"Not enough clean data points: {len(df_clean)}")
        target_dist = df_clean[target_col].value_counts()
        if len(target_dist) < 2 or target_dist.min() < 50:
            raise ValueError(f"Insufficient target class diversity: {target_dist.to_dict()}")
        feature_columns = self.feature_names
        X = df_clean[feature_columns].values
        y = df_clean[target_col].values
        return X, y, feature_columns

    def preprocess_features(self, X_train, X_test, symbol, feature_columns, y_train):
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        selector = SelectKBest(
            mutual_info_classif,
            k=min(self.config.feature_selection_k, X_train.shape[1])
        )
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        self.scalers[symbol] = scaler
        self.feature_selectors[symbol] = selector
        return X_train_selected, X_test_selected

    def evaluate_model(self, model, X_test, y_test, model_name, symbol):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        return metrics

    def cross_validate_model(self, model, X, y, model_name, symbol):
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=self.config.n_jobs)
        cv_metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
        return cv_metrics

    def save_model_artifacts(self, symbol, model, model_name, metrics, feature_columns):
        symbol_dir = os.path.join(self.config.model_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        model_path = os.path.join(symbol_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        if symbol in self.scalers:
            scaler_path = os.path.join(symbol_dir, "scaler.joblib")
            joblib.dump(self.scalers[symbol], scaler_path)
        if symbol in self.feature_selectors:
            selector_path = os.path.join(symbol_dir, "feature_selector.joblib")
            joblib.dump(self.feature_selectors[symbol], selector_path)
        metadata = {
            'symbol': symbol,
            'model_name': model_name,
            'metrics': metrics,
            'feature_columns': feature_columns,
            'training_date': datetime.now().isoformat(),
            'config': {
                'min_data_points': self.config.min_data_points,
                'test_size': self.config.test_size,
                'feature_selection_k': self.config.feature_selection_k
            }
        }
        metadata_path = os.path.join(symbol_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_training_summary(self, symbol, summary):
        symbol_dir = os.path.join(self.config.model_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        summary_path = os.path.join(symbol_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    async def train_symbol(self, symbol, progress_bar=None):
        if progress_bar:
            progress_bar.set_description(f"Loading {symbol}")
        df = await self.load_data(symbol)
        if df is None:
            logging.warning(f"No data available for {symbol}")
            return False
        try:
            X, y, feature_columns = self.prepare_data(df)
        except ValueError as e:
            logging.warning(f"Data preparation failed for {symbol}: {e}")
            return False
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, shuffle=False,
            random_state=self.config.random_state
        )
        X_train_processed, X_test_processed = self.preprocess_features(
            X_train, X_test, symbol, feature_columns, y_train
        )
        base_models = self.get_base_models()
        ensemble_models = self.get_ensemble_models(base_models) if len(base_models) >= 3 else {}
        lstm_enabled = self.config.enable_deep_learning and TENSORFLOW_AVAILABLE
        all_metrics = {}
        trained_models = {}
        total_models = len(base_models) + len(ensemble_models) + (1 if lstm_enabled else 0)
        model_progress = 0
        for model_name, model in base_models.items():
            if progress_bar:
                progress_bar.set_description(f"{symbol}: Training {model_name}")
            try:
                if (self.config.enable_hyperparameter_tuning and
                        model_name in ['xgb', 'lgbm', 'catboost']):
                    model = self.optimize_model(model_name, model, X_train_processed, y_train)
                model.fit(X_train_processed, y_train)
                test_metrics = self.evaluate_model(model, X_test_processed, y_test, model_name, symbol)
                cv_metrics = self.cross_validate_model(model, X_train_processed, y_train, model_name, symbol)
                combined_metrics = {**test_metrics, **cv_metrics}
                all_metrics[model_name] = combined_metrics
                self.save_model_artifacts(symbol, model, model_name, combined_metrics, feature_columns)
                trained_models[model_name] = model
                model_progress += 1
                if progress_bar:
                    progress_bar.set_postfix({
                        'models': f"{model_progress}/{total_models}",
                        'acc': f"{test_metrics.get('accuracy', 0):.3f}"
                    })
            except Exception as e:
                logging.error(f"Error training {model_name} for {symbol}: {e}")
                continue
        for ensemble_name, ensemble_model in ensemble_models.items():
            if progress_bar:
                progress_bar.set_description(f"{symbol}: Training {ensemble_name}")
            try:
                ensemble_model.fit(X_train_processed, y_train)
                test_metrics = self.evaluate_model(ensemble_model, X_test_processed, y_test, ensemble_name, symbol)
                cv_metrics = self.cross_validate_model(ensemble_model, X_train_processed, y_train, ensemble_name, symbol)
                combined_metrics = {**test_metrics, **cv_metrics}
                all_metrics[ensemble_name] = combined_metrics
                self.save_model_artifacts(symbol, ensemble_model, ensemble_name, combined_metrics, feature_columns)
                model_progress += 1
                if progress_bar:
                    progress_bar.set_postfix({
                        'models': f"{model_progress}/{total_models}",
                        'acc': f"{test_metrics.get('accuracy', 0):.3f}"
                    })
            except Exception as e:
                logging.error(f"Error training {ensemble_name} for {symbol}: {e}")
                continue
        if lstm_enabled:
            if progress_bar:
                progress_bar.set_description(f"{symbol}: Training LSTM")
            try:
                sequence_length = 60
                if len(X_train_processed) > sequence_length:
                    def create_sequences(data, seq_length):
                        sequences = []
                        for i in range(seq_length, len(data)):
                            sequences.append(data[i - seq_length:i])
                        return np.array(sequences)
                    X_train_seq = create_sequences(X_train_processed, sequence_length)
                    y_train_seq = y_train[sequence_length:]
                    X_test_seq = create_sequences(X_test_processed, sequence_length)
                    y_test_seq = y_test[sequence_length:]
                    lstm_model = self.create_lstm_model((sequence_length, X_train_processed.shape[1]))
                    if lstm_model is not None:
                        callbacks = [
                            EarlyStopping(patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(patience=5, factor=0.5)
                        ]
                        history = lstm_model.fit(
                            X_train_seq, y_train_seq,
                            epochs=50,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=callbacks,
                            verbose=0
                        )
                        y_pred_lstm = (lstm_model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
                        y_pred_proba_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
                        lstm_metrics = {
                            'accuracy': accuracy_score(y_test_seq, y_pred_lstm),
                            'precision': precision_score(y_test_seq, y_pred_lstm, average='weighted', zero_division=0),
                            'recall': recall_score(y_test_seq, y_pred_lstm, average='weighted', zero_division=0),
                            'f1': f1_score(y_test_seq, y_pred_lstm, average='weighted', zero_division=0),
                            'auc': roc_auc_score(y_test_seq, y_pred_proba_lstm),
                            'mcc': matthews_corrcoef(y_test_seq, y_pred_lstm)
                        }
                        all_metrics['lstm'] = lstm_metrics
                        symbol_dir = os.path.join(self.config.model_dir, symbol)
                        lstm_path = os.path.join(symbol_dir, "lstm.h5")
                        lstm_model.save(lstm_path)
                        model_progress += 1
                        if progress_bar:
                            progress_bar.set_postfix({
                                'models': f"{model_progress}/{total_models}",
                                'acc': f"{lstm_metrics.get('accuracy', 0):.3f}"
                            })
            except Exception as e:
                logging.error(f"Error training LSTM for {symbol}: {e}")
        if all_metrics:
            best_model = max(all_metrics.items(), key=lambda x: x[1].get('accuracy', 0))
            summary = {
                'symbol': symbol,
                'best_model': best_model[0],
                'all_metrics': all_metrics,
                'training_completed': datetime.now().isoformat()
            }
            self.save_training_summary(symbol, summary)
            if progress_bar:
                progress_bar.set_postfix({
                    'best': f"{best_model[0]}",
                    'acc': f"{best_model[1].get('accuracy', 0):.3f}"
                })

    async def run(self):
        logging.info("Starting advanced training pipeline")
        try:
            conn = await asyncpg.connect(self.config.pg_dsn)
            rows = await conn.fetch("SELECT DISTINCT symbol FROM stock_bars_minute ORDER BY symbol")
            await conn.close()
            symbols = [row['symbol'] for row in rows]
            logging.info(f"Found {len(symbols)} symbols to train")
        except Exception as e:
            logging.error(f"Error fetching symbols: {e}")
            return
        with tqdm(total=len(symbols), desc="Symbols") as pbar:
            for symbol in symbols:
                await self.train_symbol(symbol, progress_bar=pbar)
                pbar.update(1)
        logging.info("Training pipeline completed.")
