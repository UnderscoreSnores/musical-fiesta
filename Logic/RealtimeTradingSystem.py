# Logic/RealtimeTradingSystem.py

import asyncio
import logging
import json
from datetime import datetime
from collections import defaultdict

from Utils.TickerLoader import load_tickers_from_file
from Database.PostgreSQL_Realtime import PostgresRealtimeDB
from Logic.ModelPredictor import ModelPredictor
from Logic.FeatureEngine import FeatureEngine
from Logic.TradingSignal import TradingSignal

class RealtimeTradingSystem:
    """Main real-time trading system"""
    def __init__(self, endpoint, ticker_file="TickerList.txt", model_dir="Models", signal_threshold=0.7):
        # Always resolve model_dir relative to project root
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(model_dir):
            self.model_dir = os.path.join(project_root, model_dir)
        else:
            self.model_dir = model_dir

        self.endpoint = endpoint
        self.tickers = load_tickers_from_file(ticker_file)
        self.db = PostgresRealtimeDB()
        self.predictor = ModelPredictor(self.model_dir)
        self.feature_engine = FeatureEngine()
        self.signal_threshold = signal_threshold
        self.positions = {}
        self.last_signals = {}
        self.signal_history = defaultdict(list)
        self.total_signals = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.setup_logging()


    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler('realtime_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def subscribe(self, ws):
        msg = {"action": "subscribe", "symbols": ",".join(self.tickers)}
        await ws.send(json.dumps(msg))
        self.logger.info(f"Subscribed to: {self.tickers}")

    def generate_signal(self, symbol: str, price: float, volume: int):
        try:
            self.feature_engine.update_history(symbol, price, volume)
            features = self.feature_engine.calculate_features(symbol)
            if features is None:
                return None
            prediction, confidence = self.predictor.predict(symbol, features)
            if confidence < self.signal_threshold:
                return None
            if prediction == 1 and confidence > self.signal_threshold:
                action = "BUY"
            elif prediction == 0 and confidence > self.signal_threshold:
                action = "SELL"
            else:
                action = "HOLD"
            if action != "HOLD":
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=price,
                    timestamp=datetime.now(),
                    model_name="xgb",
                    features={
                        'feature_count': features.shape[1],
                        'volume': volume
                    }
                )
                self.total_signals += 1
                if action == "BUY":
                    self.buy_signals += 1
                else:
                    self.sell_signals += 1
                return signal
            return None
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def should_execute_signal(self, signal: TradingSignal) -> bool:
        if signal.symbol in self.last_signals:
            last_signal = self.last_signals[signal.symbol]
            time_diff = (signal.timestamp - last_signal.timestamp).total_seconds()
            if time_diff < 60 and last_signal.action == signal.action:
                return False
        current_position = self.positions.get(signal.symbol, 0)
        if signal.action == "BUY" and current_position >= 100:
            return False
        elif signal.action == "SELL" and current_position <= 0:
            return False
        return True

    def execute_signal(self, signal: TradingSignal):
        if not self.should_execute_signal(signal):
            return
        current_position = self.positions.get(signal.symbol, 0)
        if signal.action == "BUY":
            self.positions[signal.symbol] = current_position + 10
            print(f"🟢 BUY {signal.symbol} @ ${signal.price:.2f} (Confidence: {signal.confidence:.2f})")
        elif signal.action == "SELL":
            self.positions[signal.symbol] = max(0, current_position - 10)
            print(f"🔴 SELL {signal.symbol} @ ${signal.price:.2f} (Confidence: {signal.confidence:.2f})")
        self.last_signals[signal.symbol] = signal
        self.signal_history[signal.symbol].append(signal)
        if len(self.signal_history[signal.symbol]) > 100:
            self.signal_history[signal.symbol] = self.signal_history[signal.symbol][-100:]

    async def handle_tick(self, tick_data: dict):
        try:
            symbol = tick_data.get('s')
            price = float(tick_data.get('p', 0))
            volume = int(tick_data.get('v', 0))
            if not symbol or price <= 0:
                return
            await self.db.save_tick(tick_data)
            signal = self.generate_signal(symbol, price, volume)
            if signal:
                self.execute_signal(signal)
        except Exception as e:
            self.logger.error(f"Error handling tick: {e}")

    async def handle_messages(self, ws):
        async for message in ws:
            try:
                data = json.loads(message)
                if isinstance(data, list):
                    for tick in data:
                        await self.handle_tick(tick)
                elif isinstance(data, dict):
                    await self.handle_tick(data)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

    def print_status(self):
        print(f"\n📊 Trading System Status:")
        print(f"   Signals Generated: {self.total_signals}")
        print(f"   Buy Signals: {self.buy_signals}")
        print(f"   Sell Signals: {self.sell_signals}")
        print(f"   Active Positions: {len([p for p in self.positions.values() if p > 0])}")
        if self.positions:
            print(f"   Current Positions:")
            for symbol, position in self.positions.items():
                if position > 0:
                    print(f"     {symbol}: {position} shares")

    async def periodic_status(self):
        while True:
            await asyncio.sleep(300)
            self.print_status()

    async def run(self):
        await self.db.connect()
        print("🔄 Loading ML models...")
        loaded_models = 0
        for ticker in self.tickers:
            if self.predictor.load_model_for_symbol(ticker):
                loaded_models += 1
        print(f"✅ Loaded models for {loaded_models}/{len(self.tickers)} symbols")
        if loaded_models == 0:
            print("❌ No models loaded! Please train models first.")
            return
        asyncio.create_task(self.periodic_status())
        print("🚀 Starting real-time trading system...")
        print("💡 Signals will be generated when confidence > {:.1%}".format(self.signal_threshold))
        print("⚠️  This is SIMULATION mode - no real trades executed")
        while True:
            try:
                import websockets
                async with websockets.connect(self.endpoint) as ws:
                    await self.subscribe(ws)
                    await self.handle_messages(ws)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

if __name__ == "__main__":
        import os
        from Utils.Config_Loader import load_config

        config = load_config()
        PG_DSN = config.get("POSTGRES_DSN")
        print(f"DEBUG: PG_DSN = {PG_DSN}")

        # Set your endpoint and ticker file as needed
        API_KEY = config.get("EODHD_API_KEY")
        ENDPOINT = f"wss://ws.eodhistoricaldata.com/ws/us?api_token={API_KEY}"

        trading_system = RealtimeTradingSystem(endpoint=ENDPOINT)
        asyncio.run(trading_system.run())
