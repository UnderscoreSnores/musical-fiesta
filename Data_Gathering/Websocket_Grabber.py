# Data_Gathering/WebsocketDataGrabber.py

import logging
import json
from Utils.TickerLoader import load_tickers_from_file
from Database.PostgreSQL_Realtime import PostgresRealtimeDB

class WebsocketDataGrabber:
    """Legacy class for backward compatibility"""
    def __init__(self, endpoint, ticker_file="TickerList.txt"):
        self.endpoint = endpoint
        self.tickers = load_tickers_from_file(ticker_file)
        self.db = PostgresRealtimeDB()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler('websocket_downloader.log'),
                logging.StreamHandler()
            ]
        )

    async def subscribe(self, ws):
        msg = {"action": "subscribe", "symbols": ",".join(self.tickers)}
        await ws.send(json.dumps(msg))
        logging.info(f"Subscribed to: {self.tickers}")

    async def handle_messages(self, ws):
        async for message in ws:
            try:
                data = json.loads(message)
                if isinstance(data, list):
                    for tick in data:
                        await self.db.save_tick(tick)
                elif isinstance(data, dict):
                    await self.db.save_tick(data)
            except Exception as e:
                logging.error(f"Error processing message: {e}")
