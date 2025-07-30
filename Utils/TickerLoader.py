# Utils/TickerLoader.py

import os
import logging

def load_tickers_from_file(ticker_file="TickerList.txt"):
    try:
        possible_paths = [
            ticker_file,
            os.path.join("Logic", ticker_file),
            os.path.join("Config", ticker_file),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ticker_file)
        ]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
                logging.info(f"Loaded {len(tickers)} tickers from {path}")
                return tickers
        default_tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL']
        logging.warning(f"Ticker file not found, using defaults: {default_tickers}")
        return default_tickers
    except Exception as e:
        logging.error(f"Error loading tickers: {e}")
        return ['TSLA', 'NVDA']
