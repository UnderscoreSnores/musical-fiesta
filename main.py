# main.py

import os
import asyncio
from Utils.Config_Loader import load_config

def run_download_all(ticker_file):
    from Data_Gathering.API_Grabber import download_tickers_from_file
    print(f"DEBUG: Using ticker file: {ticker_file}")
    download_tickers_from_file(ticker_file)

def run_training(pg_dsn, model_dir):
    from Logic.Train import AdvancedTrainerPipeline
    print("DEBUG: Starting training pipeline")
    print(f"DEBUG: pg_dsn = {pg_dsn}")
    asyncio.run(AdvancedTrainerPipeline(pg_dsn=pg_dsn, model_dir=model_dir).run())

def main():
    print("What would you like to do?")
    print("1. Download minute data")
    print("2. Train all models")
    print("3. Download and train (all)")

    config = load_config("config.txt")
    pg_dsn = config.get("POSTGRES_DSN")
    model_dir = config.get("MODEL_DIR", "Models")
    ticker_file = config.get("TICKER_FILE", os.path.join("Logic", "TickerList.txt"))

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        run_download_all(ticker_file)
    elif choice == "2":
        run_training(pg_dsn, model_dir)
    elif choice == "3":
        run_download_all(ticker_file)
        run_training(pg_dsn, model_dir)
    else:
        print("Invalid choice. Please run the program again and enter 1, 2, or 3.")

if __name__ == "__main__":
    print("DEBUG: main.py started")
    main()
