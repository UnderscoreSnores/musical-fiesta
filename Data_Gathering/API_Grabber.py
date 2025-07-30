# Data_Gathering/API_Grabber.py

import os
import requests
import pandas as pd
import asyncio
import logging
import time
from datetime import datetime, timedelta
from Database.PostgreSQL_Minute import PostgresMinuteDB
from Utils.Config_Loader import load_config

config = load_config("config.txt")
API_KEY = config.get("EODHD_API_KEY")


class MinuteDataDownloader:
    def __init__(self, api_key=None, interval='5m'):
        self.api_key = api_key or API_KEY
        self.interval = interval
        self.db = PostgresMinuteDB()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler('downloader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_intraday_data(self, ticker, from_ts=None, to_ts=None):
        url = f"https://eodhd.com/api/intraday/{ticker}.US"
        params = {
            'interval': self.interval,
            'api_token': self.api_key,
            'fmt': 'json'
        }
        if from_ts and to_ts:
            params['from'] = int(from_ts)
            params['to'] = int(to_ts)
        try:
            self.logger.info(f"Fetching data for {ticker} with params: {params}")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                self.logger.warning(f"No data returned for {ticker}")
                return None
            if isinstance(data, dict) and 'error' in data:
                self.logger.error(f"API error for {ticker}: {data['error']}")
                return None
            df = pd.DataFrame(data)
            if df.empty:
                self.logger.warning(f"Empty DataFrame for {ticker}")
                return None
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                self.logger.error(f"No datetime column found for {ticker}")
                return None
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing columns for {ticker}: {missing_cols}")
                return None
            df = df.sort_values('datetime').reset_index(drop=True)
            self.logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {ticker}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {ticker}: {e}")
            return None

    def validate_bar_data(self, bar):
        try:
            required_fields = ['symbol', 'ts', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in bar:
                    return False
            o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
            if not (l <= o <= h and l <= c <= h):
                self.logger.warning(f"Invalid OHLC for {bar['symbol']} at {bar['ts']}: O={o}, H={h}, L={l}, C={c}")
                return False
            if any(val <= 0 for val in [o, h, l, c]):
                self.logger.warning(f"Non-positive prices for {bar['symbol']} at {bar['ts']}")
                return False
            if bar['volume'] < 0:
                self.logger.warning(f"Negative volume for {bar['symbol']} at {bar['ts']}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating bar data: {e}")
            return False

    async def save_intraday_to_sql(self, ticker, df):
        saved_count = 0
        error_count = 0
        for idx, row in df.iterrows():
            try:
                volume = row['volume']
                if pd.isna(volume):
                    volume = 0
                bar = {
                    'symbol': ticker,
                    'ts': row['datetime'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(volume)
                }
                if not self.validate_bar_data(bar):
                    error_count += 1
                    continue
                await self.db.save_bar(bar)
                saved_count += 1
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error saving bar for {ticker} at {row['datetime']}: {e}")
        self.logger.info(f"Saved {saved_count} bars for {ticker}, {error_count} errors")

    async def check_existing_data(self, ticker):
        try:
            query = """
                    SELECT MIN(ts)  as earliest_date, \
                           MAX(ts)  as latest_date, \
                           COUNT(*) as total_records
                    FROM stock_bars_minute
                    WHERE symbol = $1 \
                    """
            result = await self.db.pool.fetchrow(query, ticker)
            if result and result['total_records'] > 0:
                return {
                    'has_data': True,
                    'earliest_date': result['earliest_date'],
                    'latest_date': result['latest_date'],
                    'total_records': result['total_records']
                }
            else:
                return {'has_data': False}
        except Exception as e:
            self.logger.error(f"Error checking existing data for {ticker}: {e}")
            return {'has_data': False}

    def calculate_date_ranges(self, start_date=None, end_date=None, max_days_per_request=30):
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        if start_date is None:
            start_date = end_date - timedelta(days=90)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        ranges = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=max_days_per_request), end_date)
            from_ts = int(current_start.timestamp())
            to_ts = int(current_end.timestamp())
            ranges.append((from_ts, to_ts))
            current_start = current_end + timedelta(days=1)
        return ranges

    async def download_with_retry(self, ticker, max_retries=3, delay=1):
        for attempt in range(max_retries):
            try:
                df = self.fetch_intraday_data(ticker)
                if df is not None and not df.empty:
                    return df
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {ticker}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1} for {ticker}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
        return None

    async def download(self, tickers, start_date=None, end_date=None, force_refresh=False):
        # Set default date range: 2 years ago to today
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()

        await self.db.connect()
        print("\n—Download—")
        total_tickers = len(tickers)
        successful_downloads = 0
        failed_downloads = 0
        for i, ticker in enumerate(tickers, 1):
            try:
                self.logger.info(f"Processing ticker {i}/{total_tickers}: {ticker}")
                if not force_refresh:
                    existing_data = await self.check_existing_data(ticker)
                    if existing_data['has_data']:
                        self.logger.info(f"{ticker} already has {existing_data['total_records']} records "
                                         f"from {existing_data['earliest_date']} to {existing_data['latest_date']}")
                df = await self.download_with_retry(ticker)
                if df is not None and not df.empty:
                    await self.save_intraday_to_sql(ticker, df)
                    unique_days = df['datetime'].dt.date.nunique()
                    total_records = len(df)
                    date_range = f"{df['datetime'].min().date()} to {df['datetime'].max().date()}"
                    print(f"{ticker}: Downloaded {unique_days} days ({total_records} records) ✔")
                    print(f"  Date range: {date_range}")
                    actual_days = (df['datetime'].max() - df['datetime'].min()).days
                    print(f"  Actual span: {actual_days} calendar days")
                    avg_records_per_day = total_records / max(unique_days, 1)
                    print(f"  Avg records/day: {avg_records_per_day:.1f}")
                    successful_downloads += 1
                else:
                    print(f"{ticker}: No data found or error ✗")
                    failed_downloads += 1
            except Exception as e:
                self.logger.error(f"Failed to process {ticker}: {e}")
                print(f"{ticker}: Error - {str(e)} ✗")
                failed_downloads += 1
            if i < total_tickers:
                await asyncio.sleep(0.2)
        print(f"\n—Download Summary—")
        print(f"Total tickers: {total_tickers}")
        print(f"Successful: {successful_downloads}")
        print(f"Failed: {failed_downloads}")
        print(f"Success rate: {successful_downloads / total_tickers * 100:.1f}%")
        await self.db.close()

    async def get_available_symbols(self):
        try:
            query = """
                    SELECT symbol,
                           MIN(ts)  as earliest_date,
                           MAX(ts)  as latest_date,
                           COUNT(*) as total_records
                    FROM stock_bars_minute
                    GROUP BY symbol
                    ORDER BY symbol \
                    """
            await self.db.connect()
            rows = await self.db.pool.fetch(query)
            await self.db.close()
            symbols_info = []
            for row in rows:
                symbols_info.append({
                    'symbol': row['symbol'],
                    'earliest_date': row['earliest_date'],
                    'latest_date': row['latest_date'],
                    'total_records': row['total_records']
                })
            return symbols_info
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    def __del__(self):
        try:
            if hasattr(self, 'db') and self.db:
                pass
        except:
            pass


def download_tickers_from_file(
    ticker_file="TickerList.txt",
    start_date=None,
    end_date=None,
    force_refresh=False
):
    try:
        # Set default date range: 2 years ago to today
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ticker_path = os.path.join(project_root, ticker_file)
        if not os.path.exists(ticker_path):
            ticker_path = os.path.join(os.path.dirname(__file__), ticker_file)
        print(f"DEBUG: Looking for ticker file at: {ticker_path}")
        if not os.path.exists(ticker_path):
            raise FileNotFoundError(f"Ticker file not found: {ticker_path}")
        with open(ticker_path, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith('#')]
        print(f"DEBUG: Tickers loaded: {tickers}")
        if not tickers:
            print("No tickers found in file")
            return
        downloader = MinuteDataDownloader(interval='5m')
        asyncio.run(downloader.download(tickers, start_date, end_date, force_refresh))
    except Exception as e:
        print(f"Error in download_tickers_from_file: {e}")
        raise


if __name__ == "__main__":
    print("MinuteDataDownloader - Available operations:")
    print("1. Download from ticker file (last 2 years)")
    print("2. Download specific tickers")
    print("3. Check available symbols")

    choice = input("Enter choice (1-3): ")

    if choice == "1":
        download_tickers_from_file()
    elif choice == "2":
        tickers = input("Enter tickers (comma-separated): ").split(",")
        tickers = [t.strip().upper() for t in tickers]
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now()
        downloader = MinuteDataDownloader()
        asyncio.run(downloader.download(tickers, start_date, end_date))
    elif choice == "3":
        downloader = MinuteDataDownloader()
        symbols = asyncio.run(downloader.get_available_symbols())
        print(f"\nFound {len(symbols)} symbols with data:")
        for symbol_info in symbols:
            print(f"  {symbol_info['symbol']}: {symbol_info['total_records']} records "
                  f"({symbol_info['earliest_date']} to {symbol_info['latest_date']})")
    else:
        print("Invalid choice")
# Data_Gathering/API_Grabber.py

import os
import requests
import pandas as pd
import asyncio
import logging
import time
from datetime import datetime, timedelta
from Database.PostgreSQL_Minute import PostgresMinuteDB
from Utils.Config_Loader import load_config

config = load_config("config.txt")
API_KEY = config.get("EODHD_API_KEY")


class MinuteDataDownloader:
    def __init__(self, api_key=None, interval='5m'):
        self.api_key = api_key or API_KEY
        self.interval = interval
        self.db = PostgresMinuteDB()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler('downloader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_intraday_data(self, ticker, from_ts=None, to_ts=None):
        url = f"https://eodhd.com/api/intraday/{ticker}.US"
        params = {
            'interval': self.interval,
            'api_token': self.api_key,
            'fmt': 'json'
        }
        if from_ts and to_ts:
            params['from'] = int(from_ts)
            params['to'] = int(to_ts)
        try:
            self.logger.info(f"Fetching data for {ticker} with params: {params}")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                self.logger.warning(f"No data returned for {ticker}")
                return None
            if isinstance(data, dict) and 'error' in data:
                self.logger.error(f"API error for {ticker}: {data['error']}")
                return None
            df = pd.DataFrame(data)
            if df.empty:
                self.logger.warning(f"Empty DataFrame for {ticker}")
                return None
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                self.logger.error(f"No datetime column found for {ticker}")
                return None
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing columns for {ticker}: {missing_cols}")
                return None
            df = df.sort_values('datetime').reset_index(drop=True)
            self.logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {ticker}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {ticker}: {e}")
            return None

    def fetch_intraday_data_range(self, ticker, start_date, end_date, max_days_per_request=600):
        """
        Fetches and concatenates data in 600-day chunks.
        """
        all_dfs = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=max_days_per_request), end_date)
            from_ts = int(current_start.timestamp())
            to_ts = int(current_end.timestamp())
            df = self.fetch_intraday_data(ticker, from_ts, to_ts)
            if df is not None and not df.empty:
                all_dfs.append(df)
            current_start = current_end + timedelta(days=1)
        if all_dfs:
            return pd.concat(all_dfs).drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
        else:
            return None

    def validate_bar_data(self, bar):
        try:
            required_fields = ['symbol', 'ts', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in bar:
                    return False
            o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
            if not (l <= o <= h and l <= c <= h):
                self.logger.warning(f"Invalid OHLC for {bar['symbol']} at {bar['ts']}: O={o}, H={h}, L={l}, C={c}")
                return False
            if any(val <= 0 for val in [o, h, l, c]):
                self.logger.warning(f"Non-positive prices for {bar['symbol']} at {bar['ts']}")
                return False
            if bar['volume'] < 0:
                self.logger.warning(f"Negative volume for {bar['symbol']} at {bar['ts']}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating bar data: {e}")
            return False

    async def save_intraday_to_sql(self, ticker, df):
        saved_count = 0
        error_count = 0
        for idx, row in df.iterrows():
            try:
                volume = row['volume']
                if pd.isna(volume):
                    volume = 0
                bar = {
                    'symbol': ticker,
                    'ts': row['datetime'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(volume)
                }
                if not self.validate_bar_data(bar):
                    error_count += 1
                    continue
                await self.db.save_bar(bar)
                saved_count += 1
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error saving bar for {ticker} at {row['datetime']}: {e}")
        self.logger.info(f"Saved {saved_count} bars for {ticker}, {error_count} errors")

    async def check_existing_data(self, ticker):
        try:
            query = """
                    SELECT MIN(ts)  as earliest_date, \
                           MAX(ts)  as latest_date, \
                           COUNT(*) as total_records
                    FROM stock_bars_minute
                    WHERE symbol = $1 \
                    """
            result = await self.db.pool.fetchrow(query, ticker)
            if result and result['total_records'] > 0:
                return {
                    'has_data': True,
                    'earliest_date': result['earliest_date'],
                    'latest_date': result['latest_date'],
                    'total_records': result['total_records']
                }
            else:
                return {'has_data': False}
        except Exception as e:
            self.logger.error(f"Error checking existing data for {ticker}: {e}")
            return {'has_data': False}

    async def delete_all_data_for_ticker(self, ticker):
        try:
            await self.db.connect()
            await self.db.pool.execute("DELETE FROM stock_bars_minute WHERE symbol = $1", ticker)
            print(f"All data deleted for {ticker}")
        except Exception as e:
            print(f"Error deleting data for {ticker}: {e}")
        finally:
            await self.db.close()

    async def download(self, tickers, start_date=None, end_date=None, force_refresh=False):
        # Set default date range: 2 years ago to today
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()

        await self.db.connect()
        print("\n—Download—")
        total_tickers = len(tickers)
        successful_downloads = 0
        failed_downloads = 0
        for i, ticker in enumerate(tickers, 1):
            try:
                self.logger.info(f"Processing ticker {i}/{total_tickers}: {ticker}")
                if not force_refresh:
                    existing_data = await self.check_existing_data(ticker)
                    if existing_data['has_data']:
                        self.logger.info(f"{ticker} already has {existing_data['total_records']} records "
                                         f"from {existing_data['earliest_date']} to {existing_data['latest_date']}")
                # Use fetch_intraday_data_range for 600-day chunks
                df = self.fetch_intraday_data_range(ticker, start_date, end_date, max_days_per_request=600)
                if df is not None and not df.empty:
                    await self.save_intraday_to_sql(ticker, df)
                    unique_days = df['datetime'].dt.date.nunique()
                    total_records = len(df)
                    date_range = f"{df['datetime'].min().date()} to {df['datetime'].max().date()}"
                    print(f"{ticker}: Downloaded {unique_days} days ({total_records} records) ✔")
                    print(f"  Date range: {date_range}")
                    actual_days = (df['datetime'].max() - df['datetime'].min()).days
                    print(f"  Actual span: {actual_days} calendar days")
                    avg_records_per_day = total_records / max(unique_days, 1)
                    print(f"  Avg records/day: {avg_records_per_day:.1f}")
                    successful_downloads += 1
                else:
                    print(f"{ticker}: No data found or error ✗")
                    failed_downloads += 1
            except Exception as e:
                self.logger.error(f"Failed to process {ticker}: {e}")
                print(f"{ticker}: Error - {str(e)} ✗")
                failed_downloads += 1
            if i < total_tickers:
                await asyncio.sleep(0.2)
        print(f"\n—Download Summary—")
        print(f"Total tickers: {total_tickers}")
        print(f"Successful: {successful_downloads}")
        print(f"Failed: {failed_downloads}")
        print(f"Success rate: {successful_downloads / total_tickers * 100:.1f}%")
        await self.db.close()

    async def get_available_symbols(self):
        try:
            query = """
                    SELECT symbol,
                           MIN(ts)  as earliest_date,
                           MAX(ts)  as latest_date,
                           COUNT(*) as total_records
                    FROM stock_bars_minute
                    GROUP BY symbol
                    ORDER BY symbol \
                    """
            await self.db.connect()
            rows = await self.db.pool.fetch(query)
            await self.db.close()
            symbols_info = []
            for row in rows:
                symbols_info.append({
                    'symbol': row['symbol'],
                    'earliest_date': row['earliest_date'],
                    'latest_date': row['latest_date'],
                    'total_records': row['total_records']
                })
            return symbols_info
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    def __del__(self):
        try:
            if hasattr(self, 'db') and self.db:
                pass
        except:
            pass


def download_tickers_from_file(
    ticker_file="TickerList.txt",
    start_date=None,
    end_date=None,
    force_refresh=False
):
    try:
        # Set default date range: 2 years ago to today
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ticker_path = os.path.join(project_root, ticker_file)
        if not os.path.exists(ticker_path):
            ticker_path = os.path.join(os.path.dirname(__file__), ticker_file)
        print(f"DEBUG: Looking for ticker file at: {ticker_path}")
        if not os.path.exists(ticker_path):
            raise FileNotFoundError(f"Ticker file not found: {ticker_path}")
        with open(ticker_path, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith('#')]
        print(f"DEBUG: Tickers loaded: {tickers}")
        if not tickers:
            print("No tickers found in file")
            return

        downloader = MinuteDataDownloader(interval='5m')

        # --- NEW: Ask if user wants to delete all data for these tickers ---
        delete_choice = input(f"Delete ALL data for these tickers before download? (y/n): ").strip().lower()
        if delete_choice == 'y':
            confirm = input("Are you SURE? This cannot be undone! Type 'yes' to confirm: ").strip().lower()
            if confirm == 'yes':
                for ticker in tickers:
                    asyncio.run(downloader.delete_all_data_for_ticker(ticker))
            else:
                print("Delete cancelled.")

        asyncio.run(downloader.download(tickers, start_date, end_date, force_refresh))
    except Exception as e:
        print(f"Error in download_tickers_from_file: {e}")
        raise


if __name__ == "__main__":
    print("MinuteDataDownloader - Available operations:")
    print("1. Download from ticker file (last 2 years)")
    print("2. Download specific tickers")
    print("3. Check available symbols")

    choice = input("Enter choice (1-3): ")

    if choice == "1":
        download_tickers_from_file()
    elif choice == "2":
        tickers = input("Enter tickers (comma-separated): ").split(",")
        tickers = [t.strip().upper() for t in tickers]
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now()
        downloader = MinuteDataDownloader()
        asyncio.run(downloader.download(tickers, start_date, end_date))
    elif choice == "3":
        downloader = MinuteDataDownloader()
        symbols = asyncio.run(downloader.get_available_symbols())
        print(f"\nFound {len(symbols)} symbols with data:")
        for symbol_info in symbols:
            print(f"  {symbol_info['symbol']}: {symbol_info['total_records']} records "
                  f"({symbol_info['earliest_date']} to {symbol_info['latest_date']})")
    else:
        print("Invalid choice")
