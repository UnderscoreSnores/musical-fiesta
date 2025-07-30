# Database/PostgreSQL_Minute.py

from Utils.Config_Loader import load_config
import asyncpg

config = load_config("config.txt")
PG_DSN = "postgresql://myuser:COOLMATH1!@localhost:5432/tradingdb"
print("DEBUG: PG_DSN =", PG_DSN)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_bars_minute (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    ts TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER
);
"""

INSERT_SQL = """
INSERT INTO stock_bars_minute (symbol, ts, open, high, low, close, volume)
VALUES ($1, $2, $3, $4, $5, $6, $7)
"""

class PostgresMinuteDB:
    def __init__(self, dsn=PG_DSN):
        self.dsn = dsn
        self.pool = None

    async def connect(self):
        print(f"DEBUG: Connecting to DB with DSN: {self.dsn}")
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)

    async def save_bar(self, bar):
        async with self.pool.acquire() as conn:
            await conn.execute(
                INSERT_SQL,
                bar['symbol'],
                bar['ts'],
                bar['open'],
                bar['high'],
                bar['low'],
                bar['close'],
                bar['volume']
            )

    async def close(self):
        if self.pool:
            await self.pool.close()
