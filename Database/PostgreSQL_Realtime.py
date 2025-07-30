# Database/PostgreSQL_Realtime.py

from Utils.Config_Loader import load_config
import asyncpg
from datetime import datetime, timezone

config = load_config("config.txt")
PG_DSN = config.get("POSTGRES_DSN")
print("DEBUG: PG_DSN =", PG_DSN)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_ticks_realtime (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    price NUMERIC NOT NULL,
    volume INTEGER,
    conditions TEXT,
    dark_pool BOOLEAN,
    market_status TEXT,
    ts TIMESTAMP NOT NULL
);
"""

INSERT_SQL = """
INSERT INTO stock_ticks_realtime (symbol, price, volume, conditions, dark_pool, market_status, ts)
VALUES ($1, $2, $3, $4, $5, $6, $7)
"""

class PostgresRealtimeDB:
    def __init__(self, dsn=PG_DSN):
        self.dsn = dsn
        self.pool = None

    async def connect(self):
        print(f"DEBUG: Connecting to DB with DSN: {self.dsn}")
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)

    async def save_tick(self, tick):
        try:
            symbol = tick.get('s')
            price = float(tick.get('p'))
            volume = int(tick.get('v', 0))
            conditions = ','.join(map(str, tick.get('c', []))) if isinstance(tick.get('c'), list) else str(tick.get('c'))
            dark_pool = bool(tick.get('dp', False))
            market_status = tick.get('ms', None)
            ts = datetime.fromtimestamp(tick['t'] / 1000, tz=timezone.utc)
            async with self.pool.acquire() as conn:
                await conn.execute(INSERT_SQL, symbol, price, volume, conditions, dark_pool, market_status, ts)
        except Exception as e:
            print(f"Error saving tick: {e}")

    async def close(self):
        if self.pool:
            await self.pool.close()
