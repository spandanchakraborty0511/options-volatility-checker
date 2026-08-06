"""
Build Sampled Demo SQLite Database for Vercel Cloud Deployment.
Extracts 60 representative trading dates across 2010-2023 into demo_options.db (~15 MB)
so the repository can be pushed to GitHub and deployed live to Vercel without exceeding file size limits.
"""

import sqlite3
import os
import time

def build_demo_database():
    source_db = r"D:\Volatality_checker\sp500_data.db"
    target_db = r"D:\Volatality_checker\demo_options.db"
    
    if not os.path.exists(source_db):
        print(f"Source DB {source_db} not found!")
        return
        
    print("Building demo_options.db for Vercel deployment...", flush=True)
    start_time = time.time()
    
    src_conn = sqlite3.connect(source_db)
    
    # Pick 4-5 dates per year (2010 through 2023)
    sample_dates = []
    for yr in range(2010, 2024):
        cursor = src_conn.execute(
            "SELECT DISTINCT date FROM sp500_options WHERE date LIKE ? ORDER BY date ASC LIMIT 5",
            (f"{yr}-%",)
        )
        sample_dates.extend([r[0] for r in cursor.fetchall()])
        
    print(f"Selected {len(sample_dates)} representative trading dates across 2010-2023.", flush=True)
    
    if os.path.exists(target_db):
        os.remove(target_db)
        
    tgt_conn = sqlite3.connect(target_db)
    tgt_conn.execute("""
        CREATE TABLE sp500_options (
            date          TEXT,
            expiry        TEXT,
            strike        REAL,
            option_type   TEXT,
            underlying    REAL,
            price         REAL,
            bid           REAL,
            ask           REAL,
            volume        REAL,
            iv            REAL,
            dte           INTEGER,
            PRIMARY KEY (date, expiry, strike, option_type)
        )
    """)
    
    # Insert rows for sampled dates
    for date_str in sample_dates:
        rows = src_conn.execute(
            "SELECT date, expiry, strike, option_type, underlying, price, bid, ask, volume, iv, dte FROM sp500_options WHERE date = ?",
            (date_str,)
        ).fetchall()
        
        tgt_conn.executemany(
            "INSERT INTO sp500_options VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        
    tgt_conn.commit()
    
    # Create indexes for instant querying
    tgt_conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON sp500_options(date)")
    tgt_conn.commit()
    
    size_mb = os.path.getsize(target_db) / (1024 * 1024)
    print(f"demo_options.db successfully created ({size_mb:.2f} MB) in {time.time() - start_time:.2f}s", flush=True)

if __name__ == "__main__":
    build_demo_database()
