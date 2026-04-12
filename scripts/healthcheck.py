"""Docker health check — verifies trading loop heartbeat and DB access."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

# Max age of heartbeat before considering the loop stalled (seconds)
HEARTBEAT_MAX_AGE = 300  # 5 minutes


def check_heartbeat() -> bool:
    """Check that the heartbeat file is fresh."""
    hb = Path('data/.heartbeat')
    if not hb.exists():
        # Heartbeat file may not exist on first startup
        print('WARN: No heartbeat file yet')
        return True

    try:
        ts = float(hb.read_text().strip())
        age = time.time() - ts
        if age > HEARTBEAT_MAX_AGE:
            print(f'FAIL: Heartbeat stale — {age:.0f}s old (max {HEARTBEAT_MAX_AGE}s)')
            return False
        return True
    except (ValueError, OSError) as e:
        print(f'FAIL: Cannot read heartbeat — {e}')
        return False


def check_database() -> bool:
    """Check that at least one SQLite DB is readable."""
    for name in ('nct_paper.sqlite', 'nct_live.sqlite'):
        db_path = Path('data') / name
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=5)
                conn.execute('SELECT 1')
                conn.close()
                return True
            except sqlite3.Error as e:
                print(f'FAIL: DB {name} unreadable — {e}')
                return False

    # No DB yet is OK on first startup
    print('WARN: No database files yet')
    return True


def main() -> int:
    ok = True

    if not check_heartbeat():
        ok = False

    if not check_database():
        ok = False

    if ok:
        print('OK')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
