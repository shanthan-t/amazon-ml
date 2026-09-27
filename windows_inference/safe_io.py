"""Atomic publication tolerant of transient Windows reader/antivirus locks."""
import json
import os
from pathlib import Path
import time


def atomic(path, value, timeout=180):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2) + '\n')
        f.flush()
        os.fsync(f.fileno())
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
