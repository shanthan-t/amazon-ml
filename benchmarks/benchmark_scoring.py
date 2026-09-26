"""Benchmark exact candidate scoring strategies against the prepared index."""

from __future__ import annotations

import argparse
from array import array
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import threading
import time

from src.blocking import (
    DEFAULT_MAX_POSTINGS_PER_KEY,
    SOURCE1_BATCH_SIZE,
    _score_query_batch,
    build_block_keys,
)
from src.data_loader import DEFAULT_CHUNK_SIZE, iter_source_chunks
from src.normalization import normalize_country


_WORKER_ROWS = None
_WORKER_INDEX = None
_WORKER_CAP = None
_WORKER_BATCH_SIZE = None
_WORKER_CONNECTION = None
_WORKER_CONNECTION_CAP = None
_WORKER_SUPPLEMENTAL_ROUTES = False


def _proc_snapshot(root_pid: int) -> tuple[dict[int, int], int, int]:
    processes: dict[int, tuple[int, int, int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            raw = Path(entry.path, "stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            pid = int(entry.name)
            parent = int(fields[1])
            cpu_ticks = int(fields[11]) + int(fields[12])
            rss_kib = int(Path(entry.path, "status").read_text().split("VmRSS:")[1].split()[0])
            processes[pid] = (parent, cpu_ticks, rss_kib)
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue

    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _cpu, _rss) in processes.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True

    cpu = {pid: processes[pid][1] for pid in selected if pid in processes}
    rss = sum(processes[pid][2] for pid in selected if pid in processes)
    pss_kib = 0
    for pid in selected:
        try:
            rollup = Path(f"/proc/{pid}/smaps_rollup").read_text()
            pss_kib += int(next(line.split()[1] for line in rollup.splitlines() if line.startswith("Pss:")))
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            pass
    return cpu, rss, pss_kib


def _database_bytes(index_path: str) -> int:
    base = Path(index_path)
    return sum(
        Path(str(base) + suffix).stat().st_size
        for suffix in ("", "-journal", "-wal", "-shm")
        if Path(str(base) + suffix).exists()
    )


def _physical_core_count() -> int | None:
    cores = set()
    for topology in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology"):
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
            cores.add((package, core))
        except (FileNotFoundError, PermissionError):
            continue
    return len(cores) or None


class _ProcessMeter:
    def __init__(self, index_path: str, interval: float = 0.25):
        self.index_path = index_path
        self.interval = interval
        self.root_pid = os.getpid()
        self.stop_event = threading.Event()
        self.previous_cpu: dict[int, int] = {}
        self.cpu_ticks = 0
        self.peak_rss_kib = 0
        self.peak_pss_kib = 0
        self.peak_db_bytes = _database_bytes(index_path)
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self.previous_cpu, rss, pss = _proc_snapshot(self.root_pid)
        self.peak_rss_kib = rss
        self.peak_pss_kib = pss
        self.thread.start()
        return self

    def _sample_once(self) -> None:
        current, rss, pss = _proc_snapshot(self.root_pid)
        self.cpu_ticks += sum(
            max(0, ticks - self.previous_cpu.get(pid, 0))
            for pid, ticks in current.items()
        )
        self.previous_cpu = current
        self.peak_rss_kib = max(self.peak_rss_kib, rss)
        self.peak_pss_kib = max(self.peak_pss_kib, pss)
        self.peak_db_bytes = max(self.peak_db_bytes, _database_bytes(self.index_path))

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._sample_once()

    def __exit__(self, *_exc) -> None:
        self.stop_event.set()
        self.thread.join()
        self._sample_once()


def _make_batch(raw_rows):
    records = []
    key_rows = []
    for s1_idx, entity_id, name, address, country in raw_rows:
        normalized_country = normalize_country(country)
        records.append((s1_idx, entity_id, normalized_country))
        key_rows.extend(
            (s1_idx, key)
            for key in build_block_keys(name, address, normalized_country)
        )
    return records, key_rows


def _score_range(
    connection: sqlite3.Connection,
    raw_rows: list[tuple[int, str, str, str, str]],
    batch_size: int,
    max_postings_per_key: int,
) -> dict[str, object]:
    counts = {name: array("I") for name in ("ALL", "S2", "S3")}
    hits = {"S2": 0, "S3": 0}
    country_hits: dict[str, dict[str, int]] = {}
    for offset in range(0, len(raw_rows), batch_size):
        batch = raw_rows[offset : offset + batch_size]
        records, key_rows = _make_batch(batch)
        _score_query_batch(
            connection,
            records,
            key_rows,
            max_postings_per_key,
            counts,
            hits,
            country_hits,
            None,
        )
    return {
        "counts": {key: values.tolist() for key, values in counts.items()},
        "hits": hits,
        "country_hits": country_hits,
    }


def _open_readonly_with_temp_keys(
    index_path: str, supplemental_index: str | None = None
) -> sqlite3.Connection:
    uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    if supplemental_index:
        supplemental_uri = Path(supplemental_index).resolve().as_uri() + "?mode=ro"
        connection.execute(
            "ATTACH DATABASE ? AS supplemental", (supplemental_uri,)
        )
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TEMP TABLE query_keys ("
        "s1_idx INTEGER NOT NULL, block_key TEXT NOT NULL, "
        "PRIMARY KEY (s1_idx, block_key)) WITHOUT ROWID"
    )
    return connection


def _worker_init(raw_rows, index_path: str, cap: int, batch_size: int) -> None:
    global _WORKER_ROWS, _WORKER_INDEX, _WORKER_CAP, _WORKER_BATCH_SIZE
    _WORKER_ROWS = raw_rows
    _WORKER_INDEX = index_path
    _WORKER_CAP = cap
    _WORKER_BATCH_SIZE = batch_size


def _score_worker(bounds: tuple[int, int]) -> dict[str, object]:
    start, end = bounds
    connection = _open_readonly_with_temp_keys(_WORKER_INDEX)
    try:
        return _score_range(
            connection,
            _WORKER_ROWS[start:end],
            _WORKER_BATCH_SIZE,
            _WORKER_CAP,
        )
    finally:
        connection.close()


def _worker_init_stream(index_path: str, cap: int) -> None:
    global _WORKER_CONNECTION, _WORKER_CONNECTION_CAP, _WORKER_SUPPLEMENTAL_ROUTES
    _WORKER_CONNECTION = _open_readonly_with_temp_keys(index_path)
    _WORKER_CONNECTION_CAP = cap
    _WORKER_SUPPLEMENTAL_ROUTES = False


def _score_worker_batch(task):
    local_start, raw_rows = task
    records, key_rows = _make_batch(raw_rows)
    counts = {name: array("I") for name in ("ALL", "S2", "S3")}
    hits = {"S2": 0, "S3": 0}
    country_hits: dict[str, dict[str, int]] = {}
    _score_query_batch(
        _WORKER_CONNECTION,
        records,
        key_rows,
        _WORKER_CONNECTION_CAP,
        counts,
        hits,
        country_hits,
        None,
        use_supplemental=_WORKER_SUPPLEMENTAL_ROUTES,
    )
    return local_start, counts, hits, country_hits, len(raw_rows)


def _worker_init_stream_supplemental(
    index_path: str, cap: int, supplemental_index: str | None
) -> None:
    global _WORKER_CONNECTION, _WORKER_CONNECTION_CAP, _WORKER_SUPPLEMENTAL_ROUTES
    _WORKER_CONNECTION = _open_readonly_with_temp_keys(index_path, supplemental_index)
    _WORKER_CONNECTION_CAP = cap
    _WORKER_SUPPLEMENTAL_ROUTES = bool(supplemental_index)


def _run_streamed_workers(
    index_path: str,
    raw_rows,
    workers: int,
    cap: int,
    supplemental_index: str | None = None,
):
    result = {
        "counts": {name: array("I", [0]) * len(raw_rows) for name in ("ALL", "S2", "S3")},
        "hits": {"S2": 0, "S3": 0},
        "country_hits": {},
    }

    def tasks():
        for start in range(0, len(raw_rows), SOURCE1_BATCH_SIZE):
            yield start, raw_rows[start : start + SOURCE1_BATCH_SIZE]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init_stream_supplemental,
        initargs=(index_path, cap, supplemental_index),
    ) as pool:
        for start, counts, hits, country_hits, row_count in pool.imap_unordered(
            _score_worker_batch, tasks(), chunksize=1
        ):
            end = start + row_count
            for name in result["counts"]:
                result["counts"][name][start:end] = counts[name]
            for source in result["hits"]:
                result["hits"][source] += hits[source]
            for country, values in country_hits.items():
                target = result["country_hits"].setdefault(country, {"S2": 0, "S3": 0})
                for source in target:
                    target[source] += values[source]
    result["counts"] = {name: values.tolist() for name, values in result["counts"].items()}
    return result


def _combine_parts(parts: list[dict[str, object]]) -> dict[str, object]:
    result = {"counts": {name: [] for name in ("ALL", "S2", "S3")}, "hits": {"S2": 0, "S3": 0}, "country_hits": {}}
    for part in parts:
        for name in result["counts"]:
            result["counts"][name].extend(part["counts"][name])
        for source in result["hits"]:
            result["hits"][source] += part["hits"][source]
        for country, values in part["country_hits"].items():
            target = result["country_hits"].setdefault(country, {"S2": 0, "S3": 0})
            for source in target:
                target[source] += values[source]
    return result


def _load_sample(path: str, start: int, size: int) -> list[tuple[int, str, str, str, str]]:
    end = start + size
    rows = []
    offset = 0
    for chunk in iter_source_chunks(path, chunk_size=DEFAULT_CHUNK_SIZE):
        chunk_end = offset + len(chunk)
        if chunk_end > start and offset < end:
            first = max(0, start - offset)
            last = min(len(chunk), end - offset)
            for local_idx, values in enumerate(
                chunk.iloc[first:last].itertuples(index=False, name=None), start=first
            ):
                entity_id, name, address, country = values
                rows.append((offset + local_idx, entity_id, name, address, country))
        offset = chunk_end
        if offset >= end:
            break
    if len(rows) != size:
        raise ValueError(f"Requested {size} sample rows, found {len(rows)}")
    return rows


def _ground_truth_denominators(index_path: str, start: int, size: int) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{Path(index_path).resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT CASE substr(g.target_id, 1, 3) "
            "WHEN 'S2-' THEN 'S2' WHEN 'S3-' THEN 'S3' ELSE 'OTHER' END, COUNT(*) "
            "FROM gt_raw AS g JOIN s1 AS s ON s.entity_id = g.s1_id "
            "WHERE s.s1_idx BETWEEN ? AND ? GROUP BY 1",
            (start, start + size - 1),
        ).fetchall()
    finally:
        connection.close()
    denominators = {name: 0 for name in ("S2", "S3", "OTHER")}
    denominators.update(dict(rows))
    denominators["ALL"] = sum(denominators.values())
    return denominators


def _summary(result: dict[str, object], denominators: dict[str, int]) -> dict[str, object]:
    counts = result["counts"]
    hits = result["hits"]
    candidates = {name: sum(values) for name, values in counts.items()}
    hit_total = sum(hits.values())
    recall_by_source = {}
    for source in ("S2", "S3"):
        total = denominators[source]
        recall_by_source[source] = {
            "ground_truth_pairs": total,
            "candidate_hits": hits[source],
            "candidate_recall": hits[source] / total if total else 0.0,
        }
    return {
        "candidate_pairs": candidates,
        "true_match_hits": {**hits, "ALL": hit_total},
        "ground_truth_pairs": denominators,
        "candidate_recall": {
            "ALL": hit_total / denominators["ALL"] if denominators["ALL"] else 0.0,
            "by_source": recall_by_source,
        },
    }


def _run_baseline(index_path: str, raw_rows, cap: int):
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("DELETE FROM query_keys")
    connection.commit()
    try:
        return _score_range(connection, raw_rows, SOURCE1_BATCH_SIZE, cap)
    finally:
        connection.close()


def _run_readonly_batch(index_path: str, raw_rows, batch_size: int, cap: int):
    connection = _open_readonly_with_temp_keys(index_path)
    try:
        return _score_range(connection, raw_rows, batch_size, cap)
    finally:
        connection.close()


def _run_workers(index_path: str, raw_rows, workers: int, cap: int):
    ctx = mp.get_context("spawn")
    boundaries = [round(i * len(raw_rows) / workers) for i in range(workers + 1)]
    ranges = [(boundaries[i], boundaries[i + 1]) for i in range(workers)]
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(raw_rows, index_path, cap, SOURCE1_BATCH_SIZE),
    ) as pool:
        parts = pool.map(_score_worker, ranges)
    return _combine_parts(parts)


def _measure(label: str, index_path: str, function, denominators):
    before_db = Path(index_path).stat().st_size
    with _ProcessMeter(index_path) as meter:
        started = time.perf_counter()
        result = function()
        wall_seconds = time.perf_counter() - started
    rows = len(result["counts"]["ALL"])
    cpu_seconds = meter.cpu_ticks / os.sysconf("SC_CLK_TCK")
    summary = _summary(result, denominators)
    return {
        "name": label,
        "wall_seconds": round(wall_seconds, 3),
        "rows": rows,
        "rows_per_second": round(rows / wall_seconds, 1) if wall_seconds else 0.0,
        "cpu_seconds": round(cpu_seconds, 3),
        "cpu_cores_used_average": round(cpu_seconds / wall_seconds, 2) if wall_seconds else 0.0,
        "cpu_percent_of_logical_cpus": round(
            100 * cpu_seconds / wall_seconds / (os.cpu_count() or 1), 1
        ) if wall_seconds else 0.0,
        "peak_process_tree_rss_mib": round(meter.peak_rss_kib / 1024, 1),
        "peak_process_tree_pss_mib": round(meter.peak_pss_kib / 1024, 1),
        "database_bytes_before": before_db,
        "database_bytes_after": Path(index_path).stat().st_size,
        "database_bytes_delta": Path(index_path).stat().st_size - before_db,
        "database_and_sidecar_peak_delta_bytes": meter.peak_db_bytes - before_db,
        "exact_match_to_baseline": label == "sqlite-current-1000",
        **summary,
        "_result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--prepared-index", default="phase2_index_resume.sqlite3")
    parser.add_argument("--sample-start", type=int, default=400_000)
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--max-postings-per-key", type=int, default=DEFAULT_MAX_POSTINGS_PER_KEY)
    parser.add_argument("--report", default="reports/scoring_benchmark.json")
    args = parser.parse_args()

    raw_rows = _load_sample(args.source1, args.sample_start, args.sample_rows)
    denominators = _ground_truth_denominators(
        args.prepared_index, args.sample_start, args.sample_rows
    )
    base = _measure(
        "sqlite-current-1000",
        args.prepared_index,
        lambda: _run_baseline(args.prepared_index, raw_rows, args.max_postings_per_key),
        denominators,
    )
    expected = base["_result"]
    runs = [base]

    variants = [
        ("sqlite-readonly-batch-5000", lambda: _run_readonly_batch(args.prepared_index, raw_rows, 5_000, args.max_postings_per_key)),
        ("sqlite-readonly-batch-50000", lambda: _run_readonly_batch(args.prepared_index, raw_rows, len(raw_rows), args.max_postings_per_key)),
    ]
    variants.extend(
        (f"sqlite-readonly-workers-{workers}", lambda workers=workers: _run_workers(args.prepared_index, raw_rows, workers, args.max_postings_per_key))
        for workers in (2, 4, 8)
    )
    variants.extend(
        (f"sqlite-readonly-stream-workers-{workers}", lambda workers=workers: _run_streamed_workers(args.prepared_index, raw_rows, workers, args.max_postings_per_key))
        for workers in (2, 4, 8)
    )
    for name, function in variants:
        measured = _measure(name, args.prepared_index, function, denominators)
        measured["exact_match_to_baseline"] = measured["_result"] == expected
        if not measured["exact_match_to_baseline"]:
            raise RuntimeError(f"{name} does not match the current scorer exactly")
        runs.append(measured)

    for run in runs:
        run.pop("_result")
    report = {
        "sample": {
            "source1_path": str(Path(args.source1).resolve()),
            "prepared_index": str(Path(args.prepared_index).resolve()),
            "start_row_zero_based": args.sample_start,
            "rows": args.sample_rows,
            "max_postings_per_key_per_source": args.max_postings_per_key,
            "workers_available": os.cpu_count(),
            "cpu_cores_physical": _physical_core_count(),
        },
        "measurement_notes": {
            "rss": "Process-tree RSS sums process RSS and can double-count shared pages; PSS is reported as a physical-memory estimate.",
            "disk": "Tracks the prepared SQLite database plus its journal/WAL/SHM sidecars; temporary query-key tables are per-connection in memory.",
            "multiprocessing": "Workers use the spawn start method and read-only SQLite connections; streamed-worker runs include task serialization.",
            "exactness": "Every variant asserts identical per-S1 candidate counts, total/source match hits, and country match hits against the current scorer.",
        },
        "runs": runs,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
