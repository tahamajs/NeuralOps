#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""SQLite Backend for Master Ledger Storage and Cryptographic Provenance."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SQLiteMasterLedger:
    """Unified interface for SQLite master ledger with SHA-256 provenance."""

    def __init__(self, db_path: str = "results/master_ledger.sqlite") -> None:
        self.db_path = Path(db_path)
        if not str(self.db_path) == ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize master ledger schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware_telemetry (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_name TEXT,
                compute_capability TEXT,
                vram_total_gb REAL,
                driver_version TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS master_benchmark_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                architecture_name TEXT,
                compression_method TEXT,
                batch_size INTEGER,
                seq_len INTEGER,
                latency_ms REAL,
                throughput_tokens_per_sec REAL,
                peak_vram_mb REAL,
                accuracy_pct REAL,
                sha256_provenance TEXT,
                FOREIGN KEY (run_id) REFERENCES hardware_telemetry (run_id)
            )
        """)
        self.conn.commit()

    def record_run(
        self,
        device_name: str,
        compute_capability: str,
        vram_total_gb: float,
        driver_version: str,
    ) -> int:
        """Insert hardware run record and return run_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO hardware_telemetry (device_name, compute_capability, vram_total_gb, driver_version) VALUES (?, ?, ?, ?)",
            (device_name, compute_capability, vram_total_gb, driver_version),
        )
        self.conn.commit()
        return cursor.lastrowid

    def record_result(
        self,
        run_id: int,
        architecture_name: str,
        compression_method: str,
        batch_size: int,
        seq_len: int,
        latency_ms: float,
        throughput_tokens_per_sec: float,
        peak_vram_mb: float,
        accuracy_pct: float,
    ) -> Tuple[int, str]:
        """Insert benchmark result with deterministic SHA-256 hash."""
        raw_payload = f"{architecture_name}:{compression_method}:{batch_size}:{seq_len}:{latency_ms:.4f}:{accuracy_pct:.2f}"
        sha_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO master_benchmark_results (
                run_id, architecture_name, compression_method, batch_size, seq_len,
                latency_ms, throughput_tokens_per_sec, peak_vram_mb, accuracy_pct, sha256_provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, architecture_name, compression_method, batch_size, seq_len,
            latency_ms, throughput_tokens_per_sec, peak_vram_mb, accuracy_pct, sha_hash
        ))
        self.conn.commit()
        return cursor.lastrowid, sha_hash

    def close(self) -> None:
        self.conn.close()
