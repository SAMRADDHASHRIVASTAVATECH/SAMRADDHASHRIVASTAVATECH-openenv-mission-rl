"""
data_loader.py
--------------
Handles loading data from SQLite databases and JSON files.
Extracts schema and records dynamically — no hardcoded column names.
"""

import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Tuple


class SQLiteLoader:
    """Load and inspect any SQLite database dynamically."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self.tables: List[str] = []
        self.schema: Dict[str, List[Dict[str, str]]] = {}
        self.data: Dict[str, List[Dict[str, Any]]] = {}

    def connect(self) -> bool:
        """Open connection to the SQLite file."""
        if not os.path.isfile(self.db_path):
            print(f"[SQLiteLoader] File not found: {self.db_path}")
            return False
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            print(f"[SQLiteLoader] Connected to: {self.db_path}")
            return True
        except sqlite3.Error as exc:
            print(f"[SQLiteLoader] Connection error: {exc}")
            return False

    def discover_tables(self) -> List[str]:
        """Find every user table in the database."""
        if not self.connection:
            return []
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%';"
        )
        self.tables = [row[0] for row in cursor.fetchall()]
        print(f"[SQLiteLoader] Discovered tables: {self.tables}")
        return self.tables

    def discover_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Read column info for every discovered table."""
        if not self.connection:
            return {}
        self.schema = {}
        cursor = self.connection.cursor()
        for table in self.tables:
            cursor.execute(f"PRAGMA table_info('{table}');")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "notnull": row[3],
                    "default": row[4],
                    "pk": row[5],
                })
            self.schema[table] = columns
        print(f"[SQLiteLoader] Schema loaded for {len(self.schema)} table(s)")
        return self.schema

    def load_records(self, limit: int = 500) -> Dict[str, List[Dict[str, Any]]]:
        """Pull rows from every table (up to *limit* per table)."""
        if not self.connection:
            return {}
        self.data = {}
        cursor = self.connection.cursor()
        for table in self.tables:
            try:
                cursor.execute(f"SELECT * FROM '{table}' LIMIT {limit};")
                col_names = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                records = []
                for row in rows:
                    record = {}
                    for idx, col in enumerate(col_names):
                        record[col] = row[idx]
                    records.append(record)
                self.data[table] = records
                print(
                    f"[SQLiteLoader] Table '{table}': "
                    f"{len(records)} record(s) loaded"
                )
            except sqlite3.Error as exc:
                print(f"[SQLiteLoader] Error reading '{table}': {exc}")
                self.data[table] = []
        return self.data

    def close(self):
        if self.connection:
            self.connection.close()
            print("[SQLiteLoader] Connection closed")

    # ---- convenience -------------------------------------------------- #

    def load_all(self, limit: int = 500) -> Tuple[
        Dict[str, List[Dict[str, str]]],
        Dict[str, List[Dict[str, Any]]],
    ]:
        """One-call helper: connect → discover → load → close."""
        if not self.connect():
            return {}, {}
        self.discover_tables()
        self.discover_schema()
        self.load_records(limit)
        self.close()
        return self.schema, self.data

    def get_all_records_flat(self) -> List[Dict[str, Any]]:
        """Return every record from every table in one flat list."""
        flat: List[Dict[str, Any]] = []
        for table, records in self.data.items():
            for rec in records:
                tagged = dict(rec)
                tagged["__source_table__"] = table
                flat.append(tagged)
        return flat

    def get_column_names(self) -> List[str]:
        """Unique column names across all tables."""
        names = set()
        for cols in self.schema.values():
            for col in cols:
                names.add(col["name"].lower())
        return sorted(names)


class JSONLoader:
    """Load and inspect a JSON file dynamically."""

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.raw_data: Any = None
        self.records: List[Dict[str, Any]] = []
        self.keys: List[str] = []

    def load(self) -> bool:
        if not os.path.isfile(self.json_path):
            print(f"[JSONLoader] File not found: {self.json_path}")
            return False
        try:
            with open(self.json_path, "r", encoding="utf-8") as fh:
                self.raw_data = json.load(fh)
            print(f"[JSONLoader] Loaded: {self.json_path}")
            self._normalize()
            return True
        except (json.JSONDecodeError, IOError) as exc:
            print(f"[JSONLoader] Load error: {exc}")
            return False

    # ---- internal ----------------------------------------------------- #

    def _normalize(self):
        """
        Turn whatever JSON shape we get into a flat list of dicts.
        Supports: list-of-dicts, dict-of-dicts, single dict,
        and nested structures with a top-level list field.
        """
        data = self.raw_data

        if isinstance(data, list):
            self.records = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            # check if there is a top-level key whose value is a list
            list_key = None
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0:
                    list_key = k
                    break
            if list_key:
                self.records = [
                    r for r in data[list_key] if isinstance(r, dict)
                ]
            else:
                # dict-of-dicts  e.g. {"obj1": {...}, "obj2": {...}}
                nested = []
                for k, v in data.items():
                    if isinstance(v, dict):
                        entry = dict(v)
                        entry["__key__"] = k
                        nested.append(entry)
                if nested:
                    self.records = nested
                else:
                    # single flat dict → one record
                    self.records = [data]
        else:
            self.records = []

        # collect all keys
        key_set: set = set()
        for rec in self.records:
            key_set.update(rec.keys())
        self.keys = sorted(key_set)
        print(
            f"[JSONLoader] {len(self.records)} record(s), "
            f"{len(self.keys)} unique key(s)"
        )

    def get_records(self) -> List[Dict[str, Any]]:
        return self.records

    def get_keys(self) -> List[str]:
        return self.keys


class DataManager:
    """
    Facade that combines SQLite + JSON loaders and presents a unified
    list of raw records to the task builder.
    """

    def __init__(self):
        self.sqlite_loader: Optional[SQLiteLoader] = None
        self.json_loader: Optional[JSONLoader] = None
        self.all_records: List[Dict[str, Any]] = []
        self.all_column_names: List[str] = []

    def load_sqlite(self, path: str) -> bool:
        self.sqlite_loader = SQLiteLoader(path)
        schema, data = self.sqlite_loader.load_all()
        return bool(data)

    def load_json(self, path: str) -> bool:
        self.json_loader = JSONLoader(path)
        return self.json_loader.load()

    def merge(self):
        """Merge records from both sources into one list."""
        self.all_records = []
        names: set = set()

        if self.sqlite_loader and self.sqlite_loader.data:
            flat = self.sqlite_loader.get_all_records_flat()
            self.all_records.extend(flat)
            names.update(self.sqlite_loader.get_column_names())

        if self.json_loader and self.json_loader.records:
            self.all_records.extend(self.json_loader.get_records())
            names.update(k.lower() for k in self.json_loader.get_keys())

        self.all_column_names = sorted(names)
        print(
            f"[DataManager] Merged {len(self.all_records)} record(s), "
            f"{len(self.all_column_names)} unique column(s)"
        )

    def get_records(self) -> List[Dict[str, Any]]:
        return self.all_records

    def get_column_names(self) -> List[str]:
        return self.all_column_names

    def summary(self) -> str:
        lines = [
            "=== Data Manager Summary ===",
            f"Total records : {len(self.all_records)}",
            f"Column names  : {self.all_column_names[:20]}",
        ]
        if self.sqlite_loader:
            lines.append(
                f"SQLite tables : {self.sqlite_loader.tables}"
            )
        if self.json_loader:
            lines.append(
                f"JSON keys     : {self.json_loader.keys[:20]}"
            )
        return "\n".join(lines)