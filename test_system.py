"""
test_system.py
--------------
BRUTAL HACKATHON JUDGE TEST SUITE

This test suite simulates the most demanding judge evaluation.
Covers: Basic → Intermediate → Advanced → Edge Cases → Stress Tests

Total: 150+ tests covering every possible scenario.

Run with: python test_system.py
"""

import sys
import os
import json
import sqlite3
import tempfile
import random
import unittest
import copy
import time
import hashlib
from typing import Dict, List, Any, Tuple
from unittest.mock import patch, MagicMock

# ---- Ensure project root is on path ---- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import SQLiteLoader, JSONLoader, DataManager
from task_builder import build_tasks, _derive_task_from_record
from evaluator import evaluate_action
from rewards import compute_reward, reward_summary
from env import SpaceMissionEnv


# ================================================================== #
#  HELPER FUNCTIONS FOR TEST FIXTURES
# ================================================================== #

def create_temp_sqlite(records: List[Dict[str, Any]], table: str = "objects") -> str:
    """Create a temporary SQLite database with given records."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    if records:
        cols = list(records[0].keys())
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        conn.execute(f'CREATE TABLE "{table}" ({col_defs});')
        placeholders = ", ".join("?" for _ in cols)
        for rec in records:
            vals = [rec.get(c) for c in cols]
            conn.execute(f'INSERT INTO "{table}" VALUES ({placeholders});', vals)
        conn.commit()
    conn.close()
    return path


def create_temp_sqlite_multi_table(tables_data: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create SQLite with multiple tables."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    for table_name, records in tables_data.items():
        if records:
            cols = list(records[0].keys())
            col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
            conn.execute(f'CREATE TABLE "{table_name}" ({col_defs});')
            placeholders = ", ".join("?" for _ in cols)
            for rec in records:
                vals = [rec.get(c) for c in cols]
                conn.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders});', vals)
    conn.commit()
    conn.close()
    return path


def create_temp_json(data: Any) -> str:
    """Write data as JSON to a temp file."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def create_temp_file_with_content(content: str, suffix: str = ".json") -> str:
    """Create temp file with raw string content."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def cleanup_file(path: str):
    """Safely remove a temp file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ================================================================== #
#  SECTION 1: EVALUATOR TESTS (25 tests)
# ================================================================== #

class TestEvaluator_Basic(unittest.TestCase):
    """Basic evaluator functionality."""

    def test_perfect_match_all_zeros(self):
        """All zeros should match."""
        result = evaluate_action((0, 0, 0), {"category": 0, "priority": 0, "decision": 0})
        self.assertEqual(result["score"], 3)

    def test_perfect_match_all_ones(self):
        """All ones should match."""
        result = evaluate_action((1, 1, 1), {"category": 1, "priority": 1, "decision": 1})
        self.assertEqual(result["score"], 3)

    def test_perfect_match_all_twos(self):
        """All twos should match."""
        result = evaluate_action((2, 2, 2), {"category": 2, "priority": 2, "decision": 2})
        self.assertEqual(result["score"], 3)

    def test_perfect_match_mixed(self):
        """Mixed values should match."""
        result = evaluate_action((0, 2, 1), {"category": 0, "priority": 2, "decision": 1})
        self.assertEqual(result["score"], 3)

    def test_no_match_complete(self):
        """Complete mismatch."""
        result = evaluate_action((0, 0, 0), {"category": 1, "priority": 1, "decision": 1})
        self.assertEqual(result["score"], 0)

    def test_no_match_opposite(self):
        """Opposite values."""
        result = evaluate_action((2, 2, 2), {"category": 0, "priority": 0, "decision": 0})
        self.assertEqual(result["score"], 0)


class TestEvaluator_PartialMatches(unittest.TestCase):
    """Partial match scenarios."""

    def test_only_category_matches(self):
        result = evaluate_action((0, 1, 1), {"category": 0, "priority": 2, "decision": 2})
        self.assertEqual(result["score"], 1)
        self.assertTrue(result["category_match"])
        self.assertFalse(result["priority_match"])
        self.assertFalse(result["decision_match"])

    def test_only_priority_matches(self):
        result = evaluate_action((1, 0, 1), {"category": 2, "priority": 0, "decision": 2})
        self.assertEqual(result["score"], 1)
        self.assertFalse(result["category_match"])
        self.assertTrue(result["priority_match"])
        self.assertFalse(result["decision_match"])

    def test_only_decision_matches(self):
        result = evaluate_action((1, 1, 0), {"category": 2, "priority": 2, "decision": 0})
        self.assertEqual(result["score"], 1)
        self.assertFalse(result["category_match"])
        self.assertFalse(result["priority_match"])
        self.assertTrue(result["decision_match"])

    def test_category_and_priority_match(self):
        result = evaluate_action((1, 2, 0), {"category": 1, "priority": 2, "decision": 1})
        self.assertEqual(result["score"], 2)
        self.assertTrue(result["category_match"])
        self.assertTrue(result["priority_match"])
        self.assertFalse(result["decision_match"])

    def test_category_and_decision_match(self):
        result = evaluate_action((2, 0, 1), {"category": 2, "priority": 1, "decision": 1})
        self.assertEqual(result["score"], 2)
        self.assertTrue(result["category_match"])
        self.assertFalse(result["priority_match"])
        self.assertTrue(result["decision_match"])

    def test_priority_and_decision_match(self):
        result = evaluate_action((0, 1, 2), {"category": 2, "priority": 1, "decision": 2})
        self.assertEqual(result["score"], 2)
        self.assertFalse(result["category_match"])
        self.assertTrue(result["priority_match"])
        self.assertTrue(result["decision_match"])


class TestEvaluator_AllCombinations(unittest.TestCase):
    """Test all 27 possible action combinations against a fixed expected."""

    def test_all_27_combinations(self):
        """Verify score calculation for every possible action."""
        expected = {"category": 1, "priority": 1, "decision": 1}
        results = []
        for c in range(3):
            for p in range(3):
                for d in range(3):
                    result = evaluate_action((c, p, d), expected)
                    expected_score = (c == 1) + (p == 1) + (d == 1)
                    self.assertEqual(
                        result["score"], expected_score,
                        f"Action ({c},{p},{d}) should score {expected_score}"
                    )
                    results.append(result["score"])
        
        # Verify distribution: 1 perfect, 6 two-match, 12 one-match, 8 zero-match
        self.assertEqual(results.count(3), 1)
        self.assertEqual(results.count(2), 6)
        self.assertEqual(results.count(1), 12)
        self.assertEqual(results.count(0), 8)


class TestEvaluator_OutputFormat(unittest.TestCase):
    """Test the output dictionary format."""

    def test_result_contains_score(self):
        result = evaluate_action((0, 0, 0), {"category": 0, "priority": 0, "decision": 0})
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], int)

    def test_result_contains_all_match_flags(self):
        result = evaluate_action((0, 0, 0), {"category": 0, "priority": 0, "decision": 0})
        self.assertIn("category_match", result)
        self.assertIn("priority_match", result)
        self.assertIn("decision_match", result)
        self.assertIsInstance(result["category_match"], bool)
        self.assertIsInstance(result["priority_match"], bool)
        self.assertIsInstance(result["decision_match"], bool)

    def test_result_contains_details(self):
        result = evaluate_action((0, 0, 0), {"category": 0, "priority": 0, "decision": 0})
        self.assertIn("details", result)
        self.assertIsInstance(result["details"], str)

    def test_details_shows_checkmarks_for_matches(self):
        result = evaluate_action((1, 1, 1), {"category": 1, "priority": 1, "decision": 1})
        self.assertIn("✓", result["details"])

    def test_details_shows_x_for_mismatches(self):
        result = evaluate_action((0, 0, 0), {"category": 1, "priority": 1, "decision": 1})
        self.assertIn("✗", result["details"])

    def test_score_range(self):
        """Score must always be 0-3."""
        for _ in range(100):
            action = (random.randint(0, 2), random.randint(0, 2), random.randint(0, 2))
            expected = {
                "category": random.randint(0, 2),
                "priority": random.randint(0, 2),
                "decision": random.randint(0, 2)
            }
            result = evaluate_action(action, expected)
            self.assertIn(result["score"], [0, 1, 2, 3])


# ================================================================== #
#  SECTION 2: REWARDS TESTS (20 tests)
# ================================================================== #

class TestRewards_BaseValues(unittest.TestCase):
    """Test base reward values without scaling."""

    def test_score_3_gives_plus_10(self):
        self.assertEqual(compute_reward(3, 1, False), 10.0)

    def test_score_2_gives_plus_5(self):
        self.assertEqual(compute_reward(2, 1, False), 5.0)

    def test_score_1_gives_plus_2(self):
        self.assertEqual(compute_reward(1, 1, False), 2.0)

    def test_score_0_gives_minus_3(self):
        self.assertEqual(compute_reward(0, 1, False), -3.0)

    def test_invalid_score_negative(self):
        """Negative scores should default to -3."""
        self.assertEqual(compute_reward(-1, 1, False), -3.0)

    def test_invalid_score_high(self):
        """Scores > 3 should default to -3."""
        self.assertEqual(compute_reward(99, 1, False), -3.0)


class TestRewards_DifficultyScaling(unittest.TestCase):
    """Test difficulty scaling functionality."""

    def test_difficulty_1_multiplier(self):
        """Difficulty 1 → multiplier 1/3 ≈ 0.333"""
        reward = compute_reward(3, 1, True)
        expected = round(10 * (1/3), 2)
        self.assertAlmostEqual(reward, expected, places=1)

    def test_difficulty_3_multiplier(self):
        """Difficulty 3 → multiplier 1.0"""
        reward = compute_reward(3, 3, True)
        self.assertAlmostEqual(reward, 10.0, places=1)

    def test_difficulty_5_multiplier(self):
        """Difficulty 5 → multiplier 5/3 ≈ 1.667"""
        reward = compute_reward(3, 5, True)
        expected = round(10 * (5/3), 2)
        self.assertAlmostEqual(reward, expected, places=1)

    def test_negative_reward_scales_correctly(self):
        """Negative rewards should also scale."""
        reward = compute_reward(0, 5, True)
        expected = round(-3 * (5/3), 2)
        self.assertAlmostEqual(reward, expected, places=1)

    def test_scaling_disabled_ignores_difficulty(self):
        """When scaling disabled, difficulty shouldn't matter."""
        r1 = compute_reward(3, 1, False)
        r2 = compute_reward(3, 5, False)
        self.assertEqual(r1, r2)
        self.assertEqual(r1, 10.0)

    def test_zero_difficulty_no_crash(self):
        """Zero difficulty shouldn't cause division error."""
        reward = compute_reward(3, 0, True)
        self.assertIsInstance(reward, float)


class TestRewards_Summary(unittest.TestCase):
    """Test reward summary string generation."""

    def test_summary_contains_score(self):
        s = reward_summary(3, 4, 13.33)
        self.assertIn("3", s)

    def test_summary_contains_difficulty(self):
        s = reward_summary(3, 4, 13.33)
        self.assertIn("4", s)

    def test_summary_contains_reward(self):
        s = reward_summary(3, 4, 13.33)
        self.assertIn("13.33", s)

    def test_summary_is_string(self):
        s = reward_summary(2, 3, 5.0)
        self.assertIsInstance(s, str)

    def test_summary_positive_reward_format(self):
        s = reward_summary(3, 3, 10.0)
        self.assertIn("+", s)

    def test_summary_negative_reward_format(self):
        s = reward_summary(0, 3, -3.0)
        self.assertIn("-", s)


# ================================================================== #
#  SECTION 3: SQLITE LOADER TESTS (25 tests)
# ================================================================== #

class TestSQLiteLoader_Connection(unittest.TestCase):
    """Test database connection functionality."""

    def test_connect_valid_file(self):
        path = create_temp_sqlite([{"x": "1"}])
        try:
            loader = SQLiteLoader(path)
            self.assertTrue(loader.connect())
            loader.close()
        finally:
            cleanup_file(path)

    def test_connect_missing_file(self):
        loader = SQLiteLoader("/nonexistent/path/to/database.db")
        self.assertFalse(loader.connect())

    def test_connect_invalid_path(self):
        loader = SQLiteLoader("")
        self.assertFalse(loader.connect())

    def test_connect_directory_path(self):
        loader = SQLiteLoader(tempfile.gettempdir())
        self.assertFalse(loader.connect())

    def test_close_without_connect(self):
        """Close should not crash if never connected."""
        loader = SQLiteLoader("/fake/path.db")
        loader.close()  # Should not raise


class TestSQLiteLoader_TableDiscovery(unittest.TestCase):
    """Test table discovery functionality."""

    def test_discover_single_table(self):
        path = create_temp_sqlite([{"a": "1"}], "test_table")
        try:
            loader = SQLiteLoader(path)
            loader.connect()
            tables = loader.discover_tables()
            self.assertIn("test_table", tables)
            loader.close()
        finally:
            cleanup_file(path)

    def test_discover_multiple_tables(self):
        path = create_temp_sqlite_multi_table({
            "table1": [{"x": "1"}],
            "table2": [{"y": "2"}],
            "table3": [{"z": "3"}]
        })
        try:
            loader = SQLiteLoader(path)
            loader.connect()
            tables = loader.discover_tables()
            self.assertEqual(len(tables), 3)
            self.assertIn("table1", tables)
            self.assertIn("table2", tables)
            self.assertIn("table3", tables)
            loader.close()
        finally:
            cleanup_file(path)

    def test_discover_empty_database(self):
        """Empty database should return empty list."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path)
        conn.close()
        try:
            loader = SQLiteLoader(path)
            loader.connect()
            tables = loader.discover_tables()
            self.assertEqual(tables, [])
            loader.close()
        finally:
            cleanup_file(path)

    def test_excludes_sqlite_internal_tables(self):
        """sqlite_ prefixed tables should be excluded."""
        path = create_temp_sqlite([{"a": "1"}])
        try:
            loader = SQLiteLoader(path)
            loader.connect()
            tables = loader.discover_tables()
            for table in tables:
                self.assertFalse(table.startswith("sqlite_"))
            loader.close()
        finally:
            cleanup_file(path)


class TestSQLiteLoader_SchemaDiscovery(unittest.TestCase):
    """Test schema discovery functionality."""

    def test_discover_schema_single_column(self):
        path = create_temp_sqlite([{"col1": "value"}])
        try:
            loader = SQLiteLoader(path)
            loader.load_all()
            self.assertIn("col1", loader.get_column_names())
        finally:
            cleanup_file(path)

    def test_discover_schema_multiple_columns(self):
        path = create_temp_sqlite([{"a": "1", "b": "2", "c": "3"}])
        try:
            loader = SQLiteLoader(path)
            loader.load_all()
            cols = loader.get_column_names()
            self.assertIn("a", cols)
            self.assertIn("b", cols)
            self.assertIn("c", cols)
        finally:
            cleanup_file(path)

    def test_schema_from_multiple_tables(self):
        path = create_temp_sqlite_multi_table({
            "t1": [{"col_a": "1"}],
            "t2": [{"col_b": "2"}]
        })
        try:
            loader = SQLiteLoader(path)
            loader.load_all()
            cols = loader.get_column_names()
            self.assertIn("col_a", cols)
            self.assertIn("col_b", cols)
        finally:
            cleanup_file(path)


class TestSQLiteLoader_DataLoading(unittest.TestCase):
    """Test data loading functionality."""

    def test_load_single_record(self):
        path = create_temp_sqlite([{"name": "test", "value": "123"}])
        try:
            loader = SQLiteLoader(path)
            schema, data = loader.load_all()
            self.assertEqual(len(data["objects"]), 1)
            self.assertEqual(data["objects"][0]["name"], "test")
        finally:
            cleanup_file(path)

    def test_load_multiple_records(self):
        records = [{"id": str(i)} for i in range(100)]
        path = create_temp_sqlite(records)
        try:
            loader = SQLiteLoader(path)
            schema, data = loader.load_all()
            self.assertEqual(len(data["objects"]), 100)
        finally:
            cleanup_file(path)

    def test_load_with_limit(self):
        records = [{"id": str(i)} for i in range(1000)]
        path = create_temp_sqlite(records)
        try:
            loader = SQLiteLoader(path)
            loader.connect()
            loader.discover_tables()
            loader.discover_schema()
            loader.load_records(limit=50)
            self.assertEqual(len(loader.data["objects"]), 50)
            loader.close()
        finally:
            cleanup_file(path)

    def test_flat_records_includes_source_table(self):
        path = create_temp_sqlite([{"x": "1"}], "my_table")
        try:
            loader = SQLiteLoader(path)
            loader.load_all()
            flat = loader.get_all_records_flat()
            self.assertEqual(len(flat), 1)
            self.assertEqual(flat[0]["__source_table__"], "my_table")
        finally:
            cleanup_file(path)

    def test_handles_special_characters_in_data(self):
        records = [{"text": "Hello 'World' \"Test\" ñ 日本語"}]
        path = create_temp_sqlite(records)
        try:
            loader = SQLiteLoader(path)
            schema, data = loader.load_all()
            self.assertEqual(len(data["objects"]), 1)
        finally:
            cleanup_file(path)

    def test_handles_null_values(self):
        """NULL values should be handled gracefully."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE test (a TEXT, b TEXT)")
        conn.execute("INSERT INTO test VALUES ('value', NULL)")
        conn.commit()
        conn.close()
        try:
            loader = SQLiteLoader(path)
            schema, data = loader.load_all()
            self.assertEqual(len(data["test"]), 1)
            self.assertIsNone(data["test"][0]["b"])
        finally:
            cleanup_file(path)


# ================================================================== #
#  SECTION 4: JSON LOADER TESTS (25 tests)
# ================================================================== #

class TestJSONLoader_BasicLoading(unittest.TestCase):
    """Test basic JSON loading functionality."""

    def test_load_simple_array(self):
        data = [{"name": "test1"}, {"name": "test2"}]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            self.assertTrue(loader.load())
            self.assertEqual(len(loader.records), 2)
        finally:
            cleanup_file(path)

    def test_load_empty_array(self):
        path = create_temp_json([])
        try:
            loader = JSONLoader(path)
            self.assertTrue(loader.load())
            self.assertEqual(len(loader.records), 0)
        finally:
            cleanup_file(path)

    def test_load_single_object(self):
        data = {"name": "test", "value": 123}
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertEqual(len(loader.records), 1)
        finally:
            cleanup_file(path)

    def test_load_missing_file(self):
        loader = JSONLoader("/nonexistent/path.json")
        self.assertFalse(loader.load())

    def test_load_invalid_json(self):
        path = create_temp_file_with_content("{invalid json!!!", ".json")
        try:
            loader = JSONLoader(path)
            self.assertFalse(loader.load())
        finally:
            cleanup_file(path)

    def test_load_empty_file(self):
        path = create_temp_file_with_content("", ".json")
        try:
            loader = JSONLoader(path)
            self.assertFalse(loader.load())
        finally:
            cleanup_file(path)


class TestJSONLoader_ComplexStructures(unittest.TestCase):
    """Test loading various JSON structures."""

    def test_dict_of_dicts(self):
        data = {
            "obj1": {"name": "Alpha", "value": 1},
            "obj2": {"name": "Beta", "value": 2}
        }
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertEqual(len(loader.records), 2)
        finally:
            cleanup_file(path)

    def test_nested_array_under_key(self):
        data = {
            "metadata": {"version": "1.0"},
            "catalog": [
                {"id": 1, "name": "Item1"},
                {"id": 2, "name": "Item2"},
                {"id": 3, "name": "Item3"}
            ]
        }
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertEqual(len(loader.records), 3)
        finally:
            cleanup_file(path)

    def test_deeply_nested_structure(self):
        data = {
            "level1": {
                "level2": {
                    "items": [{"a": 1}, {"b": 2}]
                }
            }
        }
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            # Should find the nested array
            self.assertGreater(len(loader.records), 0)
        finally:
            cleanup_file(path)

    def test_mixed_array_content(self):
        """Array with mixed types - should only keep dicts."""
        data = [
            {"valid": "dict"},
            "string_item",
            123,
            {"another": "dict"},
            None
        ]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertEqual(len(loader.records), 2)
        finally:
            cleanup_file(path)


class TestJSONLoader_KeyDiscovery(unittest.TestCase):
    """Test key discovery functionality."""

    def test_discover_keys_simple(self):
        data = [{"a": 1, "b": 2}]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertIn("a", loader.keys)
            self.assertIn("b", loader.keys)
        finally:
            cleanup_file(path)

    def test_discover_keys_across_records(self):
        data = [
            {"a": 1},
            {"b": 2},
            {"c": 3}
        ]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertIn("a", loader.keys)
            self.assertIn("b", loader.keys)
            self.assertIn("c", loader.keys)
        finally:
            cleanup_file(path)

    def test_keys_are_sorted(self):
        data = [{"z": 1, "a": 2, "m": 3}]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertEqual(loader.keys, sorted(loader.keys))
        finally:
            cleanup_file(path)


class TestJSONLoader_EdgeCases(unittest.TestCase):
    """Test edge cases for JSON loading."""

    def test_unicode_content(self):
        data = [{"name": "日本語テスト", "value": "Ñoño"}]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            self.assertTrue(loader.load())
            self.assertEqual(loader.records[0]["name"], "日本語テスト")
        finally:
            cleanup_file(path)

    def test_large_numbers(self):
        data = [{"big": 99999999999999999999999999}]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            self.assertTrue(loader.load())
        finally:
            cleanup_file(path)

    def test_boolean_values(self):
        data = [{"flag": True, "other": False}]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertTrue(loader.records[0]["flag"])
            self.assertFalse(loader.records[0]["other"])
        finally:
            cleanup_file(path)

    def test_null_values(self):
        data = [{"value": None}]
        path = create_temp_json(data)
        try:
            loader = JSONLoader(path)
            loader.load()
            self.assertIsNone(loader.records[0]["value"])
        finally:
            cleanup_file(path)


# ================================================================== #
#  SECTION 5: DATA MANAGER TESTS (15 tests)
# ================================================================== #

class TestDataManager_Merging(unittest.TestCase):
    """Test data merging functionality."""

    def test_merge_sqlite_only(self):
        path = create_temp_sqlite([{"a": "1"}])
        try:
            dm = DataManager()
            dm.load_sqlite(path)
            dm.merge()
            self.assertEqual(len(dm.get_records()), 1)
        finally:
            cleanup_file(path)

    def test_merge_json_only(self):
        path = create_temp_json([{"b": "2"}])
        try:
            dm = DataManager()
            dm.load_json(path)
            dm.merge()
            self.assertEqual(len(dm.get_records()), 1)
        finally:
            cleanup_file(path)

    def test_merge_both_sources(self):
        db_path = create_temp_sqlite([{"x": "1"}])
        json_path = create_temp_json([{"y": "2"}])
        try:
            dm = DataManager()
            dm.load_sqlite(db_path)
            dm.load_json(json_path)
            dm.merge()
            self.assertEqual(len(dm.get_records()), 2)
        finally:
            cleanup_file(db_path)
            cleanup_file(json_path)

    def test_merge_no_sources(self):
        dm = DataManager()
        dm.merge()
        self.assertEqual(len(dm.get_records()), 0)

    def test_column_names_merged(self):
        db_path = create_temp_sqlite([{"col_a": "1"}])
        json_path = create_temp_json([{"col_b": "2"}])
        try:
            dm = DataManager()
            dm.load_sqlite(db_path)
            dm.load_json(json_path)
            dm.merge()
            cols = dm.get_column_names()
            self.assertIn("col_a", cols)
            self.assertIn("col_b", cols)
        finally:
            cleanup_file(db_path)
            cleanup_file(json_path)


class TestDataManager_Summary(unittest.TestCase):
    """Test summary generation."""

    def test_summary_contains_record_count(self):
        dm = DataManager()
        dm.merge()
        summary = dm.summary()
        self.assertIn("Total records", summary)

    def test_summary_contains_column_info(self):
        path = create_temp_sqlite([{"test_col": "value"}])
        try:
            dm = DataManager()
            dm.load_sqlite(path)
            dm.merge()
            summary = dm.summary()
            self.assertIn("Column names", summary)
        finally:
            cleanup_file(path)

    def test_summary_is_string(self):
        dm = DataManager()
        dm.merge()
        self.assertIsInstance(dm.summary(), str)


# ================================================================== #
#  SECTION 6: TASK BUILDER TESTS (30 tests)
# ================================================================== #

class TestTaskBuilder_Generation(unittest.TestCase):
    """Test task generation functionality."""

    def test_generates_minimum_tasks(self):
        tasks = build_tasks([], min_tasks=20, max_tasks=50)
        self.assertGreaterEqual(len(tasks), 20)

    def test_respects_maximum_tasks(self):
        tasks = build_tasks([], min_tasks=20, max_tasks=30)
        self.assertLessEqual(len(tasks), 30)

    def test_generates_with_no_data(self):
        tasks = build_tasks([], min_tasks=25, max_tasks=25)
        self.assertEqual(len(tasks), 25)

    def test_generates_from_real_data(self):
        records = [{"name": f"Object{i}", "temperature": str(3000 + i * 100)} for i in range(50)]
        tasks = build_tasks(records, min_tasks=20, max_tasks=40)
        self.assertGreaterEqual(len(tasks), 20)
        self.assertLessEqual(len(tasks), 40)


class TestTaskBuilder_Structure(unittest.TestCase):
    """Test task structure conformance."""

    def test_task_has_input(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        for task in tasks:
            self.assertIn("input", task)

    def test_task_has_expected(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        for task in tasks:
            self.assertIn("expected", task)

    def test_task_has_difficulty(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        for task in tasks:
            self.assertIn("difficulty", task)

    def test_input_has_description(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        for task in tasks:
            self.assertIn("description", task["input"])
            self.assertIsInstance(task["input"]["description"], str)

    def test_expected_has_category(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        for task in tasks:
            self.assertIn("category", task["expected"])

    def test_expected_has_priority(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        for task in tasks:
            self.assertIn("priority", task["expected"])

    def test_expected_has_decision(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        for task in tasks:
            self.assertIn("decision", task["expected"])


class TestTaskBuilder_ValueRanges(unittest.TestCase):
    """Test that generated values are within valid ranges."""

    def test_category_in_range(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        for task in tasks:
            self.assertIn(task["expected"]["category"], [0, 1, 2])

    def test_priority_in_range(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        for task in tasks:
            self.assertIn(task["expected"]["priority"], [0, 1, 2])

    def test_decision_in_range(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        for task in tasks:
            self.assertIn(task["expected"]["decision"], [0, 1, 2])

    def test_difficulty_in_range(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        for task in tasks:
            self.assertIn(task["difficulty"], [1, 2, 3, 4, 5])


class TestTaskBuilder_Variety(unittest.TestCase):
    """Test task variety and diversity."""

    def test_not_all_same_category(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        categories = {t["expected"]["category"] for t in tasks}
        self.assertGreater(len(categories), 1)

    def test_not_all_same_priority(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        priorities = {t["expected"]["priority"] for t in tasks}
        self.assertGreater(len(priorities), 1)

    def test_not_all_same_decision(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        decisions = {t["expected"]["decision"] for t in tasks}
        self.assertGreater(len(decisions), 1)

    def test_multiple_unique_answer_triples(self):
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        triples = set()
        for t in tasks:
            e = t["expected"]
            triples.add((e["category"], e["priority"], e["decision"]))
        self.assertGreaterEqual(len(triples), 5)

    def test_all_categories_represented(self):
        """With 30+ tasks, all categories should appear."""
        tasks = build_tasks([], min_tasks=40, max_tasks=40)
        categories = {t["expected"]["category"] for t in tasks}
        self.assertEqual(categories, {0, 1, 2})

    def test_no_single_answer_dominates(self):
        """No single answer should be > 40% of tasks."""
        tasks = build_tasks([], min_tasks=30, max_tasks=30)
        counts = {}
        for t in tasks:
            e = t["expected"]
            key = (e["category"], e["priority"], e["decision"])
            counts[key] = counts.get(key, 0) + 1
        max_count = max(counts.values())
        self.assertLess(max_count / len(tasks), 0.45)


class TestTaskBuilder_DataDriven(unittest.TestCase):
    """Test task derivation from data."""

    def test_high_temp_high_radiation_abort(self):
        record = {"temperature": 7000, "radiation": 5.0, "name": "HotStar"}
        task = _derive_task_from_record(record, 0)
        self.assertIsNotNone(task)
        self.assertEqual(task["expected"]["category"], 0)  # system_failure
        self.assertEqual(task["expected"]["priority"], 2)  # high
        self.assertEqual(task["expected"]["decision"], 2)  # abort

    def test_low_fuel_abort(self):
        record = {"fuel": 10, "name": "LowFuelShip"}
        task = _derive_task_from_record(record, 0)
        self.assertIsNotNone(task)
        self.assertEqual(task["expected"]["category"], 2)  # resource
        self.assertEqual(task["expected"]["decision"], 2)  # abort

    def test_unstable_orbit_abort(self):
        record = {"eccentricity": 0.85, "name": "WildOrbit"}
        task = _derive_task_from_record(record, 0)
        self.assertIsNotNone(task)
        self.assertEqual(task["expected"]["category"], 1)  # navigation
        self.assertEqual(task["expected"]["decision"], 2)  # abort

    def test_nominal_record_produces_task(self):
        record = {"name": "NormalObject", "type": "star"}
        task = _derive_task_from_record(record, 0)
        self.assertIsNotNone(task)
        self.assertIn("input", task)
        self.assertIn("expected", task)


# ================================================================== #
#  SECTION 7: ENVIRONMENT TESTS (35 tests)
# ================================================================== #

class TestSpaceMissionEnv_Initialization(unittest.TestCase):
    """Test environment initialization."""

    def test_init_with_valid_tasks(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        self.assertIsNotNone(env)

    def test_init_empty_tasks_raises(self):
        with self.assertRaises(ValueError):
            SpaceMissionEnv([])

    def test_action_space_is_multi_discrete(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        self.assertEqual(env.action_space.shape, (3,))

    def test_action_space_has_correct_dimensions(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        # Each dimension should have 3 options (0, 1, 2)
        self.assertTrue(all(n == 3 for n in env.action_space.nvec))

    def test_observation_space_exists(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        self.assertIsNotNone(env.observation_space)

    def test_num_tasks_property(self):
        tasks = build_tasks([], min_tasks=10, max_tasks=10)
        env = SpaceMissionEnv(tasks)
        self.assertEqual(env.num_tasks, 10)


class TestSpaceMissionEnv_Reset(unittest.TestCase):
    """Test reset functionality."""

    def test_reset_returns_tuple(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        result = env.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_reset_returns_obs_and_info(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        obs, info = env.reset()
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

    def test_info_contains_description(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        obs, info = env.reset()
        self.assertIn("description", info)
        self.assertIsInstance(info["description"], str)

    def test_info_contains_difficulty(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        obs, info = env.reset()
        self.assertIn("difficulty", info)

    def test_reset_after_episode_done(self):
        tasks = build_tasks([], min_tasks=2, max_tasks=2)
        env = SpaceMissionEnv(tasks)
        env.reset()
        env.step((0, 0, 0))
        env.step((0, 0, 0))
        # Should be done now
        obs, info = env.reset()
        self.assertIn("description", info)

    def test_reset_with_seed(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        obs, info = env.reset(seed=42)
        self.assertIsNotNone(obs)


class TestSpaceMissionEnv_Step(unittest.TestCase):
    """Test step functionality."""

    def test_step_returns_five_values(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        env.reset()
        result = env.step((0, 0, 0))
        self.assertEqual(len(result), 5)

    def test_step_returns_correct_types(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        env.reset()
        obs, reward, terminated, truncated, info = env.step((0, 0, 0))
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_reward_is_float(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        env.reset()
        _, reward, _, _, _ = env.step((1, 1, 1))
        self.assertIsInstance(reward, float)

    def test_info_contains_eval(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        env.reset()
        _, _, _, _, info = env.step((0, 0, 0))
        self.assertIn("eval", info)
        self.assertIn("score", info["eval"])

    def test_truncated_always_false(self):
        """Truncated should always be False in this env."""
        tasks = build_tasks([], min_tasks=10, max_tasks=10)
        env = SpaceMissionEnv(tasks)
        env.reset()
        for _ in range(10):
            _, _, _, truncated, _ = env.step((0, 0, 0))
            self.assertFalse(truncated)


class TestSpaceMissionEnv_EpisodeFlow(unittest.TestCase):
    """Test episode progression."""

    def test_episode_terminates_after_all_tasks(self):
        n = 5
        tasks = build_tasks([], min_tasks=n, max_tasks=n)
        env = SpaceMissionEnv(tasks)
        env.reset()
        
        for i in range(n):
            _, _, terminated, _, _ = env.step((0, 0, 0))
            if i < n - 1:
                self.assertFalse(terminated)
            else:
                self.assertTrue(terminated)

    def test_step_after_done_raises(self):
        tasks = build_tasks([], min_tasks=1, max_tasks=1)
        env = SpaceMissionEnv(tasks)
        env.reset()
        env.step((0, 0, 0))
        with self.assertRaises(RuntimeError):
            env.step((0, 0, 0))

    def test_get_current_task_before_done(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        env.reset()
        task = env.get_current_task()
        self.assertIsNotNone(task)
        self.assertIn("input", task)

    def test_get_current_task_after_done(self):
        tasks = build_tasks([], min_tasks=1, max_tasks=1)
        env = SpaceMissionEnv(tasks)
        env.reset()
        env.step((0, 0, 0))
        self.assertIsNone(env.get_current_task())

    def test_episode_log_records_all_steps(self):
        n = 5
        tasks = build_tasks([], min_tasks=n, max_tasks=n)
        env = SpaceMissionEnv(tasks)
        env.reset()
        
        for _ in range(n):
            env.step(tuple(env.action_space.sample()))
        
        log = env.get_episode_log()
        self.assertEqual(len(log), n)
        
        for entry in log:
            self.assertIn("score", entry)
            self.assertIn("reward", entry)
            self.assertIn("action", entry)


class TestSpaceMissionEnv_Actions(unittest.TestCase):
    """Test action handling."""

    def test_accepts_tuple_action(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        env.reset()
        obs, reward, _, _, _ = env.step((1, 2, 0))
        self.assertIsInstance(reward, float)

    def test_accepts_list_action(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        env.reset()
        obs, reward, _, _, _ = env.step([1, 2, 0])
        self.assertIsInstance(reward, float)

    def test_accepts_numpy_action(self):
        import numpy as np
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        env.reset()
        action = np.array([0, 1, 2])
        obs, reward, _, _, _ = env.step(action)
        self.assertIsInstance(reward, float)

    def test_action_space_sample(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks)
        env.reset()
        
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)
            self.assertIsInstance(reward, float)


class TestSpaceMissionEnv_Rewards(unittest.TestCase):
    """Test reward calculation in environment."""

    def test_perfect_action_gives_positive_reward(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks, scale_by_difficulty=False)
        env.reset()
        
        task = env.tasks[0]
        e = task["expected"]
        perfect_action = (e["category"], e["priority"], e["decision"])
        
        _, reward, _, _, _ = env.step(perfect_action)
        self.assertEqual(reward, 10.0)

    def test_wrong_action_can_give_negative_reward(self):
        tasks = build_tasks([], min_tasks=10, max_tasks=10)
        env = SpaceMissionEnv(tasks, scale_by_difficulty=False)
        env.reset()
        
        got_negative = False
        for _ in range(10):
            _, reward, done, _, _ = env.step((0, 0, 0))
            if reward < 0:
                got_negative = True
                break
            if done:
                break
        
        # With random tasks, we should occasionally get negative rewards
        # (This might not always happen, so we don't assert)

    def test_perfect_agent_maximizes_reward(self):
        tasks = build_tasks([], min_tasks=5, max_tasks=5)
        env = SpaceMissionEnv(tasks, scale_by_difficulty=False)
        env.reset()
        
        total_reward = 0.0
        for task in tasks:
            e = task["expected"]
            action = (e["category"], e["priority"], e["decision"])
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
        
        # All perfect actions should give +10 each
        self.assertEqual(total_reward, 50.0)


# ================================================================== #
#  SECTION 8: INTEGRATION TESTS (15 tests)
# ================================================================== #

class TestIntegration_FullPipeline(unittest.TestCase):
    """End-to-end integration tests."""

    def test_pipeline_sqlite_only(self):
        records = [
            {"name": "Star-A", "temperature": "8000", "radiation": "3.5"},
            {"name": "Star-B", "temperature": "2500", "fuel": "90"},
        ]
        db_path = create_temp_sqlite(records)
        try:
            dm = DataManager()
            dm.load_sqlite(db_path)
            dm.merge()
            
            tasks = build_tasks(dm.get_records(), min_tasks=20, max_tasks=30)
            env = SpaceMissionEnv(tasks)
            
            obs, info = env.reset()
            total_reward = 0.0
            
            while True:
                action = env.action_space.sample()
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            
            self.assertIsInstance(total_reward, float)
        finally:
            cleanup_file(db_path)

    def test_pipeline_json_only(self):
        data = [
            {"object_name": "Galaxy-X", "mass": "15", "velocity": "200"},
            {"object_name": "Nebula-Y", "distance": "4.2"},
        ]
        json_path = create_temp_json(data)
        try:
            dm = DataManager()
            dm.load_json(json_path)
            dm.merge()
            
            tasks = build_tasks(dm.get_records(), min_tasks=20, max_tasks=30)
            env = SpaceMissionEnv(tasks)
            
            obs, info = env.reset()
            steps = 0
            
            while True:
                _, _, done, _, _ = env.step(env.action_space.sample())
                steps += 1
                if done:
                    break
            
            self.assertEqual(steps, len(tasks))
        finally:
            cleanup_file(json_path)

    def test_pipeline_both_sources(self):
        db_path = create_temp_sqlite([{"x": "1"}])
        json_path = create_temp_json([{"y": "2"}])
        try:
            dm = DataManager()
            dm.load_sqlite(db_path)
            dm.load_json(json_path)
            dm.merge()
            
            self.assertEqual(len(dm.get_records()), 2)
            
            tasks = build_tasks(dm.get_records(), min_tasks=20, max_tasks=30)
            env = SpaceMissionEnv(tasks)
            
            env.reset()
            for _ in range(len(tasks)):
                env.step(env.action_space.sample())
            
            log = env.get_episode_log()
            self.assertEqual(len(log), len(tasks))
        finally:
            cleanup_file(db_path)
            cleanup_file(json_path)

    def test_pipeline_no_data_fallback(self):
        """System must work with zero external data."""
        tasks = build_tasks([], min_tasks=25, max_tasks=40)
        self.assertGreaterEqual(len(tasks), 25)
        
        env = SpaceMissionEnv(tasks)
        obs, info = env.reset()
        
        while True:
            action = env.action_space.sample()
            _, _, done, _, _ = env.step(action)
            if done:
                break
        
        self.assertEqual(len(env.get_episode_log()), len(tasks))


class TestIntegration_AgentComparison(unittest.TestCase):
    """Test that different agent strategies produce different results."""

    def test_perfect_agent_beats_random(self):
        """Perfect agent should always outperform random."""
        tasks = build_tasks([], min_tasks=20, max_tasks=20)
        
        # Random agent
        env = SpaceMissionEnv(tasks, scale_by_difficulty=False)
        env.reset()
        random_total = 0.0
        for _ in range(20):
            action = tuple(env.action_space.sample())
            _, reward, done, _, _ = env.step(action)
            random_total += reward
            if done:
                break
        
        # Perfect agent
        env = SpaceMissionEnv(tasks, scale_by_difficulty=False)
        env.reset()
        perfect_total = 0.0
        for task in tasks:
            e = task["expected"]
            action = (e["category"], e["priority"], e["decision"])
            _, reward, done, _, _ = env.step(action)
            perfect_total += reward
            if done:
                break
        
        # Perfect should always be 20 * 10 = 200
        self.assertEqual(perfect_total, 200.0)
        # Random should be less (statistically almost certain)
        self.assertLessEqual(random_total, perfect_total)

    def test_fixed_agent_consistent(self):
        """Fixed agent should produce same reward on same tasks."""
        tasks = build_tasks([], min_tasks=10, max_tasks=10)
        fixed_action = (1, 1, 1)
        
        # Run 1
        env = SpaceMissionEnv(tasks, scale_by_difficulty=False)
        env.reset()
        total1 = 0.0
        for _ in range(10):
            _, reward, _, _, _ = env.step(fixed_action)
            total1 += reward
        
        # Run 2
        env = SpaceMissionEnv(tasks, scale_by_difficulty=False)
        env.reset()
        total2 = 0.0
        for _ in range(10):
            _, reward, _, _, _ = env.step(fixed_action)
            total2 += reward
        
        self.assertEqual(total1, total2)


# ================================================================== #
#  SECTION 9: STRESS TESTS (10 tests)
# ================================================================== #

class TestStress_LargeData(unittest.TestCase):
    """Test system performance with large datasets."""

    def test_large_sqlite_database(self):
        """Test with 1000+ records."""
        records = [{"id": str(i), "name": f"Object{i}", "value": str(i * 10)} for i in range(1000)]
        db_path = create_temp_sqlite(records)
        try:
            loader = SQLiteLoader(db_path)
            schema, data = loader.load_all()
            self.assertEqual(len(data["objects"]), 500)  # Limited by default
        finally:
            cleanup_file(db_path)

    def test_large_json_file(self):
        """Test with 1000+ records."""
        data = [{"id": i, "name": f"Item{i}"} for i in range(1000)]
        json_path = create_temp_json(data)
        try:
            loader = JSONLoader(json_path)
            loader.load()
            self.assertEqual(len(loader.records), 1000)
        finally:
            cleanup_file(json_path)

    def test_many_tasks(self):
        """Test generating maximum tasks."""
        tasks = build_tasks([], min_tasks=50, max_tasks=50)
        self.assertEqual(len(tasks), 50)

    def test_full_episode_50_tasks(self):
        """Complete a 50-task episode."""
        tasks = build_tasks([], min_tasks=50, max_tasks=50)
        env = SpaceMissionEnv(tasks)
        env.reset()
        
        step_count = 0
        while True:
            _, _, done, _, _ = env.step(env.action_space.sample())
            step_count += 1
            if done:
                break
        
        self.assertEqual(step_count, 50)


class TestStress_Performance(unittest.TestCase):
    """Test execution time constraints."""

    def test_task_generation_time(self):
        """Task generation should complete in < 2 seconds."""
        start = time.time()
        tasks = build_tasks([], min_tasks=50, max_tasks=50)
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0)

    def test_episode_completion_time(self):
        """50-step episode should complete in < 1 second."""
        tasks = build_tasks([], min_tasks=50, max_tasks=50)
        env = SpaceMissionEnv(tasks)
        
        start = time.time()
        env.reset()
        for _ in range(50):
            env.step(env.action_space.sample())
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 1.0)

    def test_evaluation_speed(self):
        """1000 evaluations should complete in < 0.5 seconds."""
        start = time.time()
        for _ in range(1000):
            action = (random.randint(0, 2), random.randint(0, 2), random.randint(0, 2))
            expected = {"category": random.randint(0, 2), "priority": random.randint(0, 2), "decision": random.randint(0, 2)}
            evaluate_action(action, expected)
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.5)


# ================================================================== #
#  SECTION 10: EDGE CASES (15 tests)
# ================================================================== #

class TestEdgeCases_DataLoader(unittest.TestCase):
    """Edge cases for data loading."""

    def test_sqlite_with_very_long_text(self):
        long_text = "A" * 10000
        records = [{"text": long_text}]
        db_path = create_temp_sqlite(records)
        try:
            loader = SQLiteLoader(db_path)
            schema, data = loader.load_all()
            self.assertEqual(data["objects"][0]["text"], long_text)
        finally:
            cleanup_file(db_path)

    def test_json_with_nested_objects(self):
        data = [{"outer": {"inner": {"deep": "value"}}}]
        json_path = create_temp_json(data)
        try:
            loader = JSONLoader(json_path)
            loader.load()
            self.assertEqual(len(loader.records), 1)
        finally:
            cleanup_file(json_path)

    def test_sqlite_special_table_names(self):
        """Test tables with spaces/special chars."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path)
        conn.execute('CREATE TABLE "My Table" (col TEXT)')
        conn.execute('INSERT INTO "My Table" VALUES ("test")')
        conn.commit()
        conn.close()
        try:
            loader = SQLiteLoader(path)
            schema, data = loader.load_all()
            self.assertIn("My Table", data)
        finally:
            cleanup_file(path)


class TestEdgeCases_Tasks(unittest.TestCase):
    """Edge cases for task generation."""

    def test_single_task_generation(self):
        tasks = build_tasks([], min_tasks=1, max_tasks=1)
        self.assertEqual(len(tasks), 1)

    def test_task_description_not_empty(self):
        tasks = build_tasks([], min_tasks=20, max_tasks=20)
        for task in tasks:
            self.assertGreater(len(task["input"]["description"]), 0)

    def test_min_equals_max(self):
        tasks = build_tasks([], min_tasks=15, max_tasks=15)
        self.assertEqual(len(tasks), 15)


class TestEdgeCases_Environment(unittest.TestCase):
    """Edge cases for environment."""

    def test_single_task_episode(self):
        tasks = build_tasks([], min_tasks=1, max_tasks=1)
        env = SpaceMissionEnv(tasks)
        env.reset()
        
        _, _, done, _, _ = env.step((0, 0, 0))
        self.assertTrue(done)

    def test_reset_multiple_times(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        
        for _ in range(10):
            env.reset()
            env.step((0, 0, 0))
        
        # Should not crash

    def test_action_boundary_values(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        env.reset()
        
        # Test all corners
        env.step((0, 0, 0))
        env.reset()
        env.step((2, 2, 2))
        env.reset()
        env.step((0, 2, 0))


# ================================================================== #
#  SECTION 11: API CONFORMANCE TESTS (10 tests)
# ================================================================== #

class TestAPIConformance_GymInterface(unittest.TestCase):
    """Test Gymnasium API conformance."""

    def test_inherits_from_gym_env(self):
        import gymnasium as gym
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        self.assertIsInstance(env, gym.Env)

    def test_has_action_space_attribute(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        self.assertTrue(hasattr(env, "action_space"))

    def test_has_observation_space_attribute(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        self.assertTrue(hasattr(env, "observation_space"))

    def test_has_reset_method(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        self.assertTrue(callable(getattr(env, "reset", None)))

    def test_has_step_method(self):
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        self.assertTrue(callable(getattr(env, "step", None)))

    def test_reset_signature(self):
        """Reset should accept seed and options kwargs."""
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        # Should not raise
        env.reset(seed=42, options={})

    def test_step_signature(self):
        """Step should accept action and return 5-tuple."""
        tasks = build_tasks([], min_tasks=3, max_tasks=3)
        env = SpaceMissionEnv(tasks)
        env.reset()
        result = env.step((0, 0, 0))
        self.assertEqual(len(result), 5)


class TestAPIConformance_TaskFormat(unittest.TestCase):
    """Test task format conformance to specification."""

    def test_task_format_matches_spec(self):
        """Each task must match the exact specified format."""
        tasks = build_tasks([], min_tasks=10, max_tasks=10)
        
        for task in tasks:
            # Top level keys
            self.assertIn("input", task)
            self.assertIn("expected", task)
            self.assertIn("difficulty", task)
            
            # Input structure
            self.assertIsInstance(task["input"], dict)
            self.assertIn("description", task["input"])
            self.assertIsInstance(task["input"]["description"], str)
            
            # Expected structure
            self.assertIsInstance(task["expected"], dict)
            self.assertIn("category", task["expected"])
            self.assertIn("priority", task["expected"])
            self.assertIn("decision", task["expected"])
            self.assertIsInstance(task["expected"]["category"], int)
            self.assertIsInstance(task["expected"]["priority"], int)
            self.assertIsInstance(task["expected"]["decision"], int)
            
            # Difficulty
            self.assertIsInstance(task["difficulty"], int)


# ================================================================== #
#  TEST RUNNER
# ================================================================== #

if __name__ == "__main__":
    print("=" * 70)
    print("  🔥 BRUTAL HACKATHON JUDGE TEST SUITE 🔥")
    print("  Testing: Basic → Intermediate → Advanced → Edge Cases → Stress")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        # Section 1: Evaluator (25 tests)
        TestEvaluator_Basic,
        TestEvaluator_PartialMatches,
        TestEvaluator_AllCombinations,
        TestEvaluator_OutputFormat,
        
        # Section 2: Rewards (20 tests)
        TestRewards_BaseValues,
        TestRewards_DifficultyScaling,
        TestRewards_Summary,
        
        # Section 3: SQLite Loader (25 tests)
        TestSQLiteLoader_Connection,
        TestSQLiteLoader_TableDiscovery,
        TestSQLiteLoader_SchemaDiscovery,
        TestSQLiteLoader_DataLoading,
        
        # Section 4: JSON Loader (25 tests)
        TestJSONLoader_BasicLoading,
        TestJSONLoader_ComplexStructures,
        TestJSONLoader_KeyDiscovery,
        TestJSONLoader_EdgeCases,
        
        # Section 5: Data Manager (15 tests)
        TestDataManager_Merging,
        TestDataManager_Summary,
        
        # Section 6: Task Builder (30 tests)
        TestTaskBuilder_Generation,
        TestTaskBuilder_Structure,
        TestTaskBuilder_ValueRanges,
        TestTaskBuilder_Variety,
        TestTaskBuilder_DataDriven,
        
        # Section 7: Environment (35 tests)
        TestSpaceMissionEnv_Initialization,
        TestSpaceMissionEnv_Reset,
        TestSpaceMissionEnv_Step,
        TestSpaceMissionEnv_EpisodeFlow,
        TestSpaceMissionEnv_Actions,
        TestSpaceMissionEnv_Rewards,
        
        # Section 8: Integration (15 tests)
        TestIntegration_FullPipeline,
        TestIntegration_AgentComparison,
        
        # Section 9: Stress Tests (10 tests)
        TestStress_LargeData,
        TestStress_Performance,
        
        # Section 10: Edge Cases (15 tests)
        TestEdgeCases_DataLoader,
        TestEdgeCases_Tasks,
        TestEdgeCases_Environment,
        
        # Section 11: API Conformance (10 tests)
        TestAPIConformance_GymInterface,
        TestAPIConformance_TaskFormat,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("  📊 BRUTAL JUDGE FINAL VERDICT 📊")
    print("=" * 70)
    print(f"  Tests Run     : {result.testsRun}")
    print(f"  Failures      : {len(result.failures)}")
    print(f"  Errors        : {len(result.errors)}")
    print(f"  Skipped       : {len(result.skipped)}")
    
    passed = result.testsRun - len(result.failures) - len(result.errors)
    success_rate = (passed / max(result.testsRun, 1)) * 100
    
    print(f"  Passed        : {passed}")
    print(f"  Success Rate  : {success_rate:.1f}%")
    print()
    
    if success_rate == 100:
        print("  ✅ VERDICT: PERFECT SCORE CONGRATS BUDDY!")
    elif success_rate >= 90:
        print("  ✅ VERDICT: EXCELLENT — Minor issues to fix")
    elif success_rate >= 75:
        print("  ⚠️  VERDICT: GOOD — Several issues need attention")
    elif success_rate >= 50:
        print("  ⚠️  VERDICT: NEEDS WORK — Significant issues found")
    else:
        print("  ❌ VERDICT: FAILING — Major problems detected")
    
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)