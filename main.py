#!/usr/bin/env python3
"""
main.py
-------
Entry point for the Space Mission Control RL Environment.

1. Prompts the user to select SQLite and JSON data files.
2. Loads and merges data.
3. Builds mission tasks.
4. Instantiates the Gym environment.
5. Runs a demo agent loop (random + perfect + fixed).
"""

import os
import sys
import random

# ---- project modules ---- #
from data_loader import DataManager
from task_builder import build_tasks
from env import SpaceMissionEnv


# ================================================================== #
#  File selection helpers
# ================================================================== #

def _try_tkinter_file_dialog(title: str, filetypes: list) -> str:
    """Try to open a Tk file dialog; return '' on failure."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path or ""
    except Exception:
        return ""


def select_file(prompt: str, extension: str, default_name: str) -> str:
    """
    Ask the user to select a file.
    Tries GUI dialog first, falls back to manual path input.
    Returns the chosen path, or '' to skip.
    """
    print(f"\n{'─' * 60}")
    print(f"  {prompt}")
    print(f"{'─' * 60}")
    print("  [1] Browse with file dialog")
    print("  [2] Type file path manually")
    print("  [3] Skip (use fallback/synthetic data)")

    choice = input("  Your choice (1/2/3): ").strip()

    if choice == "1":
        filetypes = [
            (f"{extension.upper()} files", f"*{extension}"),
            ("All files", "*.*"),
        ]
        path = _try_tkinter_file_dialog(prompt, filetypes)
        if path:
            print(f"  Selected: {path}")
            return path
        else:
            print("  File dialog unavailable or cancelled. Falling back...")
            choice = "2"  # fall through to manual input

    if choice == "2":
        path = input(f"  Enter path to {extension} file: ").strip()
        if path and os.path.isfile(path):
            print(f"  ✓ File found: {path}")
            return path
        elif path:
            print(f"  ✗ File not found: {path}")
            return ""
        else:
            return ""

    # choice == "3" or anything else
    print("  Skipped — will use fallback data.")
    return ""


# ================================================================== #
#  Demo agents
# ================================================================== #

def run_random_agent(env: SpaceMissionEnv, label: str = "Random Agent"):
    """Run the environment with an agent that picks random actions."""
    print(f"\n{'▓' * 70}")
    print(f"  {label}")
    print(f"{'▓' * 70}")

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        raw_action = env.action_space.sample()
        # Convert numpy int64 to plain Python int for clean display
        action = tuple(int(a) for a in raw_action)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Display expected answer alongside agent action
        expected = info.get("step_log", {}).get("expected", "?")
        print(
            f"  Step {step:3d} | "
            f"Action {action} | "
            f"Expected {expected} | "
            f"Score {info['eval']['score']}/3 | "
            f"Reward {reward:+6.2f} | "
            f"Cumulative {total_reward:+8.2f}"
        )

    print(f"\n  {'─' * 50}")
    print(f"  {label} finished: {step} steps, total reward = {total_reward:+.2f}")
    print(f"  {'─' * 50}\n")
    return total_reward


def run_perfect_agent(env: SpaceMissionEnv, label: str = "Perfect Agent"):
    """Run the environment with an agent that always picks the correct answer."""
    print(f"\n{'▓' * 70}")
    print(f"  {label}")
    print(f"{'▓' * 70}")

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        # Peek at the expected answer for the current task
        task = env.tasks[env._current_index]
        e = task["expected"]
        action = (int(e["category"]), int(e["priority"]), int(e["decision"]))
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        expected = info.get("step_log", {}).get("expected", "?")
        print(
            f"  Step {step:3d} | "
            f"Action {action} | "
            f"Expected {expected} | "
            f"Score {info['eval']['score']}/3 | "
            f"Reward {reward:+6.2f} | "
            f"Cumulative {total_reward:+8.2f}"
        )

    print(f"\n  {'─' * 50}")
    print(f"  {label} finished: {step} steps, total reward = {total_reward:+.2f}")
    print(f"  {'─' * 50}\n")
    return total_reward


def run_fixed_agent(
    env: SpaceMissionEnv,
    fixed_action=(0, 1, 1),
    label: str = "Fixed Agent (0,1,1)",
):
    """Run with a single fixed action every step."""
    print(f"\n{'▓' * 70}")
    print(f"  {label}")
    print(f"{'▓' * 70}")

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    # Ensure clean Python ints
    action = tuple(int(a) for a in fixed_action)

    while not done:
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        expected = info.get("step_log", {}).get("expected", "?")
        print(
            f"  Step {step:3d} | "
            f"Action {action} | "
            f"Expected {expected} | "
            f"Score {info['eval']['score']}/3 | "
            f"Reward {reward:+6.2f} | "
            f"Cumulative {total_reward:+8.2f}"
        )

    print(f"\n  {'─' * 50}")
    print(f"  {label} finished: {step} steps, total reward = {total_reward:+.2f}")
    print(f"  {'─' * 50}\n")
    return total_reward


# ================================================================== #
#  Detailed task preview
# ================================================================== #

def show_task_preview(tasks, count=5):
    """Display a detailed preview of the first N tasks."""
    preview_count = min(count, len(tasks))
    print("─" * 70)
    print(f"  SAMPLE TASKS (first {preview_count})")
    print("─" * 70)

    for i, t in enumerate(tasks[:preview_count]):
        desc = t["input"]["description"]
        # Show up to 120 chars
        if len(desc) > 120:
            desc_display = desc[:117] + "..."
        else:
            desc_display = desc

        expected = t["expected"]
        cat_name = {0: "system_failure", 1: "navigation", 2: "resource"}.get(
            expected["category"], "?"
        )
        pri_name = {0: "low", 1: "medium", 2: "high"}.get(
            expected["priority"], "?"
        )
        dec_name = {0: "continue", 1: "adjust", 2: "abort"}.get(
            expected["decision"], "?"
        )

        print(f"\n  Task {i + 1}:")
        print(f"    Description : {desc_display}")
        print(
            f"    Expected    : category={expected['category']} ({cat_name}), "
            f"priority={expected['priority']} ({pri_name}), "
            f"decision={expected['decision']} ({dec_name})"
        )
        print(f"    Difficulty  : {t['difficulty']}/5")

    # Show answer variety statistics
    print(f"\n  {'─' * 50}")
    answer_counts = {}
    cat_counts = {0: 0, 1: 0, 2: 0}
    pri_counts = {0: 0, 1: 0, 2: 0}
    dec_counts = {0: 0, 1: 0, 2: 0}

    for t in tasks:
        e = t["expected"]
        key = (e["category"], e["priority"], e["decision"])
        answer_counts[key] = answer_counts.get(key, 0) + 1
        cat_counts[e["category"]] += 1
        pri_counts[e["priority"]] += 1
        dec_counts[e["decision"]] += 1

    print(f"  TASK DISTRIBUTION ({len(tasks)} total):")
    print(
        f"    Categories : system_failure={cat_counts[0]}, "
        f"navigation={cat_counts[1]}, resource={cat_counts[2]}"
    )
    print(
        f"    Priorities : low={pri_counts[0]}, "
        f"medium={pri_counts[1]}, high={pri_counts[2]}"
    )
    print(
        f"    Decisions  : continue={dec_counts[0]}, "
        f"adjust={dec_counts[1]}, abort={dec_counts[2]}"
    )
    print(f"    Unique answer triples: {len(answer_counts)}")
    print(f"  {'─' * 50}")


# ================================================================== #
#  Summary display
# ================================================================== #

def show_agent_comparison(r_random, r_perfect, r_fixed):
    """Display final comparison table."""
    print("\n" + "═" * 60)
    print("  AGENT COMPARISON")
    print("═" * 60)
    print(f"  {'Agent':<25} {'Total Reward':>15}")
    print(f"  {'─' * 40}")
    print(f"  {'🎲 Random Agent':<25} {r_random:>+15.2f}")
    print(f"  {'🎯 Perfect Agent':<25} {r_perfect:>+15.2f}")
    print(f"  {'🔧 Fixed Agent (0,1,1)':<25} {r_fixed:>+15.2f}")
    print(f"  {'─' * 40}")

    # Performance ratios
    if r_perfect != 0:
        random_pct = (r_random / r_perfect) * 100
        fixed_pct = (r_fixed / r_perfect) * 100
        print(f"  Random vs Perfect : {random_pct:.1f}%")
        print(f"  Fixed vs Perfect  : {fixed_pct:.1f}%")

    print("═" * 60)


# ================================================================== #
#  Main
# ================================================================== #

def main():
    print("╔" + "═" * 68 + "╗")
    print("║   SPACE MISSION CONTROL — RL ENVIRONMENT                         ║")
    print("║   Gymnasium-compatible Decision Trainer                           ║")
    print("╚" + "═" * 68 + "╝")

    # ──────────────────────────────────────────────────────────────── #
    #  Step 1: File selection
    # ──────────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  STEP 1: DATA SOURCE SELECTION")
    print("=" * 60)

    sqlite_path = select_file(
        "Select SQLite database (e.g. SpaceEngine_Index.db)",
        ".db",
        "SpaceEngine_Index.db",
    )

    json_path = select_file(
        "Select JSON catalog (e.g. astrophysical_object_catalog.json)",
        ".json",
        "astrophysical_object_catalog.json",
    )

    # ──────────────────────────────────────────────────────────────── #
    #  Step 2: Load and merge data
    # ──────────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  STEP 2: LOADING DATA")
    print("=" * 60)

    dm = DataManager()

    if sqlite_path:
        print(f"\n  Loading SQLite: {sqlite_path}")
        dm.load_sqlite(sqlite_path)
    else:
        print("\n  No SQLite file selected.")

    if json_path:
        print(f"\n  Loading JSON: {json_path}")
        dm.load_json(json_path)
    else:
        print("\n  No JSON file selected.")

    dm.merge()
    print("\n" + dm.summary())

    records = dm.get_records()

    if not records:
        print("\n  ⚠ No external data loaded — using synthetic fallback tasks.")

    # ──────────────────────────────────────────────────────────────── #
    #  Step 3: Build mission tasks
    # ──────────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  STEP 3: GENERATING MISSION TASKS")
    print("=" * 60)

    tasks = build_tasks(records, min_tasks=20, max_tasks=50)
    print(f"\n  ✓ {len(tasks)} tasks generated successfully.\n")

    # Detailed preview
    show_task_preview(tasks, count=5)

    # ──────────────────────────────────────────────────────────────── #
    #  Step 4: Create the Gym environment
    # ──────────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  STEP 4: INITIALIZING ENVIRONMENT")
    print("=" * 60)

    env = SpaceMissionEnv(tasks, render_mode=None, scale_by_difficulty=True)
    print(f"\n  ✓ SpaceMissionEnv created")
    print(f"    Tasks loaded  : {env.num_tasks}")
    print(f"    Action space  : {env.action_space}")
    print(f"    Obs space     : {env.observation_space}")
    print(f"    Action format : (category, priority, decision)")
    print(f"    category      : 0=system_failure, 1=navigation, 2=resource")
    print(f"    priority      : 0=low, 1=medium, 2=high")
    print(f"    decision      : 0=continue, 1=adjust, 2=abort")

    # ──────────────────────────────────────────────────────────────── #
    #  Step 5: Run demo agents
    # ──────────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  STEP 5: RUNNING DEMO AGENTS")
    print("=" * 60)

    r_random = run_random_agent(env, "🎲 Random Agent")
    r_perfect = run_perfect_agent(env, "🎯 Perfect Agent")
    r_fixed = run_fixed_agent(env, (0, 1, 1), "🔧 Fixed Agent (0,1,1)")

    # ──────────────────────────────────────────────────────────────── #
    #  Step 6: Comparison summary
    # ──────────────────────────────────────────────────────────────── #
    show_agent_comparison(r_random, r_perfect, r_fixed)

    # ──────────────────────────────────────────────────────────────── #
    #  Step 7: Run test suite (optional)
    # ──────────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  STEP 6: TEST SUITE")
    print("=" * 60)

    run_tests = input("\n  Run test suite? (y/n): ").strip().lower()
    if run_tests == "y":
        print("\n  Running all tests...\n")
        import unittest

        # Load the test module
        try:
            test_module = __import__("test_system")
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            # Summary
            print("\n" + "─" * 60)
            print(f"  Tests run    : {result.testsRun}")
            print(f"  Failures     : {len(result.failures)}")
            print(f"  Errors       : {len(result.errors)}")
            print(
                f"  Success rate : "
                f"{((result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)) * 100:.1f}%"
            )
            print("─" * 60)

        except ImportError as exc:
            print(f"  ✗ Could not load test_system module: {exc}")
        except Exception as exc:
            print(f"  ✗ Test execution error: {exc}")
    else:
        print("  Skipped test suite.")

    # ──────────────────────────────────────────────────────────────── #
    #  Done
    # ──────────────────────────────────────────────────────────────── #
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║   MISSION COMPLETE — All systems nominal                          ║")
    print("╚" + "═" * 68 + "╝")
    print("\n[main] Done. Goodbye! 🚀\n")


if __name__ == "__main__":
    main()