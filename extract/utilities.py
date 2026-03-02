#!/usr/bin/env python3
"""
Shared utility functions and constants for data extraction scripts.
"""

import os
import csv
import sqlite3
from pathlib import Path


# Define policy branch types for consistent access
POLICY_BRANCHES = [
    'tradition', 'authority', 'progress',      # Ancient era
    'fealty', 'statecraft', 'artistry',        # Classical/Medieval era
    'industry', 'imperialism', 'rationalism',  # Renaissance/Industrial era
    'freedom', 'autocracy', 'order'            # Modern era
]

# Define strategy types and their corresponding ratio fields
STRATEGY_MAPPINGS = {
    'Conquest': 'domination_ratio',
    'Culture': 'culture_ratio',
    'UnitedNations': 'diplomatic_ratio',
    'Spaceship': 'science_ratio'
}

# Define change tracking fields
CHANGE_FIELDS = [
    'strategy_changes',
    'persona_changes',
    'research_changes',
    'policy_changes'
]

# Define core player data fields
PLAYER_CORE_FIELDS = [
    'civilization',
    'score',
    'survival_turn',
    'score_ratio',
    'is_winner',
    'nuke',
    'use_nuke'
]


def find_all_databases(root_dir):
    """
    Find all .db files in the directory tree and extract their game IDs.
    Excludes player-specific database files (those containing "-player-" in the filename).

    Args:
        root_dir: Root directory to search

    Returns:
        tuple: (list of database file paths, set of game IDs)
    """
    db_files = []
    game_ids = set()

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.db'):
                # Skip files with "-player-" in the name (these are player-specific exports)
                if "-player-" in file:
                    continue

                db_files.append(os.path.join(root, file))

                # Extract game_id from filename (e.g., "003aaf62-b887-4d33-9984-24b1ae78b949_1761770718599.db")
                # Split by underscore and take the first part (the UUID)
                parts = file[:-3].split('_')  # Remove ".db" and split
                if len(parts) >= 1:
                    game_ids.add(parts[0])

    return db_files, game_ids


def get_player_info_cache(cursor):
    """
    Pre-fetch and cache player information from the database.

    Args:
        cursor: Database cursor

    Returns:
        dict: Dictionary mapping player_id to player information
              {player_id: {'civilization': name, 'is_major': bool}}
    """
    cursor.execute("""
        SELECT Key, Civilization, IsMajor
        FROM PlayerInformations
    """)

    player_info = {}
    for row in cursor.fetchall():
        player_id, civilization, is_major = row
        player_info[player_id] = {
            'civilization': civilization if civilization else 'N/A',
            'is_major': bool(is_major)
        }

    return player_info


def get_major_players(cursor):
    """
    Get list of major civilization player IDs.

    Args:
        cursor: Database cursor

    Returns:
        list: List of player IDs that are major civilizations
    """
    cursor.execute("""
        SELECT Key
        FROM PlayerInformations
        WHERE IsMajor = 1
        ORDER BY Key
    """)

    return [row[0] for row in cursor.fetchall()]


def read_existing_csv(filepath, expected_fields):
    """
    Read existing CSV file and validate its structure.

    Args:
        filepath: Path to CSV file
        expected_fields: List of expected field names

    Returns:
        tuple: (list of existing data rows, set of existing game IDs, bool indicating if structure matches)
    """
    if not os.path.exists(filepath):
        return [], set(), True

    existing_data = []
    existing_game_ids = set()

    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        existing_fieldnames = reader.fieldnames

        # Check if columns match
        if existing_fieldnames != expected_fields:
            print(f"WARNING: Column mismatch in {filepath}")
            print(f"  Expected {len(expected_fields)} columns")
            print(f"  Found {len(existing_fieldnames)} columns")

            missing = set(expected_fields) - set(existing_fieldnames)
            extra = set(existing_fieldnames) - set(expected_fields)

            if missing:
                print(f"  Missing columns: {missing}")
            if extra:
                print(f"  Extra columns: {extra}")

            return [], set(), False

        # Read data and extract game IDs
        for row in reader:
            existing_data.append(row)
            game_id = row.get('game_id')
            if game_id and game_id != 'N/A':
                existing_game_ids.add(game_id)

    return existing_data, existing_game_ids, True


def filter_existing_data(existing_data, available_game_ids):
    """
    Filter existing data to keep only games that still have database files.

    Args:
        existing_data: List of existing data rows
        available_game_ids: Set of game IDs that have database files

    Returns:
        tuple: (filtered data list, set of kept game IDs, number of pruned rows, set of pruned game IDs)
    """
    filtered_data = []
    kept_game_ids = set()
    pruned_rows = 0
    pruned_game_ids = set()

    for row in existing_data:
        game_id = row.get('game_id')
        if game_id and game_id != 'N/A':
            if game_id in available_game_ids:
                filtered_data.append(row)
                kept_game_ids.add(game_id)
            else:
                pruned_rows += 1
                pruned_game_ids.add(game_id)
        else:
            # Keep rows with N/A game_id (shouldn't normally happen but be safe)
            filtered_data.append(row)

    return filtered_data, kept_game_ids, pruned_rows, pruned_game_ids


def should_skip_game(game_id, existing_game_ids):
    """
    Check if a game should be skipped because it's already processed.

    Args:
        game_id: Game ID to check
        existing_game_ids: Set of already processed game IDs

    Returns:
        bool: True if game should be skipped
    """
    return game_id in existing_game_ids


def get_game_id_from_path(db_path):
    """
    Extract game ID from database file path.

    Args:
        db_path: Path to database file

    Returns:
        str: Game ID or None if cannot extract
    """
    db_filename = os.path.basename(db_path)

    if db_filename.endswith('.db'):
        # Split by underscore and take the first part (the UUID)
        parts = db_filename[:-3].split('_')  # Remove ".db" and split
        if len(parts) >= 1:
            return parts[0]

    return None


def get_experiment_from_path(db_path):
    """
    Extract experiment name from database file path (parent folder name).

    Args:
        db_path: Path to database file

    Returns:
        str: Experiment name
    """
    return os.path.basename(os.path.dirname(db_path))


def open_database_readonly(db_path):
    """
    Open a SQLite database in read-only mode with immutable flag.

    Args:
        db_path: Path to database file

    Returns:
        tuple: (connection, cursor) or (None, None) on error
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print(f"Error opening {os.path.basename(db_path)}: {e}")
        return None, None


def write_csv_file(filepath, fieldnames, data_rows):
    """
    Write data to CSV file.

    Args:
        filepath: Path to output CSV file
        fieldnames: List of field names for CSV header
        data_rows: List of dictionaries containing data

    Returns:
        bool: True if successful
    """
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)
        return True
    except Exception as e:
        print(f"Error writing to {filepath}: {e}")
        return False