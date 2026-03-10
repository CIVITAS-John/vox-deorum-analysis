#!/usr/bin/env python3
"""
Main entry point for data extraction scripts.
Extracts both panel data and turn-based data from Civilization game databases.
"""

import os
import sys
from .utilities import find_all_databases, export_game_timestamps
from .extract_panel import export_panel_data
from .extract_turns import export_turn_data


def main():
    """
    Main function to process all databases and export both panel and turn-based data.
    """
    # Get root directory (parent of analysis folder)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(current_dir, "..")

    print("="*60)
    print("Civilization Game Data Extraction Tool")
    print("="*60)

    # Find all database files and extract game IDs
    print("\nSearching for database files...")
    db_files, available_game_ids = find_all_databases(root_dir)
    print(f"Found {len(db_files)} database files with {len(available_game_ids)} unique games")

    # Report duplicate database files (same game ID, multiple files)
    if len(db_files) > len(available_game_ids):
        from collections import defaultdict
        from .utilities import get_game_id_from_path
        games_to_files = defaultdict(list)
        for db_file in db_files:
            game_id = get_game_id_from_path(db_file)
            if game_id:
                folder = os.path.basename(os.path.dirname(db_file))
                games_to_files[game_id].append(f"{folder}/{os.path.basename(db_file)}")
        for game_id, files in sorted(games_to_files.items()):
            if len(files) > 1:
                print(f"  Duplicate game {game_id}: {', '.join(files)}")

    if not db_files:
        print("\nNo database files found. Exiting.")
        return

    # Export game timestamps
    print("\n" + "-"*60)
    print("EXTRACTING GAME TIMESTAMPS")
    print("-"*60)
    export_game_timestamps(root_dir)

    # Export panel data
    print("\n" + "-"*60)
    print("EXTRACTING PANEL DATA")
    print("-"*60)
    new_panel_rows = export_panel_data(db_files, available_game_ids)

    # Export turn-based data
    print("\n" + "-"*60)
    print("EXTRACTING TURN-BASED DATA")
    print("-"*60)
    new_turn_rows = export_turn_data(db_files, available_game_ids)

    # Summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Panel data: {new_panel_rows} new rows added to panel_data.csv")
    print(f"Turn data: {new_turn_rows} new records added to turn_data.csv (includes flavor change columns)")

    if new_panel_rows > 0 or new_turn_rows > 0:
        print("\nData files have been updated successfully.")
    else:
        print("\nNo new data to extract. All databases were already processed.")


if __name__ == "__main__":
    main()