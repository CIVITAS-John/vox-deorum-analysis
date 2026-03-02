#!/usr/bin/env python3
"""
Extract panel data from SQLite databases with one row per player per game.
Each row contains player-specific information along with game metadata.
"""

import json
import os
from .utilities import (
    POLICY_BRANCHES, STRATEGY_MAPPINGS, CHANGE_FIELDS, PLAYER_CORE_FIELDS,
    get_player_info_cache, get_major_players,
    read_existing_csv, filter_existing_data, should_skip_game,
    get_game_id_from_path, get_experiment_from_path, open_database_readonly,
    write_csv_file
)

# Combine all player-specific fields for error handling
ALL_PLAYER_FIELDS = (
    PLAYER_CORE_FIELDS +
    CHANGE_FIELDS +
    list(STRATEGY_MAPPINGS.values()) +
    POLICY_BRANCHES
)

# Define field mappings for panel data structure
PANEL_FIELD_MAPPINGS = {
    # Game-level fields
    'experiment': None,  # Special handling: use folder name
    'game_id': 'gameId',
    'turn': 'turn',
    'map_type': 'mapType',
    'map_size': 'mapSize',
    'difficulty': 'difficulty',
    'game_speed': 'gameSpeed',
    'victory_type': 'victoryType',
    'victory_player_id': 'victoryPlayerID',

    # Player-specific fields
    'player_id': None,  # Player ID (0, 1, 2, etc.)
    'civilization': None,  # Player's civilization name
    'score': None,  # Player's highest score during the game
    'score_rank': None,  # Player's rank by score
    'score_ratio': None,  # Player's score as proportion of highest score
    'survival_turn': None,  # Last turn player was alive
    'is_winner': None,  # Whether this player won the game
    'input_tokens': None,  # Total input tokens used by this player
    'reasoning_tokens': None,  # Total reasoning tokens used by this player
    'output_tokens': None,  # Total output tokens used by this player
    'strategy_changes': None,  # Number of strategy changes
    'persona_changes': None,  # Number of persona changes
    'research_changes': None,  # Number of research changes
    'policy_changes': None,  # Number of policy changes
    'nuke': None,  # Max Nuke flavor value from first non-default change onwards
    'use_nuke': None,  # Max UseNuke flavor value from first non-default change onwards
    'domination_ratio': None,  # Ratio of turns with Conquest strategy
    'culture_ratio': None,  # Ratio of turns with Culture strategy
    'diplomatic_ratio': None,  # Ratio of turns with UnitedNations strategy
    'science_ratio': None,  # Ratio of turns with Spaceship strategy

    # Policy branch adoption fields (turn of first adoption)
    'tradition': None,  # Turn when Tradition was first adopted
    'authority': None,  # Turn when Authority was first adopted
    'progress': None,  # Turn when Progress was first adopted
    'fealty': None,  # Turn when Fealty was first adopted
    'statecraft': None,  # Turn when Statecraft was first adopted
    'artistry': None,  # Turn when Artistry was first adopted
    'industry': None,  # Turn when Industry was first adopted
    'imperialism': None,  # Turn when Imperialism was first adopted
    'rationalism': None,  # Turn when Rationalism was first adopted
    'freedom': None,  # Turn when Freedom was first adopted
    'autocracy': None,  # Turn when Autocracy was first adopted
    'order': None,  # Turn when Order was first adopted
}


def calculate_score_ranks(cursor):
    """
    Calculate score ranks for major civilization players based on their highest scores during the game.

    Args:
        cursor: Database cursor

    Returns:
        dict: Mapping of player_id to rank
    """
    cursor.execute("""
        SELECT ps.Key, MAX(ps.Score) as MaxScore
        FROM PlayerSummaries ps
        INNER JOIN PlayerInformations pi ON ps.Key = pi.Key
        WHERE pi.IsMajor = 1
        GROUP BY ps.Key
        HAVING MaxScore IS NOT NULL
        ORDER BY MaxScore DESC
    """)

    final_scores = cursor.fetchall()
    rank_map = {}

    for rank, (player_key, score) in enumerate(final_scores, start=1):
        rank_map[player_key] = rank

    return rank_map


def extract_flavor_max(cursor, player_id, column_name, default=50):
    """
    Extract the max value of a flavor column from FlavorChanges,
    starting from the first row where the value differs from the default.

    Args:
        cursor: Database cursor
        player_id: Player ID to query
        column_name: Name of the flavor column (e.g., 'Nuke', 'UseNuke')
        default: Default flavor value (50)

    Returns:
        int: Max flavor value from first non-default row onwards, or default if never changed
    """
    try:
        cursor.execute(f"""
            SELECT {column_name}
            FROM FlavorChanges
            WHERE Key = ?
            ORDER BY Turn
        """, (player_id,))
        rows = cursor.fetchall()
    except Exception:
        return default

    # Find the first row where value != default
    first_changed_idx = None
    for i, (value,) in enumerate(rows):
        if value != default:
            first_changed_idx = i
            break

    if first_changed_idx is None:
        return default

    # Return max from that row onwards
    return max(row[0] for row in rows[first_changed_idx:])


def perform_strategy_sanity_checks(cursor, player_id, player_data, db_name, experiment_name):
    """
    Perform sanity checks on strategy changes for a specific player.

    Args:
        cursor: Database cursor
        player_id: Player ID to check
        player_data: Player data dictionary containing survival_turn
        db_name: Database filename (for warnings)
        experiment_name: Name of the experiment folder
    """
    if not experiment_name or experiment_name.startswith("none-strategist"):
        return

    # Try FlavorChanges first, fall back to StrategyChanges
    try:
        cursor.execute("""
            SELECT Turn, Changes
            FROM FlavorChanges
            WHERE Key = ?
            ORDER BY Turn
        """, (player_id,))
        changes = cursor.fetchall()
        has_flavor_changes = True
    except:
        cursor.execute("""
            SELECT Turn, Changes
            FROM StrategyChanges
            WHERE Key = ?
            ORDER BY Turn
        """, (player_id,))
        changes = cursor.fetchall()
        has_flavor_changes = False

    strategy_turns = [row[0] for row in changes]

    if len(strategy_turns) > 1:
        # Calculate gaps between consecutive strategy changes
        max_gap = 0
        gap_start_turn = 0
        for i in range(1, len(strategy_turns)):
            gap = strategy_turns[i] - strategy_turns[i-1]
            if gap > max_gap:
                max_gap = gap
                gap_start_turn = strategy_turns[i-1]

        # Warn if gap is larger than 15
        if max_gap > 15:
            print(f"  WARNING: Player {player_id} in {db_name} ({experiment_name}) has max strategy change gap of {max_gap} turns starting from turn {gap_start_turn}")

    # Check gap between last strategy change and survival turn
    if strategy_turns and player_data['survival_turn'] != 'N/A':
        last_strategy_turn = strategy_turns[-1]
        survival_turn = player_data['survival_turn']
        final_gap = survival_turn - last_strategy_turn

        if final_gap > 15:
            print(f"  WARNING: Player {player_id} in {db_name} ({experiment_name}) has {final_gap} turns between last strategy change (turn {last_strategy_turn}) and survival (turn {survival_turn})")

    # Check for multiple "In-Game AI" rationale changes (only for StrategyChanges)
    if not has_flavor_changes:
        cursor.execute("""
            SELECT COUNT(*)
            FROM StrategyChanges
            WHERE Key = ? AND Rationale = 'Tweaked by In-Game AI(Unknown)'
        """, (player_id,))

        in_game_ai_count = cursor.fetchone()[0]
        if in_game_ai_count > 3:
            print(f"  WARNING: Player {player_id} in {db_name} ({experiment_name}) has {in_game_ai_count} strategy changes with 'In-Game AI' rationale")


def extract_player_data(cursor, player_id, player_info_cache, highest_score, victory_player_id, db_name, experiment_name, metadata):
    """
    Extract data for a specific player.

    Args:
        cursor: Database cursor
        player_id: Player ID to extract data for
        player_info_cache: Cached player information
        highest_score: Highest score among all players
        victory_player_id: ID of the winning player
        db_name: Database filename (for warnings)
        experiment_name: Name of the experiment folder (for sanity checks)
        metadata: Game metadata dictionary

    Returns:
        dict: Player-specific data
    """
    player_data = {}

    try:
        # Get player's civilization from cache
        if player_id in player_info_cache:
            player_data['civilization'] = player_info_cache[player_id]['civilization']
        else:
            player_data['civilization'] = 'N/A'

        # Extract token usage from metadata (inputTokens-0, outputTokens-0, etc.)
        input_token_key = f'inputTokens-{player_id}'
        reasoning_token_key = f'reasoningTokens-{player_id}'
        output_token_key = f'outputTokens-{player_id}'
        player_data['input_tokens'] = metadata.get(input_token_key, 'N/A')
        player_data['reasoning_tokens'] = metadata.get(reasoning_token_key, 'N/A')
        player_data['output_tokens'] = metadata.get(output_token_key, 'N/A')

        # Get player's highest score during the game
        cursor.execute("""
            SELECT MAX(Score) as MaxScore
            FROM PlayerSummaries
            WHERE Key = ?
        """, (player_id,))

        max_score_result = cursor.fetchone()
        if max_score_result and max_score_result[0] is not None:
            player_data['score'] = max_score_result[0]
        else:
            player_data['score'] = 'N/A'

        # Get player's survival turn (from the latest record)
        cursor.execute("""
            SELECT Turn
            FROM PlayerSummaries
            WHERE Key = ? AND IsLatest = 1
        """, (player_id,))

        survival_result = cursor.fetchone()
        if survival_result and survival_result[0] is not None:
            player_data['survival_turn'] = survival_result[0]
        else:
            player_data['survival_turn'] = 'N/A'

        # Calculate score ratio based on highest scores
        if player_data['score'] != 'N/A':
            if highest_score > 0:
                player_data['score_ratio'] = round(player_data['score'] / highest_score, 4)
            else:
                player_data['score_ratio'] = 0 if player_data['score'] == 0 else 1
        else:
            player_data['score_ratio'] = 'N/A'

        # Determine if this player is the winner
        player_data['is_winner'] = 1 if player_id == victory_player_id else 0

        # Get all strategy changes with turns and GrandStrategy for this player
        cursor.execute("""
            SELECT Turn, GrandStrategy, Changes
            FROM StrategyChanges
            WHERE Key = ?
            ORDER BY Turn
        """, (player_id,))

        all_strategy_changes = cursor.fetchall()

        # Try to get FlavorChanges count if table has data, otherwise count from StrategyChanges
        cursor.execute("""
            SELECT COUNT(*)
            FROM FlavorChanges
            WHERE Key = ? AND Changes != '["Rationale"]'
        """, (player_id,))

        flavor_changes = cursor.fetchone()[0] or 0
        if flavor_changes != 0:
            player_data['strategy_changes'] = flavor_changes
        else:
            player_data['strategy_changes'] = sum(1 for row in all_strategy_changes if row[2] != '["Rationale"]')

        # Extract Nuke and UseNuke flavor max values
        # Only meaningful if the player ever researched nuke-related techs
        cursor.execute("""
            SELECT COUNT(*)
            FROM PlayerSummaries
            WHERE Key = ? AND (
                CurrentResearch LIKE 'Nuclear Fission%'
                OR CurrentResearch LIKE 'Satellites%'
                OR CurrentResearch LIKE 'Advanced Ballistics%'
            )
        """, (player_id,))
        has_nuke_research = cursor.fetchone()[0] > 0

        if has_nuke_research:
            player_data['nuke'] = extract_flavor_max(cursor, player_id, 'Nuke')
            player_data['use_nuke'] = extract_flavor_max(cursor, player_id, 'UseNuke')
        else:
            player_data['nuke'] = 'N/A'
            player_data['use_nuke'] = 'N/A'

        # Calculate ratios for each strategy type
        # Initialize all strategy turn counts to 0
        strategy_turns = {strategy: 0 for strategy in STRATEGY_MAPPINGS.keys()}

        if all_strategy_changes:
            for i, (turn, grand_strategy, changes) in enumerate(all_strategy_changes):
                if grand_strategy and grand_strategy in strategy_turns:  # Only process known strategies
                    # Calculate duration until next change or survival turn
                    if i < len(all_strategy_changes) - 1:
                        duration = all_strategy_changes[i + 1][0] - turn
                    else:
                        # For last strategy, use survival turn if available
                        if player_data['survival_turn'] != 'N/A':
                            duration = player_data['survival_turn'] - turn + 1
                        else:
                            duration = 1  # Default to 1 turn if no survival info

                    strategy_turns[grand_strategy] += duration

        # Calculate ratios based on total strategy turns
        total_turns = sum(strategy_turns.values())
        if total_turns > 0:
            for strategy, ratio_field in STRATEGY_MAPPINGS.items():
                player_data[ratio_field] = round(strategy_turns[strategy] / total_turns, 4)
        else:
            # If no strategy turns recorded, set all ratios to 0
            for ratio_field in STRATEGY_MAPPINGS.values():
                player_data[ratio_field] = 0

        # Perform sanity checks for player 3 and player 4
        if player_id in [2, 3]:
            perform_strategy_sanity_checks(cursor, player_id, player_data, db_name, experiment_name)

        # Count persona changes
        cursor.execute("""
            SELECT COUNT(*)
            FROM PersonaChanges
            WHERE Key = ? AND Changes != '["Rationale"]'
        """, (player_id,))
        player_data['persona_changes'] = cursor.fetchone()[0] or 0

        # Count research changes
        cursor.execute("""
            SELECT COUNT(*)
            FROM ResearchChanges
            WHERE Key = ? AND Changes != '["Rationale"]'
        """, (player_id,))
        player_data['research_changes'] = cursor.fetchone()[0] or 0

        # Count policy changes
        cursor.execute("""
            SELECT COUNT(*)
            FROM PolicyChanges
            WHERE Key = ? AND Changes != '["Rationale"]'
        """, (player_id,))
        player_data['policy_changes'] = cursor.fetchone()[0] or 0

        # Extract policy branch adoptions from GameEvents
        # Initialize all policy fields to N/A
        for field in POLICY_BRANCHES:
            player_data[field] = 'N/A'

        # Query for PlayerAdoptPolicyBranch events for this player
        cursor.execute(f"""
            SELECT Turn, Payload
            FROM GameEvents
            WHERE Player{player_id} = 2
            AND (Type = 'PlayerAdoptPolicyBranch' OR Type = 'IdeologyAdopted')
            ORDER BY Turn
        """)

        policy_adoptions = cursor.fetchall()

        # Track adopted branches to only record first adoption
        adopted_branches = set()

        for turn, payload in policy_adoptions:
            try:
                # Parse the JSON payload
                data = json.loads(payload)

                # Get the branch type
                branch_type = data.get('BranchType', '').lower()

                # Check the player ID in payload
                payload_player_id = data.get('PlayerID', -1)
                if payload_player_id != player_id:
                    continue  # Skip if not for this player

                # Map to our field names and record first adoption
                if branch_type and branch_type not in adopted_branches:
                    if branch_type in POLICY_BRANCHES:
                        player_data[branch_type] = turn
                        adopted_branches.add(branch_type)
                    else:
                        # Log warning for unknown branch type
                        print(f"  WARNING: Unknown policy branch type '{branch_type}' found for player {player_id} in {db_name}")

            except Exception as e:
                print(f"  ERROR: Error processing policy branch for player {player_id}: {e}")
    except Exception as e:
        print(f"Error extracting data for player {player_id}: {e}")
        # Return partial data with N/A values
        for key in ALL_PLAYER_FIELDS:
            if key not in player_data:
                player_data[key] = 'N/A'

    return player_data


def extract_game_panel_data(db_path):
    """
    Extract panel data from a SQLite database (one row per player).

    Args:
        db_path: Path to the SQLite database file

    Returns:
        list: List of dictionaries, one per player
    """
    panel_rows = []

    conn, cursor = open_database_readonly(db_path)
    if not conn:
        return []

    try:
        # Query game metadata
        cursor.execute("SELECT Key, Value FROM GameMetadata")
        metadata = dict(cursor.fetchall())

        # Get player info cache
        player_info_cache = get_player_info_cache(cursor)

        # Get list of major civilization players
        major_players = [pid for pid, info in player_info_cache.items() if info['is_major']]

        # Get highest score across all turns for ratio calculation (only from major civs)
        cursor.execute("""
            SELECT MAX(ps.Score)
            FROM PlayerSummaries ps
            INNER JOIN PlayerInformations pi ON ps.Key = pi.Key
            WHERE pi.IsMajor = 1
        """)
        highest_score = cursor.fetchone()[0] or 0

        # Calculate score ranks (only for major civs)
        rank_map = calculate_score_ranks(cursor)

        # Get victory player ID and convert to integer
        victory_player_id_raw = metadata.get('victoryPlayerID', 'N/A')
        try:
            victory_player_id = int(float(victory_player_id_raw))
        except (ValueError, TypeError):
            victory_player_id = -1  # Use -1 for invalid/missing victory player ID

        # Check if cultural victory is actually a "survival" victory
        victory_type = metadata.get('victoryType', 'N/A')
        turn = metadata.get('turn', 'N/A')
        if victory_type == 'Cultural' and turn != 'N/A':
            try:
                turn_number = int(turn)
                if turn_number <= 300:
                    victory_type = 'Survival'
                    print(f"  Note: Cultural victory at turn {turn_number} marked as Survival in {os.path.basename(db_path)}")
            except (ValueError, TypeError):
                pass

        # Create a row for each major player
        for player_id in major_players:
            row = {}

            # Add game-level metadata
            row['experiment'] = get_experiment_from_path(db_path)
            row['game_id'] = metadata.get('gameId', 'N/A')
            row['turn'] = metadata.get('turn', 'N/A')
            row['map_type'] = metadata.get('mapType', 'N/A')
            row['map_size'] = metadata.get('mapSize', 'N/A')
            row['difficulty'] = metadata.get('difficulty', 'N/A')
            row['game_speed'] = metadata.get('gameSpeed', 'N/A')
            row['victory_type'] = victory_type
            row['victory_player_id'] = victory_player_id if victory_player_id != -1 else 'N/A'

            # Add player ID
            row['player_id'] = player_id

            # Extract player-specific data
            player_data = extract_player_data(
                cursor, player_id, player_info_cache, highest_score,
                victory_player_id, os.path.basename(db_path),
                row['experiment'], metadata
            )
            row.update(player_data)

            # Add score rank from pre-calculated ranks
            row['score_rank'] = rank_map.get(player_id, 'N/A')

            panel_rows.append(row)

        conn.close()

        # Sanity check: folder name should start with metadata experiment field
        experiment_name = get_experiment_from_path(db_path)
        if 'experiment' in metadata:
            metadata_experiment = metadata['experiment']
            if not experiment_name.startswith(metadata_experiment):
                print(f"WARNING: Folder name '{experiment_name}' does not start with metadata experiment '{metadata_experiment}' in {os.path.basename(db_path)}")

        # Warning for experiments that don't start with "none-strategist" AND have zero tokens
        if experiment_name and not experiment_name.startswith("none-strategist"):
            # Check if player 0 has zero input/output tokens
            input_tokens_0 = metadata.get('inputTokens-0', 'N/A')
            output_tokens_0 = metadata.get('outputTokens-0', 'N/A')

            if input_tokens_0 == 'N/A' or output_tokens_0 == 'N/A':
                print(f"  WARNING: Experiment '{experiment_name}' does not start with 'none-strategist' but has 0 tokens for player 0 in {os.path.basename(db_path)}")

        return panel_rows

    except Exception as e:
        print(f"Error processing {os.path.basename(db_path)}: {e}")
        conn.close()
        return []


def export_panel_data(db_files, available_game_ids):
    """
    Export panel data from multiple databases to CSV.

    Args:
        db_files: List of database file paths
        available_game_ids: Set of available game IDs

    Returns:
        int: Number of new rows added
    """
    output_file = 'panel_data.csv'
    expected_fieldnames = list(PANEL_FIELD_MAPPINGS.keys())

    # Read existing data
    existing_data, existing_game_ids, structure_matches = read_existing_csv(output_file, expected_fieldnames)

    if not structure_matches:
        print("Discarding existing panel data due to structure mismatch...")
        existing_data = []
        existing_game_ids = set()
    else:
        # Filter existing data
        filtered_data, kept_game_ids, pruned_rows, pruned_game_ids = filter_existing_data(
            existing_data, available_game_ids
        )

        if pruned_rows > 0:
            print(f"  Filtered out {pruned_rows} rows from {len(pruned_game_ids)} games without database files")
            print(f"  Pruned game IDs: {sorted(list(pruned_game_ids))[:10]}{'...' if len(pruned_game_ids) > 10 else ''}")

        existing_data = filtered_data
        existing_game_ids = kept_game_ids
        print(f"Found {len(existing_data)} existing rows from {len(existing_game_ids)} games")

    # Extract new panel data
    print("\nExtracting panel data...")
    new_panel_data = []
    skipped_count = 0

    for i, db_file in enumerate(db_files):
        game_id = get_game_id_from_path(db_file)

        # Skip if already processed
        if game_id and should_skip_game(game_id, existing_game_ids):
            skipped_count += 1
            continue

        panel_rows = extract_game_panel_data(db_file)
        if panel_rows:
            new_panel_data.extend(panel_rows)
            print(f"Processed: {os.path.basename(db_file)} ({len(panel_rows)} player rows)")

    print(f"\nSkipped {skipped_count} databases that were already exported")

    # Combine and export data
    all_panel_data = existing_data + new_panel_data

    if new_panel_data or pruned_rows > 0:
        if write_csv_file(output_file, expected_fieldnames, all_panel_data):
            print(f"\nSuccessfully exported total of {len(all_panel_data)} player-game observations to {output_file}")
            if new_panel_data:
                print(f"  - {len(existing_data)} existing rows")
                print(f"  - {len(new_panel_data)} new rows added")

            # Count unique games
            unique_games = set(row['game_id'] for row in all_panel_data if row.get('game_id') != 'N/A')
            if unique_games:
                print(f"Data now includes {len(unique_games)} unique games")
    else:
        print(f"\nNo new panel data to export. Existing file contains {len(existing_data)} rows.")

    return len(new_panel_data)