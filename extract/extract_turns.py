#!/usr/bin/env python3
"""
Extract turn-based data from SQLite databases.
Creates one row per player per turn with game state information,
including FlavorChanges data and an is_changed flag.
"""

import json
import os
from .utilities import (
    find_all_databases, get_player_info_cache, get_major_players,
    read_existing_csv, filter_existing_data, should_skip_game,
    get_game_id_from_path, get_experiment_from_path, open_database_readonly,
    write_csv_file
)

# Mapping from FlavorChanges DB column (PascalCase) to CSV column name (snake_case with flavor_ prefix)
FLAVOR_COLUMNS = [
    ('Offense', 'flavor_offense'),
    ('Defense', 'flavor_defense'),
    ('Mobilization', 'flavor_mobilization'),
    ('CityDefense', 'flavor_city_defense'),
    ('MilitaryTraining', 'flavor_military_training'),
    ('Recon', 'flavor_recon'),
    ('Ranged', 'flavor_ranged'),
    ('Mobile', 'flavor_mobile'),
    ('Nuke', 'flavor_nuke'),
    ('UseNuke', 'flavor_use_nuke'),
    ('Naval', 'flavor_naval'),
    ('NavalRecon', 'flavor_naval_recon'),
    ('NavalGrowth', 'flavor_naval_growth'),
    ('NavalTileImprovement', 'flavor_naval_tile_improvement'),
    ('Air', 'flavor_air'),
    ('AirCarrier', 'flavor_air_carrier'),
    ('Antiair', 'flavor_antiair'),
    ('Airlift', 'flavor_airlift'),
    ('Expansion', 'flavor_expansion'),
    ('Growth', 'flavor_growth'),
    ('TileImprovement', 'flavor_tile_improvement'),
    ('Infrastructure', 'flavor_infrastructure'),
    ('Production', 'flavor_production'),
    ('WaterConnection', 'flavor_water_connection'),
    ('Gold', 'flavor_gold'),
    ('Science', 'flavor_science'),
    ('Culture', 'flavor_culture'),
    ('Happiness', 'flavor_happiness'),
    ('GreatPeople', 'flavor_great_people'),
    ('Wonder', 'flavor_wonder'),
    ('Religion', 'flavor_religion'),
    ('Diplomacy', 'flavor_diplomacy'),
    ('Spaceship', 'flavor_spaceship'),
    ('Espionage', 'flavor_espionage'),
]

# Define field mappings for turn-based data
TURN_FIELD_MAPPINGS = {
    'experiment': None,          # Experiment name from folder
    'game_id': None,            # Game ID from metadata
    'player_id': None,          # Player ID
    'civilization': None,       # Player's civilization name
    'turn': None,              # Current turn number
    'max_turn': None,          # Maximum turn in the game (static)
    'score': None,             # Player's score at this turn
    'rank': None,              # Player's rank at this turn
    'max_score': None,         # Maximum score among all players at this turn
    'cities': None,            # Number of cities
    'population': None,        # Total population
    'territory': None,         # Territory tiles controlled
    'technologies': None,      # Number of technologies researched
    'military_strength': None, # Military strength value
    'gold': None,              # Total gold accumulated
    'gold_per_turn': None,     # Gold per turn
    'production_per_turn': None,  # Production per turn
    'food_per_turn': None,     # Food per turn
    'happiness_percentage': None,  # Happiness percentage (0-100)
    'culture_per_turn': None,  # Culture per turn
    'science_per_turn': None,  # Science per turn
    'tourism_per_turn': None,  # Tourism per turn
    'faith_per_turn': None,    # Faith per turn
    'policies': None,          # Total number of policies adopted
    'votes': None,             # World Congress votes
    'religion_percentage': None,  # Percentage (0-100) of global cities that have adopted player's religion
    'minor_allies': None,      # Number of city-state allies
    'is_winner': None,         # Whether this player won the game
    # FlavorChanges fields
    'is_changed': None,        # 1 if this turn has actual flavor number changes, 0 otherwise
    **{csv_col: None for _, csv_col in FLAVOR_COLUMNS},
    'grand_strategy': None,    # AI grand strategy at time of change
    'rationale': None,         # Rationale for flavor change
}

def _fetch_flavor_events(cursor, major_players):
    """
    Pre-fetch all FlavorChanges rows for major players (deduplicated per player+turn).
    Returns per-player sorted event lists for carry-forward processing.

    Deduplication: for each (player, turn), later rows override earlier ones,
    except a non-changed row never overrides a changed one.

    Returns:
        dict: {player_id: [(turn, flavor_values_tuple, grand_strategy, rationale, changes_json, is_changed), ...]}
              sorted by turn ascending. Empty dict if table doesn't exist.
    """
    placeholders = ','.join(['?'] * len(major_players))
    db_cols = ', '.join(f'fc.{db_col}' for db_col, _ in FLAVOR_COLUMNS)

    try:
        cursor.execute(f"""
            SELECT fc.Key, fc.Turn, {db_cols},
                   fc.GrandStrategy, fc.Rationale, fc.Changes
            FROM FlavorChanges fc
            WHERE fc.Key IN ({placeholders})
            ORDER BY fc.Key, fc.Turn, fc.ID
        """, major_players)
    except Exception:
        return {}

    # Deduplicate per (player, turn): keep latest row, but don't let a
    # non-changed row override a changed one.
    # When a duplicate would overwrite an entry, try to reassign
    # the earlier (displaced) entry to turn-1 instead of dropping it.
    latest = {}  # (player_id, turn) -> (flavor_values, grand_strategy, rationale, changes_json, is_changed)
    for row in cursor.fetchall():
        player_id = row[0]
        turn = row[1]
        flavor_values = row[2:2 + len(FLAVOR_COLUMNS)]
        grand_strategy = row[2 + len(FLAVOR_COLUMNS)]
        rationale = row[3 + len(FLAVOR_COLUMNS)]
        changes_json = row[4 + len(FLAVOR_COLUMNS)]
        is_changed = 0 if changes_json in ('[]', '["Rationale"]') else 1

        key = (player_id, turn)
        existing = latest.get(key)
        if existing is None:
            latest[key] = (flavor_values, grand_strategy, rationale, changes_json, is_changed)
        elif existing[4] == 1 and is_changed == 0:
            continue  # don't let a non-changed row override a changed one
        else:
            # New entry takes the current slot; cascade displaced entry backward
            displaced = existing
            latest[key] = (flavor_values, grand_strategy, rationale, changes_json, is_changed)

            # Iteratively reassign displaced entries to turn-1, turn-2, etc.
            cascade_turn = turn - 1
            while displaced is not None:
                prev_key = (player_id, cascade_turn)
                prev_existing = latest.get(prev_key)

                if prev_existing is None:
                    # Empty slot found — place displaced entry and stop
                    latest[prev_key] = displaced
                    break
                elif prev_existing[4] == 1 and displaced[4] == 0:
                    # Non-changed can't override changed — chain breaks, drop displaced
                    break
                else:
                    # Displace existing entry, continue cascading
                    latest[prev_key] = displaced
                    displaced = prev_existing
                    cascade_turn -= 1

    # Convert to per-player sorted event lists
    result = {}
    for (player_id, turn), entry in sorted(latest.items()):
        if player_id not in result:
            result[player_id] = []
        result[player_id].append((turn, *entry))

    return result


def _build_flavor_lookup(flavor_events, major_players, all_turns):
    """
    Build a carry-forward flavor lookup from per-player event lists.

    For each (player, turn):
    - If a FlavorChanges event exists at exactly this turn: use its data, is_changed from event
    - If no event this turn but a prior event exists: carry forward flavor values and
      grand_strategy; rationale and flavor_changes are empty; is_changed=0
    - If no prior event exists: no entry (flavor columns will be empty in output)

    Returns:
        dict: {(player_id, turn): (flavor_values_tuple, grand_strategy, rationale, changes_json, is_changed)}
    """
    lookup = {}

    for player_id in major_players:
        events = flavor_events.get(player_id, [])
        if not events:
            continue

        event_idx = 0
        n_events = len(events)

        for turn in all_turns:
            # Advance pointer past all events up to and including this turn
            while event_idx < n_events and events[event_idx][0] <= turn:
                event_idx += 1

            if event_idx == 0:
                continue  # No events at or before this turn yet

            ev_turn, ev_flavors, ev_gs, ev_rationale, ev_changes, ev_is_changed = events[event_idx - 1]

            if ev_turn == turn:
                # Event at exactly this turn: full data
                lookup[(player_id, turn)] = (ev_flavors, ev_gs, ev_rationale, ev_changes, ev_is_changed)
            else:
                # Carry forward: inherit flavor values and grand_strategy, clear event-specific fields
                lookup[(player_id, turn)] = (ev_flavors, ev_gs, '', '', 0)

    return lookup


def extract_game_turn_data(db_path):
    """
    Extract turn-based data from a SQLite database.
    Optimized to process turn-by-turn with minimal queries.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        list: List of dictionaries, one per player per turn
    """
    turn_data = []

    conn, cursor = open_database_readonly(db_path)
    if not conn:
        return []

    try:
        # Get experiment name and game metadata
        experiment_name = get_experiment_from_path(db_path)

        cursor.execute("SELECT Key, Value FROM GameMetadata")
        metadata = dict(cursor.fetchall())
        game_id = metadata.get('gameId', 'N/A')

        # Pre-fetch player info (cached)
        player_info_cache = get_player_info_cache(cursor)
        major_players = [pid for pid, info in player_info_cache.items() if info['is_major']]

        if not major_players:
            print(f"  No major players found in {os.path.basename(db_path)}")
            return []

        # Get max turn for the game
        cursor.execute("SELECT MAX(Turn) FROM PlayerSummaries")
        max_turn = cursor.fetchone()[0]

        if max_turn is None:
            print(f"  No turn data found in {os.path.basename(db_path)}")
            return []

        # Get victory player ID for is_winner determination
        victory_player_id_raw = metadata.get('victoryPlayerID', 'N/A')
        try:
            victory_player_id = int(float(victory_player_id_raw))
        except (ValueError, TypeError):
            victory_player_id = -1  # Use -1 for invalid/missing victory player ID

        # Pre-fetch city religion data for all turns
        # Structure: {turn: {religion: count}}
        cursor.execute("""
            SELECT Turn, MajorityReligion, COUNT(*) as city_count
            FROM CityInformations
            WHERE MajorityReligion IS NOT NULL
            GROUP BY Turn, MajorityReligion
        """)
        city_religion_by_turn = {}
        for turn, religion, count in cursor.fetchall():
            if turn not in city_religion_by_turn:
                city_religion_by_turn[turn] = {}
            city_religion_by_turn[turn][religion] = count

        # Also get total city count per turn
        cursor.execute("""
            SELECT Turn, COUNT(*) as total_cities
            FROM CityInformations
            GROUP BY Turn
        """)
        total_cities_by_turn = dict(cursor.fetchall())

        # Pre-fetch city-state allies for all turns
        # Structure: {turn: {civilization: ally_count}}
        cursor.execute("""
            SELECT ps.Turn, ps.MajorAlly, COUNT(*) as ally_count
            FROM PlayerSummaries ps
            INNER JOIN PlayerInformations pi ON ps.Key = pi.Key
            WHERE pi.IsMajor = 0 AND ps.MajorAlly IS NOT NULL
            GROUP BY ps.Turn, ps.MajorAlly
        """)
        allies_by_turn = {}
        for turn, civ, count in cursor.fetchall():
            if turn not in allies_by_turn:
                allies_by_turn[turn] = {}
            allies_by_turn[turn][civ] = count

        # Pre-fetch city production and food data for all turns and major players
        # Structure: {turn: {player_id: {'production': total, 'food': total}}}
        placeholders = ','.join(['?'] * len(major_players))
        cursor.execute("""
            SELECT ci.Turn, pi.Key,
                   SUM(ci.ProductionPerTurn) as total_production,
                   SUM(ci.FoodPerTurn) as total_food
            FROM CityInformations ci
            INNER JOIN PlayerInformations pi ON ci.Owner = pi.Civilization
            WHERE pi.Key IN ({})
            GROUP BY ci.Turn, pi.Key
        """.format(placeholders), major_players)

        city_yields_by_turn = {}
        for turn, player_id, production, food in cursor.fetchall():
            if turn not in city_yields_by_turn:
                city_yields_by_turn[turn] = {}
            city_yields_by_turn[turn][player_id] = {
                'production': production if production is not None else 0,
                'food': food if food is not None else 0
            }

        # Get sorted list of all turns (needed for carry-forward)
        cursor.execute("SELECT DISTINCT Turn FROM PlayerSummaries ORDER BY Turn")
        all_turns = [row[0] for row in cursor.fetchall()]

        # Pre-fetch FlavorChanges events and build carry-forward lookup
        flavor_events = _fetch_flavor_events(cursor, major_players)
        flavor_by_player_turn = _build_flavor_lookup(flavor_events, major_players, all_turns)

        # Modified query to include ALL player-turn combinations, even for eliminated players
        # Uses CROSS JOIN to generate all combinations, then LEFT JOIN for actual data
        cursor.execute(f"""
            WITH all_turns AS (
                SELECT DISTINCT Turn FROM PlayerSummaries
            ),
            all_player_turns AS (
                SELECT t.Turn, p.Key, p.Civilization
                FROM all_turns t
                CROSS JOIN (
                    SELECT Key, Civilization
                    FROM PlayerInformations
                    WHERE Key IN ({placeholders})
                ) p
            ),
            latest_summaries AS (
                SELECT Turn, Key, MAX(ID) as MaxID
                FROM PlayerSummaries
                WHERE Key IN ({placeholders})
                GROUP BY Turn, Key
            )
            SELECT
                apt.Turn,
                apt.Key,
                COALESCE(ps.Score, 0) as Score,
                COALESCE(ps.Cities, 0) as Cities,
                COALESCE(ps.Population, 0) as Population,
                COALESCE(ps.Territory, 0) as Territory,
                COALESCE(ps.Technologies, 0) as Technologies,
                COALESCE(ps.MilitaryStrength, 0) as MilitaryStrength,
                COALESCE(ps.Gold, 0) as Gold,
                COALESCE(ps.GoldPerTurn, 0) as GoldPerTurn,
                COALESCE(ps.HappinessPercentage, 0) as HappinessPercentage,
                COALESCE(ps.CulturePerTurn, 0) as CulturePerTurn,
                COALESCE(ps.SciencePerTurn, 0) as SciencePerTurn,
                COALESCE(ps.TourismPerTurn, 0) as TourismPerTurn,
                COALESCE(ps.FaithPerTurn, 0) as FaithPerTurn,
                ps.PolicyBranches,
                COALESCE(ps.Votes, 0) as Votes,
                ps.FoundedReligion,
                apt.Civilization
            FROM all_player_turns apt
            LEFT JOIN latest_summaries ls
                ON apt.Turn = ls.Turn AND apt.Key = ls.Key
            LEFT JOIN PlayerSummaries ps
                ON ls.Turn = ps.Turn AND ls.Key = ps.Key AND ls.MaxID = ps.ID
            ORDER BY apt.Turn, ps.Score DESC
        """, major_players + major_players)

        # Process results turn-by-turn
        current_turn = None
        turn_players = []

        for row in cursor.fetchall():
            (turn_num, player_id, score, cities, pop, territory, tech, military,
             gold, gold_per_turn, happiness_percentage, culture_per_turn, science_per_turn,
             tourism_per_turn, faith_per_turn, policy_branches_json, votes, founded_religion, civilization) = row

            # When we hit a new turn, process the previous turn's data
            if current_turn != turn_num:
                if turn_players:
                    process_turn_group(
                        turn_players, turn_data, experiment_name, game_id,
                        max_turn, player_info_cache, victory_player_id,
                        city_religion_by_turn, total_cities_by_turn, allies_by_turn,
                        city_yields_by_turn, flavor_by_player_turn
                    )
                turn_players = []
                current_turn = turn_num

            # Count policies from PolicyBranches JSON
            policies_count = 0
            if policy_branches_json:
                try:
                    policy_branches = json.loads(policy_branches_json)
                    # Count total number of policies across all branches
                    for branch_policies in policy_branches.values():
                        if isinstance(branch_policies, list):
                            policies_count += len(branch_policies)
                except (json.JSONDecodeError, ValueError):
                    policies_count = 0

            # Accumulate this player's data for the current turn
            turn_players.append({
                'turn': turn_num,
                'player_id': player_id,
                'score': score if score is not None else 0,
                'cities': cities if cities is not None else 0,
                'population': pop if pop is not None else 0,
                'territory': territory if territory is not None else 0,
                'technologies': tech if tech is not None else 0,
                'military_strength': military if military is not None else 0,
                'gold': gold if gold is not None else 0,
                'gold_per_turn': gold_per_turn if gold_per_turn is not None else 0,
                'happiness_percentage': happiness_percentage if happiness_percentage is not None else 0,
                'culture_per_turn': culture_per_turn if culture_per_turn is not None else 0,
                'science_per_turn': science_per_turn if science_per_turn is not None else 0,
                'tourism_per_turn': tourism_per_turn if tourism_per_turn is not None else 0,
                'faith_per_turn': faith_per_turn if faith_per_turn is not None else 0,
                'policies': policies_count,
                'votes': votes if votes is not None else 0,
                'religion_percentage': founded_religion,  # Will be converted to percentage in process_turn_group
                'civilization': civilization  # Needed for ally lookup
            })

        # Process the last turn's data
        if turn_players:
            process_turn_group(
                turn_players, turn_data, experiment_name, game_id,
                max_turn, player_info_cache, victory_player_id,
                city_religion_by_turn, total_cities_by_turn, allies_by_turn,
                city_yields_by_turn, flavor_by_player_turn
            )

        conn.close()
        return turn_data

    except Exception as e:
        print(f"Error processing {os.path.basename(db_path)}: {e}")
        conn.close()
        return []


def process_turn_group(turn_players, turn_data, experiment_name, game_id, max_turn, player_info_cache, victory_player_id,
                        city_religion_by_turn, total_cities_by_turn, allies_by_turn, city_yields_by_turn,
                        flavor_by_player_turn):
    """
    Process a group of players from the same turn.
    Calculate ranks and max_score, then add records to turn_data.

    Args:
        turn_players: List of player data for a single turn (already sorted by score DESC)
        turn_data: Output list to append processed records to
        experiment_name: Name of the experiment
        game_id: Game ID
        max_turn: Maximum turn in the game
        player_info_cache: Cached player information
        victory_player_id: ID of the winning player
        city_religion_by_turn: Dict mapping turn -> religion -> city count
        total_cities_by_turn: Dict mapping turn -> total city count
        allies_by_turn: Dict mapping turn -> civilization -> ally count
        city_yields_by_turn: Dict mapping turn -> player_id -> {'production': total, 'food': total}
        flavor_by_player_turn: Dict mapping (player_id, turn) -> flavor row data
    """
    if not turn_players:
        return

    # Separate alive and eliminated players
    alive_players = [p for p in turn_players if p['score'] > 0]
    eliminated_players = [p for p in turn_players if p['score'] == 0]

    # Get max score from alive players only
    max_score = alive_players[0]['score'] if alive_players else 0

    # Rank alive players
    current_rank = 1
    last_score = None
    players_with_rank = []

    for i, player_data in enumerate(alive_players):
        # If score is different from last player, update rank
        if last_score is not None and player_data['score'] != last_score:
            current_rank = i + 1

        players_with_rank.append({
            **player_data,
            'rank': current_rank
        })

        last_score = player_data['score']

    # Add eliminated players with worst rank
    worst_rank = len(turn_players)  # Total number of players
    for player_data in eliminated_players:
        players_with_rank.append({
            **player_data,
            'rank': worst_rank
        })

    # Create final records with all fields
    for player_info in players_with_rank:
        player_id = player_info['player_id']
        turn_num = player_info['turn']
        civilization = player_info['civilization']
        founded_religion = player_info['religion_percentage']

        # Calculate religion adoption percentage
        religion_percentage = 0
        if founded_religion and founded_religion != 'Pantheon (Religion Possible)':
            total_cities = total_cities_by_turn.get(turn_num, 0)
            if total_cities > 0:
                cities_with_religion = city_religion_by_turn.get(turn_num, {}).get(founded_religion, 0)
                religion_percentage = round((cities_with_religion / total_cities) * 100, 2)

        # Count city-state allies
        minor_allies_count = allies_by_turn.get(turn_num, {}).get(civilization, 0)

        # Get production and food from city yields
        city_yields = city_yields_by_turn.get(turn_num, {}).get(player_id, {'production': 0, 'food': 0})
        production_per_turn = city_yields.get('production', 0)
        food_per_turn = city_yields.get('food', 0)

        # Look up FlavorChanges data for this (player, turn)
        # tuple: (flavor_values_tuple, grand_strategy, rationale, changes_json, is_changed)
        flavor_entry = flavor_by_player_turn.get((player_id, turn_num))

        if flavor_entry is not None:
            flavor_values, grand_strategy, rationale, flavor_changes, is_changed = flavor_entry
        else:
            flavor_values = (None,) * len(FLAVOR_COLUMNS)
            grand_strategy = None
            rationale = None
            flavor_changes = None
            is_changed = 0

        # Create the complete record
        record = {
            'experiment': experiment_name,
            'game_id': game_id,
            'player_id': player_id,
            'civilization': civilization,
            'turn': turn_num,
            'max_turn': max_turn,
            'score': player_info['score'],
            'rank': player_info['rank'],
            'max_score': max_score,
            'cities': player_info['cities'],
            'population': player_info['population'],
            'territory': player_info['territory'],
            'technologies': player_info['technologies'],
            'military_strength': player_info['military_strength'],
            'gold': player_info['gold'],
            'gold_per_turn': player_info['gold_per_turn'],
            'production_per_turn': production_per_turn,
            'food_per_turn': food_per_turn,
            'happiness_percentage': player_info['happiness_percentage'],
            'culture_per_turn': player_info['culture_per_turn'],
            'science_per_turn': player_info['science_per_turn'],
            'tourism_per_turn': player_info['tourism_per_turn'],
            'faith_per_turn': player_info['faith_per_turn'],
            'policies': player_info['policies'],
            'votes': player_info['votes'],
            'religion_percentage': religion_percentage,
            'minor_allies': minor_allies_count,
            'is_winner': 1 if player_id == victory_player_id else 0,
            'is_changed': is_changed,
        }

        # Add flavor columns
        for (_, csv_col), value in zip(FLAVOR_COLUMNS, flavor_values):
            record[csv_col] = value if value is not None else ''

        record['grand_strategy'] = grand_strategy if grand_strategy is not None else ''
        record['rationale'] = rationale if rationale is not None else ''

        turn_data.append(record)


def export_turn_data(db_files, available_game_ids):
    """
    Export turn-based data from multiple databases to CSV.

    Args:
        db_files: List of database file paths
        available_game_ids: Set of available game IDs

    Returns:
        int: Number of new rows added
    """
    output_file = 'turn_data.csv'
    expected_fieldnames = list(TURN_FIELD_MAPPINGS.keys())

    # Read existing data
    existing_data, existing_game_ids, structure_matches = read_existing_csv(output_file, expected_fieldnames)
    pruned_rows = 0

    if not structure_matches:
        print("Discarding existing turn data due to structure mismatch...")
        existing_data = []
        existing_game_ids = set()
    else:
        # Filter existing data
        filtered_data, kept_game_ids, pruned_rows, pruned_game_ids = filter_existing_data(
            existing_data, available_game_ids
        )

        if pruned_rows > 0:
            print(f"  Filtered out {pruned_rows} turn records from {len(pruned_game_ids)} games without database files")
            print(f"  Pruned game IDs: {sorted(list(pruned_game_ids))[:10]}{'...' if len(pruned_game_ids) > 10 else ''}")

        existing_data = filtered_data
        existing_game_ids = kept_game_ids
        print(f"Found {len(existing_data)} existing turn records from {len(existing_game_ids)} games")

    # Extract new turn data
    print("\nExtracting turn-based data...")
    new_turn_data = []
    skipped_count = 0
    processed_count = 0

    for i, db_file in enumerate(db_files):
        game_id = get_game_id_from_path(db_file)

        # Skip if already processed
        if game_id and should_skip_game(game_id, existing_game_ids):
            skipped_count += 1
            continue

        turn_rows = extract_game_turn_data(db_file)
        if turn_rows:
            new_turn_data.extend(turn_rows)
            processed_count += 1

            # Calculate number of turns (unique turn numbers in this game)
            unique_turns = len(set(row['turn'] for row in turn_rows))
            num_players = len(set(row['player_id'] for row in turn_rows))
            changed_rows = sum(1 for row in turn_rows if row['is_changed'])

            print(f"Processed: {os.path.basename(db_file)} ({num_players} players × {unique_turns} turns = {len(turn_rows)} records, {changed_rows} with flavor changes)")

    print(f"\nProcessed {processed_count} new databases")
    print(f"Skipped {skipped_count} databases that were already exported")

    # Combine and export data
    all_turn_data = existing_data + new_turn_data

    if new_turn_data or pruned_rows > 0:
        if write_csv_file(output_file, expected_fieldnames, all_turn_data):
            print(f"\nSuccessfully exported total of {len(all_turn_data)} turn records to {output_file}")
            if new_turn_data:
                print(f"  - {len(existing_data)} existing records")
                print(f"  - {len(new_turn_data)} new records added")

            # Count unique games and turns
            unique_games = set(row['game_id'] for row in all_turn_data if row.get('game_id') != 'N/A')
            if unique_games:
                print(f"Data now includes {len(unique_games)} unique games")

                # Calculate average turns per game
                total_turns = len(set((row['game_id'], row['turn']) for row in all_turn_data
                                     if row.get('game_id') != 'N/A'))
                avg_turns = total_turns / len(unique_games) if unique_games else 0
                print(f"Average turns per game: {avg_turns:.1f}")
    else:
        print(f"\nNo new turn data to export. Existing file contains {len(existing_data)} records.")

    return len(new_turn_data)
