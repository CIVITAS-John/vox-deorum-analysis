Analysis of LLM strategist behavior and performance in **Civilization V Vox Populi** (via the Vox Deorum platform). Each game features 8 players, with select slots controlled by LLM strategists and the rest by vanilla AI. The "4player" in the folder name refers to the experimental configuration, not the actual player count.

## Directory Structure

```
analysis/
├── behaviors/                # Behavioral analysis (strategy, nukes)
│   ├── nuke/                 # Nuclear weapon qualitative coding
│   │   ├── coding/           # Codebook, coded quotes, categories
│   │   └── replay/           # Game replays with nuke events
│   ├── panel_nuke_domination.ipynb
│   ├── strategy_profiles.ipynb
│   └── use_nuke_flavor_rationale.ipynb
├── exploratory/              # Descriptive & exploratory analyses
│   ├── panel_exploratory.ipynb
│   └── turn_exploratory.ipynb
├── models/                   # Win probability prediction
│   ├── models/               # Model implementations
│   │   ├── base_predictor.py
│   │   ├── grouped_mlp_model.py   # Primary model (MLP + softmax)
│   │   ├── lightgbm_model.py
│   │   ├── random_forest_model.py
│   │   └── xgboost_model.py
│   ├── utils/                # Training utilities
│   │   ├── data_utils.py
│   │   ├── model_evaluator.py
│   │   └── model_registry.py
│   ├── compare_models.py
│   ├── evaluate_model.py
│   └── tune_models.py
├── performance/              # Performance & strength analysis
│   ├── panel_score_ratio.ipynb
│   └── turn_predicted.ipynb
├── ratings/                  # Player rating systems
│   ├── bradley_terry.py      # Bradley-Terry MLE (via R)
│   ├── plackett_luce.py      # Plackett-Luce model
│   ├── matchups.py           # Head-to-head matrices
│   └── iterative_bt.py       # Iterative BT algorithm
├── extract/                  # Data extraction from game DBs
│   ├── __main__.py
│   ├── extract_panel.py      # Game-level player data
│   ├── extract_turns.py      # Turn-level data
│   └── utilities.py          # Extraction helpers
├── plot_utilities.py         # Shared plotting & data loading
├── requirements.txt
├── turn_data.csv             # Turn-level dataset (~1.6M rows)
├── panel_data.csv            # Game-level dataset (3,672 rows)
└── game_timestamps.csv       # Game metadata (474 games)
```

## Data Files

### turn_data.csv (~1.6M rows, one per player per turn per game)

| Column Group | Columns |
|---|---|
| Identifiers | `experiment`, `game_id`, `player_id`, `civilization`, `turn`, `max_turn` |
| Game state | `score`, `rank`, `max_score`, `cities`, `population`, `technologies`, `military_strength`, `gold`, `gold_per_turn`, `production_per_turn`, `food_per_turn`, `happiness_percentage`, `culture_per_turn`, `science_per_turn`, `tourism_per_turn`, `faith_per_turn`, `policies`, `votes`, `religion_percentage`, `minor_allies` |
| Prediction | `predicted_win_probability` (MLP softmax output) |
| Strategy flavors | `flavor_offense`, `flavor_defense`, `flavor_mobilization`, `flavor_city_defense`, `flavor_military_training`, `flavor_recon`, `flavor_ranged`, `flavor_mobile`, `flavor_nuke`, `flavor_use_nuke`, `flavor_naval*`, `flavor_air*`, `flavor_expansion`, `flavor_growth`, `flavor_tile_improvement`, `flavor_infrastructure`, `flavor_production`, `flavor_gold`, `flavor_science`, `flavor_culture`, `flavor_happiness`, `flavor_great_people`, `flavor_wonder`, `flavor_religion`, `flavor_diplomacy`, `flavor_spaceship`, `flavor_espionage` |
| AI reasoning | `grand_strategy`, `rationale` |

### panel_data.csv (~4,000 rows, one per player per game)

| Column Group | Columns |
|---|---|
| Victory | `victory_type`, `victory_player_id`, `is_winner` |
| Token usage | `input_tokens`, `reasoning_tokens`, `output_tokens` |
| Strategy changes | `strategy_changes`, `persona_changes`, `research_changes`, `policy_changes` |
| Nuclear prefs | `nuke`, `use_nuke` (max flavor values reached) |
| Victory ratios | `domination_ratio`, `culture_ratio`, `diplomatic_ratio`, `science_ratio` |
| Policy adoption | `tradition`, `authority`, `progress`, `fealty`, `statecraft`, `artistry`, `industry`, `imperialism`, `rationalism`, `freedom`, `autocracy`, `order` (turn first adopted) |

### game_timestamps.csv (474 games)

`game_id` (UUID), `timestamp` (Unix ms), `experiment` (condition name)

## Experimental Conditions

Each condition assigns specific LLM models to player slots in 8-player games:

| Player Type | Variants |
|---|---|
| Vanilla | Control AI (no LLM) |
| Deepseek-3.2 | Simple, Briefed |
| GLM-4.7 | Simple, Briefed |
| GPT-OSS-120B | Simple, Briefed |
| Kimi-K2.5 | Simple, Briefed |
| Minimax-M2.5 | Simple, Briefed |
| Qwen-3.5 | Simple, Briefed |
| Sonnet-4.5 | Simple, Briefed |

**Simple** = minimal prompt; **Briefed** = receives game context summary.

## Key Notebooks

| Notebook | Purpose |
|---|---|
| `performance/turn_predicted.ipynb` | Win probability trends, weighted strength, Bradley-Terry Elo ratings, matchup matrices, civilization-adjusted probabilities |
| `performance/panel_score_ratio.ipynb` | OLS regression on score ratio by player type and civilization; forest plots |
| `exploratory/panel_exploratory.ipynb` | Descriptive stats: win rates, victory types, ideology, survival, grand strategy profiles |
| `exploratory/turn_exploratory.ipynb` | Metric progressions over turns (score, rank, cities, population, tech, policies) |
| `behaviors/strategy_profiles.ipynb` | Ideology choices, nuclear preferences, victory pursuit over time, adaptation rates |
| `behaviors/use_nuke_flavor_rationale.ipynb` | Nuke flavor vs. actual game decisions |
| `behaviors/panel_nuke_domination.ipynb` | Nuclear behavior and domination outcomes |

## Predictive Model

The primary model is `models/models/grouped_mlp_model.py`:

- **Input**: Turn-level game state features + strategy flavors for all players in a game-turn group
- **Architecture**: MLP with residual connections; utility network produces per-player scalar, then softmax across players in the same `(game_id, turn)` group
- **Loss**: Group-wise cross-entropy (predicts actual game winner)
- **Output**: Per-player victory probability (sums to 1.0 per turn)

## Rating Systems

- **Bradley-Terry MLE** (`ratings/bradley_terry.py`): Pairwise decomposition of game results, weighted by score margin, fitted via R's `BradleyTerry2` package. Outputs Elo ratings (1500 baseline).
- **Matchup matrices** (`ratings/matchups.py`): Head-to-head empirical win rates and mean strength differences between all player type pairs.

## Key Metrics

| Metric | Definition |
|---|---|
| Score Ratio | Player score / highest score in game (0–1) |
| Relative Strength | Player strength / game leader strength (0–1) |
| Weighted Strength | Quadratic-weighted average of win probability (emphasizes late game) |
| Victory Probability | MLP softmax output per player per turn |
| Elo Rating | Bradley-Terry "Worth" converted to Elo scale |
| Grand Strategy Ratio | Proportion of turns pursuing each victory condition |

## Dependencies

**Python** (see `requirements.txt`): pandas, numpy, scipy, statsmodels, matplotlib, seaborn, plotly, jupyter, ipykernel

**R** (called via subprocess): `BradleyTerry2`, `tidyverse`

## Shared Utilities

`plot_utilities.py` provides:

- `load_turn_data()` / `load_panel_data()` — data loading with player type mapping
- `plot_bar_chart()`, `plot_grouped_bar_chart()` — categorical comparisons
- `plot_forest_plot()` — regression coefficient visualization
- `plot_metric_over_time()` — turn progression with 95% CIs
- `plot_strategy_radar_charts()` — multi-axis strategy profiles
- `plot_matchup_heatmap()` — head-to-head matrix visualization
- `prepare_coefficient_data()` — regression output formatting
- `CONDITION_PLAYER_MAPPING` — maps experiment names to per-slot LLM assignments
