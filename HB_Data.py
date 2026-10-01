from pathlib import Path
import re
import unicodedata

import nflreadpy as nfl
import numpy as np
import pandas as pd


# ==================================================
# SETTINGS
# ==================================================

SEASON = 2025

# Full-confidence thresholds for individual HB metrics.
RUSH_QUALIFIER = 140
RECEIVING_QUALIFIER = 25
TOUCH_QUALIFIER = 100
ROUTE_QUALIFIER = 200
BLOCK_QUALIFIER = 50

# Final advanced-vs-legacy confidence.
# This is a starting point and can be tuned after we inspect the first HB rankings.
FULL_CONFIDENCE_OPPORTUNITIES = 240

OUTPUT_FILE = f"HB_Advanced_{SEASON}.xlsx"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PARQUET_OUTPUT = OUTPUT_DIR / f"HB_{SEASON}.parquet"

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

PBP_CACHE = CACHE_DIR / f"pbp_{SEASON}.parquet"

HISTORICAL_SEASONS = list(
    range(SEASON - 4, SEASON + 1)
)

HIST_PBP_CACHE = (
    CACHE_DIR /
    f"pbp_{HISTORICAL_SEASONS[0]}_{HISTORICAL_SEASONS[-1]}.parquet"
)

LEGACY_DB_FILE = "2026_NFLActive.xlsx"

BASE_ADVANCED_OVR_WEIGHT = 0.60
NEUTRAL_PERCENTILE = 50.0
HB_PRIME_AGE = 27.0
MISSING_LEGACY_RATING = 65.0

# Rushing is the HB baseline.
# Receiving and blocking modify that baseline based on quality and usage,
# similar to the mobility adjustment in the QB model.
RECEIVING_BOOST_MULTIPLIER = 1.25
BLOCKING_BOOST_MULTIPLIER = 0.50

# Used only to decide how much trust to place in the current advanced model
# while some planned sources are still unavailable.
RUSHING_COVERAGE_WEIGHT = 0.70
RECEIVING_COVERAGE_WEIGHT = 0.25
BLOCKING_COVERAGE_WEIGHT = 0.05

# If/when pass-block data is added, compare HB blocking only against skill players.
SKILL_POSITIONS = {"RB", "FB", "WR", "TE"}


# ==================================================
# HELPERS
# ==================================================

def normalize_player_name(name):
    if pd.isna(name):
        return ""

    name = str(name)

    name = unicodedata.normalize(
        "NFKD",
        name
    ).encode(
        "ascii",
        "ignore"
    ).decode()

    name = name.lower()

    name = re.sub(
        r"[^a-z0-9 ]",
        "",
        name
    )

    suffixes = {
        "jr",
        "sr",
        "ii",
        "iii",
        "iv"
    }

    parts = [
        part
        for part in name.split()
        if part not in suffixes
    ]

    return " ".join(parts)


def percentile_vs_history(
    value,
    pool,
    higher_is_better=True
):
    if pd.isna(value):
        return np.nan

    pool = (
        pd.to_numeric(
            pool,
            errors="coerce"
        )
        .dropna()
    )

    if len(pool) == 0:
        return np.nan

    if higher_is_better:
        below = (pool < value).sum()
        equal = (pool == value).sum()
    else:
        below = (pool > value).sum()
        equal = (pool == value).sum()

    return (
        (
            below +
            0.5 * equal
        )
        /
        len(pool)
        *
        100
    )


def add_historical_percentile(
    current_df,
    historical_df,
    value_column,
    percentile_column,
    historical_eligible,
    higher_is_better=True
):
    """
    Every current HB with a raw value receives a percentile.

    The qualifier applies ONLY to the historical comparison pool.
    Low-volume current players are later regressed toward 50.
    """

    current_df[percentile_column] = np.nan

    if value_column not in historical_df.columns:
        return

    pool = historical_df.loc[
        historical_eligible,
        value_column
    ]

    valid = current_df[value_column].notna()

    current_df.loc[
        valid,
        percentile_column
    ] = current_df.loc[
        valid,
        value_column
    ].apply(
        lambda x:
        percentile_vs_history(
            x,
            pool,
            higher_is_better
        )
    )


def calculate_weighted_score(row, weights):
    weighted_total = 0.0
    available_weight = 0.0

    for metric, weight in weights.items():
        value = row.get(metric)

        if pd.notna(value) and weight > 0:
            weighted_total += value * weight
            available_weight += weight

    if available_weight == 0:
        return np.nan

    return weighted_total / available_weight


def calculate_available_weight(row, weights):
    available_weight = 0.0

    for metric, weight in weights.items():
        if pd.notna(row.get(metric)):
            available_weight += weight

    return available_weight


def first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def safe_numeric(df, column, default=np.nan):
    if column is None:
        return pd.Series(
            default,
            index=df.index,
            dtype="float64"
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def normalize_legacy_position(value):
    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

    if value in {"RB", "HB"}:
        return "HB"

    return value


def add_first_down_context(df):
    """
    Build simple rushing-situation buckets used to estimate expected
    first-down conversion rate from the five-year historical skill-player pool.
    """

    result = df.copy()

    if "down" not in result.columns:
        result["down"] = np.nan

    if "ydstogo" not in result.columns:
        result["ydstogo"] = np.nan

    if "goal_to_go" not in result.columns:
        result["goal_to_go"] = 0

    result["YTG_Bucket"] = pd.cut(
        pd.to_numeric(
            result["ydstogo"],
            errors="coerce"
        ),
        bins=[-np.inf, 1, 3, 6, 10, np.inf],
        labels=["1", "2-3", "4-6", "7-10", "11+"]
    )

    result["Down_Value"] = pd.to_numeric(
        result["down"],
        errors="coerce"
    )

    result["Goal_To_Go_Value"] = pd.to_numeric(
        result["goal_to_go"],
        errors="coerce"
    ).fillna(0)

    first_down_column = first_existing_column(
        result,
        [
            "first_down",
            "first_down_rush"
        ]
    )

    result["First_Down_Value"] = safe_numeric(
        result,
        first_down_column
    )

    return result


def build_first_down_expectation_table(skill_rushes):
    context = add_first_down_context(skill_rushes)

    valid = context[
        context["First_Down_Value"].notna()
        & context["Down_Value"].notna()
        & context["YTG_Bucket"].notna()
    ].copy()

    detailed = (
        valid
        .groupby(
            [
                "Down_Value",
                "YTG_Bucket",
                "Goal_To_Go_Value"
            ],
            observed=True
        )["First_Down_Value"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(
            columns={
                "mean": "Expected_First_Down_Rate",
                "size": "Expectation_Sample"
            }
        )
    )

    # Avoid highly unstable tiny context cells.
    detailed.loc[
        detailed["Expectation_Sample"] < 25,
        "Expected_First_Down_Rate"
    ] = np.nan

    down_ytg = (
        valid
        .groupby(
            [
                "Down_Value",
                "YTG_Bucket"
            ],
            observed=True
        )["First_Down_Value"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "First_Down_Value":
                    "Fallback_First_Down_Rate"
            }
        )
    )

    overall = valid["First_Down_Value"].mean()

    return detailed, down_ytg, overall


def apply_first_down_expectation(
    rushes,
    detailed_table,
    fallback_table,
    overall_rate
):
    context = add_first_down_context(rushes)

    context = context.merge(
        detailed_table,
        on=[
            "Down_Value",
            "YTG_Bucket",
            "Goal_To_Go_Value"
        ],
        how="left"
    )

    context = context.merge(
        fallback_table,
        on=[
            "Down_Value",
            "YTG_Bucket"
        ],
        how="left"
    )

    context["Expected_First_Down_Rate"] = (
        context["Expected_First_Down_Rate"]
        .fillna(
            context["Fallback_First_Down_Rate"]
        )
        .fillna(overall_rate)
    )

    context["First_Down_Over_Expected_Play"] = (
        context["First_Down_Value"]
        -
        context["Expected_First_Down_Rate"]
    )

    return context


# ==================================================
# LOAD PLAYER MAP EARLY
# ==================================================

players = nfl.load_players().to_pandas()

player_columns = [
    "gsis_id",
    "display_name",
    "position",
    "pfr_id",
    "pff_id",
    "espn_id",
    "birth_date"
]

players = (
    players[player_columns]
    .drop_duplicates("gsis_id")
)

player_position_map = players[
    [
        "gsis_id",
        "position"
    ]
].rename(
    columns={
        "gsis_id": "Player_ID_Map",
        "position": "Mapped_Position"
    }
)


# ==================================================
# LOAD / CACHE PLAY-BY-PLAY
# ==================================================

if PBP_CACHE.exists():
    pbp = pd.read_parquet(PBP_CACHE)
else:
    pbp = nfl.load_pbp([SEASON]).to_pandas()
    pbp.to_parquet(
        PBP_CACHE,
        index=False
    )

pbp = pbp[
    pbp["season_type"] == "REG"
].copy()

pbp = pbp.sort_values(
    [
        "week",
        "game_id",
        "play_id"
    ]
)


# ==================================================
# CURRENT-SEASON PLAYER STATS
# BASE HB POPULATION
# ==================================================

season_stats = nfl.load_player_stats(
    [SEASON],
    summary_level="reg"
).to_pandas()

current_hb = season_stats[
    season_stats["position"] == "RB"
].copy()

carries_column = first_existing_column(
    current_hb,
    [
        "carries",
        "rushing_attempts",
        "rush_attempts"
    ]
)

targets_column = first_existing_column(
    current_hb,
    ["targets"]
)

receptions_column = first_existing_column(
    current_hb,
    ["receptions"]
)

rushing_yards_column = first_existing_column(
    current_hb,
    ["rushing_yards"]
)

rushing_tds_column = first_existing_column(
    current_hb,
    ["rushing_tds", "rushing_touchdowns"]
)

receiving_yards_column = first_existing_column(
    current_hb,
    ["receiving_yards"]
)

receiving_tds_column = first_existing_column(
    current_hb,
    ["receiving_tds", "receiving_touchdowns"]
)

if carries_column is not None:
    current_hb = current_hb.sort_values(
        carries_column,
        ascending=False
    )

current_hb = current_hb.drop_duplicates(
    "player_id"
)

if "team" in current_hb.columns:
    current_hb_team = current_hb["team"]
elif "recent_team" in current_hb.columns:
    current_hb_team = current_hb["recent_team"]
else:
    current_hb_team = pd.Series(
        np.nan,
        index=current_hb.index,
        dtype="object"
    )

hb_stats = pd.DataFrame({
    "Player_ID":
        current_hb["player_id"],

    "Short_Name":
        current_hb["player_name"],

    "Player":
        current_hb[
            "player_display_name"
        ].fillna(
            current_hb["player_name"]
        ),

    "Team_2025":
        current_hb_team,

    "Position":
        "HB",

    "Season_Rush_Attempts":
        safe_numeric(
            current_hb,
            carries_column,
            default=0
        ),

    "Season_Targets":
        safe_numeric(
            current_hb,
            targets_column,
            default=0
        ),

    "Receptions":
        safe_numeric(
            current_hb,
            receptions_column,
            default=0
        ),

    "Rushing_Yards":
        safe_numeric(
            current_hb,
            rushing_yards_column,
            default=0
        ),

    "Rushing_TDs":
        safe_numeric(
            current_hb,
            rushing_tds_column,
            default=0
        ),

    "Receiving_Yards":
        safe_numeric(
            current_hb,
            receiving_yards_column,
            default=0
        ),

    "Receiving_TDs":
        safe_numeric(
            current_hb,
            receiving_tds_column,
            default=0
        )
})


# ==================================================
# CURRENT RUSHING PBP METRICS
# ==================================================

rush_plays = pbp[
    (pbp["rush_attempt"] == 1)
    & (pbp["rusher_player_id"].notna())
].copy()

if "qb_scramble" in rush_plays.columns:
    rush_plays = rush_plays[
        rush_plays["qb_scramble"].fillna(0) != 1
    ].copy()

if "qb_kneel" in rush_plays.columns:
    rush_plays = rush_plays[
        rush_plays["qb_kneel"].fillna(0) != 1
    ].copy()

rush_plays = rush_plays.merge(
    player_position_map,
    left_on="rusher_player_id",
    right_on="Player_ID_Map",
    how="left"
)

rush_plays = rush_plays[
    rush_plays["Mapped_Position"] == "RB"
].copy()

rush_stats = (
    rush_plays
    .groupby("rusher_player_id")
    .agg(
        PBP_Rush_Attempts=(
            "rusher_player_id",
            "size"
        ),

        Total_Rushing_EPA=(
            "epa",
            "sum"
        ),

        EPA_per_Attempt=(
            "epa",
            "mean"
        ),

        Rush_Success_Rate=(
            "success",
            "mean"
        ),

        Team_From_Rush_PBP=(
            "posteam",
            "last"
        )
    )
    .reset_index()
    .rename(
        columns={
            "rusher_player_id":
                "Player_ID"
        }
    )
)

hb_stats = hb_stats.merge(
    rush_stats,
    on="Player_ID",
    how="left"
)

hb_stats["Rush_Attempts"] = (
    hb_stats["PBP_Rush_Attempts"]
    .fillna(
        hb_stats["Season_Rush_Attempts"]
    )
    .fillna(0)
)

hb_stats["Rush_Success_Rate"] = (
    hb_stats["Rush_Success_Rate"]
    * 100
)

hb_stats["Team_2025"] = (
    hb_stats["Team_From_Rush_PBP"]
    .fillna(
        hb_stats["Team_2025"]
    )
)


# ==================================================
# CURRENT RECEIVING PBP METRICS
# ==================================================

target_plays = pbp[
    pbp["receiver_player_id"].notna()
].copy()

target_plays = target_plays.merge(
    player_position_map,
    left_on="receiver_player_id",
    right_on="Player_ID_Map",
    how="left"
)

target_plays = target_plays[
    target_plays["Mapped_Position"] == "RB"
].copy()

receiving_stats = (
    target_plays
    .groupby("receiver_player_id")
    .agg(
        PBP_Targets=(
            "receiver_player_id",
            "size"
        ),

        Total_Receiving_EPA=(
            "epa",
            "sum"
        ),

        Receiving_EPA_per_Attempt=(
            "epa",
            "mean"
        ),

        Receiving_Success_Rate=(
            "success",
            "mean"
        ),

        Team_From_Target_PBP=(
            "posteam",
            "last"
        )
    )
    .reset_index()
    .rename(
        columns={
            "receiver_player_id":
                "Player_ID"
        }
    )
)

hb_stats = hb_stats.merge(
    receiving_stats,
    on="Player_ID",
    how="left"
)

hb_stats["Targets"] = (
    hb_stats["PBP_Targets"]
    .fillna(
        hb_stats["Season_Targets"]
    )
    .fillna(0)
)

hb_stats["Receiving_Success_Rate"] = (
    hb_stats["Receiving_Success_Rate"]
    * 100
)

hb_stats["Team_2025"] = (
    hb_stats["Team_From_Target_PBP"]
    .fillna(
        hb_stats["Team_2025"]
    )
)


# ==================================================
# PLAYER IDS / NAMES
# ==================================================

hb_stats = hb_stats.merge(
    players,
    left_on="Player_ID",
    right_on="gsis_id",
    how="left",
    suffixes=("", "_player_map")
)

hb_stats["Player"] = (
    hb_stats["display_name"]
    .fillna(
        hb_stats["Player"]
    )
)


# ==================================================
# FIVE-YEAR HISTORICAL PBP
# ==================================================

if HIST_PBP_CACHE.exists():
    hist_pbp = pd.read_parquet(
        HIST_PBP_CACHE
    )
else:
    hist_pbp = nfl.load_pbp(
        HISTORICAL_SEASONS
    ).to_pandas()

    hist_pbp.to_parquet(
        HIST_PBP_CACHE,
        index=False
    )

hist_pbp = hist_pbp[
    hist_pbp["season_type"] == "REG"
].copy()


# ==================================================
# FIRST-DOWN RATE OVER EXPECTED
# EXPECTATION MODEL: SKILL-POSITION RUSHES ONLY
# ==================================================

hist_skill_rushes = hist_pbp[
    (hist_pbp["rush_attempt"] == 1)
    & (hist_pbp["rusher_player_id"].notna())
].copy()

if "qb_scramble" in hist_skill_rushes.columns:
    hist_skill_rushes = hist_skill_rushes[
        hist_skill_rushes["qb_scramble"].fillna(0) != 1
    ].copy()

if "qb_kneel" in hist_skill_rushes.columns:
    hist_skill_rushes = hist_skill_rushes[
        hist_skill_rushes["qb_kneel"].fillna(0) != 1
    ].copy()

hist_skill_rushes = hist_skill_rushes.merge(
    player_position_map,
    left_on="rusher_player_id",
    right_on="Player_ID_Map",
    how="left"
)

hist_skill_rushes = hist_skill_rushes[
    hist_skill_rushes["Mapped_Position"].isin(
        SKILL_POSITIONS
    )
].copy()

(
    fd_expectation_table,
    fd_fallback_table,
    fd_overall_rate
) = build_first_down_expectation_table(
    hist_skill_rushes
)

current_rush_fd = apply_first_down_expectation(
    rush_plays,
    fd_expectation_table,
    fd_fallback_table,
    fd_overall_rate
)

current_fd_stats = (
    current_rush_fd
    .groupby("rusher_player_id")
    .agg(
        Actual_First_Down_Rate=(
            "First_Down_Value",
            "mean"
        ),

        Expected_First_Down_Rate=(
            "Expected_First_Down_Rate",
            "mean"
        ),

        First_Down_Rate_Over_Expected=(
            "First_Down_Over_Expected_Play",
            "mean"
        )
    )
    .reset_index()
    .rename(
        columns={
            "rusher_player_id":
                "Player_ID"
        }
    )
)

for column in [
    "Actual_First_Down_Rate",
    "Expected_First_Down_Rate",
    "First_Down_Rate_Over_Expected"
]:
    current_fd_stats[column] *= 100

hb_stats = hb_stats.merge(
    current_fd_stats,
    on="Player_ID",
    how="left"
)


# ==================================================
# NEXT GEN STATS — RUSHING
# RYOE / ATTEMPT
# ==================================================

try:
    ngs_rush = nfl.load_nextgen_stats(
        seasons=[SEASON],
        stat_type="rushing"
    ).to_pandas()

    if "season_type" in ngs_rush.columns:
        ngs_rush = ngs_rush[
            ngs_rush["season_type"] == "REG"
        ].copy()

    if "player_position" in ngs_rush.columns:
        ngs_rush = ngs_rush[
            ngs_rush["player_position"] == "RB"
        ].copy()

    # Prefer the NGS season-summary rows when they are supplied.
    if (
        "week" in ngs_rush.columns
        and (ngs_rush["week"] == 0).any()
    ):
        ngs_rush = ngs_rush[
            ngs_rush["week"] == 0
        ].copy()

    ngs_rush["rush_attempts"] = pd.to_numeric(
        ngs_rush["rush_attempts"],
        errors="coerce"
    ).fillna(0)

    ngs_rush["RYOE_Weighted"] = (
        pd.to_numeric(
            ngs_rush[
                "rush_yards_over_expected_per_att"
            ],
            errors="coerce"
        )
        *
        ngs_rush["rush_attempts"]
    )

    ngs_current = (
        ngs_rush
        .groupby("player_gsis_id")
        .agg(
            NGS_Rush_Attempts=(
                "rush_attempts",
                "sum"
            ),

            RYOE_Weighted=(
                "RYOE_Weighted",
                "sum"
            )
        )
        .reset_index()
        .rename(
            columns={
                "player_gsis_id":
                    "Player_ID"
            }
        )
    )

    ngs_current["RYOE_per_Attempt"] = (
        ngs_current["RYOE_Weighted"]
        /
        ngs_current[
            "NGS_Rush_Attempts"
        ].replace(0, np.nan)
    )

    hb_stats = hb_stats.merge(
        ngs_current[
            [
                "Player_ID",
                "RYOE_per_Attempt"
            ]
        ],
        on="Player_ID",
        how="left"
    )

except Exception as error:
    print(
        f"NGS rushing data failed: {error}"
    )

    hb_stats["RYOE_per_Attempt"] = np.nan


# ==================================================
# PFR ADVANCED RUSHING + RECEIVING
# MISSED TACKLES FORCED / TOUCH PROXY
# ==================================================

try:
    pfr_rush = nfl.load_pfr_advstats(
        seasons=[SEASON],
        stat_type="rush",
        summary_level="season"
    ).to_pandas()

    pfr_rush_id_column = (
        "pfr_id"
        if "pfr_id" in pfr_rush.columns
        else "pfr_player_id"
    )

    rush_broken_tackles_column = first_existing_column(
        pfr_rush,
        [
            "brk_tkl",
            "broken_tackles",
            "rushing_broken_tackles",
            "rushing_brk_tkl"
        ]
    )

    pfr_rush_clean = pd.DataFrame({
        "pfr_id":
            pfr_rush[pfr_rush_id_column],

        "PFR_Rush_Broken_Tackles":
            safe_numeric(
                pfr_rush,
                rush_broken_tackles_column,
                default=0
            )
    })

    pfr_rush_clean = pfr_rush_clean.drop_duplicates(
        "pfr_id"
    )

except Exception as error:
    print(
        f"PFR advanced rushing failed: {error}"
    )

    pfr_rush_clean = pd.DataFrame(
        columns=[
            "pfr_id",
            "PFR_Rush_Broken_Tackles"
        ]
    )

try:
    pfr_rec = nfl.load_pfr_advstats(
        seasons=[SEASON],
        stat_type="rec",
        summary_level="season"
    ).to_pandas()

    pfr_rec_id_column = (
        "pfr_id"
        if "pfr_id" in pfr_rec.columns
        else "pfr_player_id"
    )

    rec_broken_tackles_column = first_existing_column(
        pfr_rec,
        [
            "brk_tkl",
            "broken_tackles",
            "receiving_broken_tackles",
            "receiving_brk_tkl"
        ]
    )

    pfr_rec_clean = pd.DataFrame({
        "pfr_id":
            pfr_rec[pfr_rec_id_column],

        "PFR_Rec_Broken_Tackles":
            safe_numeric(
                pfr_rec,
                rec_broken_tackles_column,
                default=0
            )
    })

    pfr_rec_clean = pfr_rec_clean.drop_duplicates(
        "pfr_id"
    )

except Exception as error:
    print(
        f"PFR advanced receiving failed: {error}"
    )

    pfr_rec_clean = pd.DataFrame(
        columns=[
            "pfr_id",
            "PFR_Rec_Broken_Tackles"
        ]
    )

hb_stats = hb_stats.merge(
    pfr_rush_clean,
    on="pfr_id",
    how="left"
)

hb_stats = hb_stats.merge(
    pfr_rec_clean,
    on="pfr_id",
    how="left"
)

hb_stats["Total_Touches"] = (
    hb_stats["Rush_Attempts"].fillna(0)
    +
    hb_stats["Receptions"].fillna(0)
)

hb_stats["Total_Broken_Tackles"] = (
    hb_stats[
        "PFR_Rush_Broken_Tackles"
    ].fillna(0)
    +
    hb_stats[
        "PFR_Rec_Broken_Tackles"
    ].fillna(0)
)

hb_stats["Missed_Tackles_Forced_Per_Touch"] = (
    hb_stats["Total_Broken_Tackles"]
    /
    hb_stats["Total_Touches"].replace(0, np.nan)
)


# ==================================================
# FUTURE ROUTE / BLOCKING DATA
# ==================================================

# True route counts are not currently available from the nflreadpy sources
# being used in this project, so Yards/Route remains a placeholder.
hb_stats["Routes"] = np.nan
hb_stats["Yards_Per_Route"] = np.nan

# Pass Block Efficiency is intentionally left blank for now.
# When a trustworthy source is added, historical blocking percentiles MUST use
# only skill-position players (RB/FB/WR/TE), not offensive linemen.
hb_stats["Pass_Block_Snaps"] = np.nan
hb_stats["Pass_Block_Efficiency"] = np.nan


# ==================================================
# HISTORICAL HB RUSHING METRICS
# ==================================================

hist_hb_rushes = hist_skill_rushes[
    hist_skill_rushes["Mapped_Position"] == "RB"
].copy()

hist_hb_rushes = apply_first_down_expectation(
    hist_hb_rushes,
    fd_expectation_table,
    fd_fallback_table,
    fd_overall_rate
)

hist_hb = (
    hist_hb_rushes
    .groupby(
        [
            "season",
            "rusher_player_id"
        ]
    )
    .agg(
        Rush_Attempts=(
            "rusher_player_id",
            "size"
        ),

        EPA_per_Attempt=(
            "epa",
            "mean"
        ),

        Rush_Success_Rate=(
            "success",
            "mean"
        ),

        First_Down_Rate_Over_Expected=(
            "First_Down_Over_Expected_Play",
            "mean"
        )
    )
    .reset_index()
    .rename(
        columns={
            "rusher_player_id":
                "Player_ID"
        }
    )
)

hist_hb["Rush_Success_Rate"] *= 100
hist_hb["First_Down_Rate_Over_Expected"] *= 100

hist_hb = hist_hb.merge(
    players[
        [
            "gsis_id",
            "pfr_id"
        ]
    ],
    left_on="Player_ID",
    right_on="gsis_id",
    how="left"
)


# ==================================================
# HISTORICAL HB RECEIVING METRICS
# ==================================================

hist_targets = hist_pbp[
    hist_pbp[
        "receiver_player_id"
    ].notna()
].copy()

hist_targets = hist_targets.merge(
    player_position_map,
    left_on="receiver_player_id",
    right_on="Player_ID_Map",
    how="left"
)

hist_targets = hist_targets[
    hist_targets["Mapped_Position"] == "RB"
].copy()

hist_rec = (
    hist_targets
    .groupby(
        [
            "season",
            "receiver_player_id"
        ]
    )
    .agg(
        Targets=(
            "receiver_player_id",
            "size"
        ),

        Receptions=(
            "complete_pass",
            "sum"
        ),

        Receiving_EPA_per_Attempt=(
            "epa",
            "mean"
        ),

        Receiving_Success_Rate=(
            "success",
            "mean"
        )
    )
    .reset_index()
    .rename(
        columns={
            "receiver_player_id":
                "Player_ID"
        }
    )
)

hist_rec["Receiving_Success_Rate"] *= 100

hist_hb = hist_hb.merge(
    hist_rec,
    on=[
        "season",
        "Player_ID"
    ],
    how="left"
)

hist_hb["Targets"] = (
    hist_hb["Targets"]
    .fillna(0)
)

hist_hb["Receptions"] = (
    hist_hb["Receptions"]
    .fillna(0)
)

hist_hb["Total_Touches"] = (
    hist_hb["Rush_Attempts"]
    +
    hist_hb["Receptions"]
)


# ==================================================
# HISTORICAL NEXT GEN RUSHING
# ==================================================

try:
    hist_ngs = nfl.load_nextgen_stats(
        seasons=HISTORICAL_SEASONS,
        stat_type="rushing"
    ).to_pandas()

    if "season_type" in hist_ngs.columns:
        hist_ngs = hist_ngs[
            hist_ngs["season_type"] == "REG"
        ].copy()

    if "player_position" in hist_ngs.columns:
        hist_ngs = hist_ngs[
            hist_ngs["player_position"] == "RB"
        ].copy()

    if (
        "week" in hist_ngs.columns
        and (hist_ngs["week"] == 0).any()
    ):
        hist_ngs = hist_ngs[
            hist_ngs["week"] == 0
        ].copy()

    hist_ngs["rush_attempts"] = pd.to_numeric(
        hist_ngs["rush_attempts"],
        errors="coerce"
    ).fillna(0)

    hist_ngs["RYOE_Weighted"] = (
        pd.to_numeric(
            hist_ngs[
                "rush_yards_over_expected_per_att"
            ],
            errors="coerce"
        )
        *
        hist_ngs["rush_attempts"]
    )

    hist_ngs_season = (
        hist_ngs
        .groupby(
            [
                "season",
                "player_gsis_id"
            ]
        )
        .agg(
            NGS_Rush_Attempts=(
                "rush_attempts",
                "sum"
            ),

            RYOE_Weighted=(
                "RYOE_Weighted",
                "sum"
            )
        )
        .reset_index()
        .rename(
            columns={
                "player_gsis_id":
                    "Player_ID"
            }
        )
    )

    hist_ngs_season["RYOE_per_Attempt"] = (
        hist_ngs_season["RYOE_Weighted"]
        /
        hist_ngs_season[
            "NGS_Rush_Attempts"
        ].replace(0, np.nan)
    )

    hist_hb = hist_hb.merge(
        hist_ngs_season[
            [
                "season",
                "Player_ID",
                "RYOE_per_Attempt"
            ]
        ],
        on=[
            "season",
            "Player_ID"
        ],
        how="left"
    )

except Exception as error:
    print(
        f"Historical NGS rushing data failed: {error}"
    )

    hist_hb["RYOE_per_Attempt"] = np.nan


# ==================================================
# HISTORICAL PFR BROKEN TACKLES
# ==================================================

try:
    hist_pfr_rush = nfl.load_pfr_advstats(
        seasons=HISTORICAL_SEASONS,
        stat_type="rush",
        summary_level="season"
    ).to_pandas()

    hist_pfr_rush_id = (
        "pfr_id"
        if "pfr_id" in hist_pfr_rush.columns
        else "pfr_player_id"
    )

    hist_rush_broken_column = first_existing_column(
        hist_pfr_rush,
        [
            "brk_tkl",
            "broken_tackles",
            "rushing_broken_tackles",
            "rushing_brk_tkl"
        ]
    )

    hist_pfr_rush_clean = pd.DataFrame({
        "season":
            hist_pfr_rush["season"],

        "pfr_id":
            hist_pfr_rush[
                hist_pfr_rush_id
            ],

        "Rush_Broken_Tackles":
            safe_numeric(
                hist_pfr_rush,
                hist_rush_broken_column,
                default=0
            )
    })

except Exception as error:
    print(
        f"Historical PFR rushing failed: {error}"
    )

    hist_pfr_rush_clean = pd.DataFrame(
        columns=[
            "season",
            "pfr_id",
            "Rush_Broken_Tackles"
        ]
    )

try:
    hist_pfr_rec = nfl.load_pfr_advstats(
        seasons=HISTORICAL_SEASONS,
        stat_type="rec",
        summary_level="season"
    ).to_pandas()

    hist_pfr_rec_id = (
        "pfr_id"
        if "pfr_id" in hist_pfr_rec.columns
        else "pfr_player_id"
    )

    hist_rec_broken_column = first_existing_column(
        hist_pfr_rec,
        [
            "brk_tkl",
            "broken_tackles",
            "receiving_broken_tackles",
            "receiving_brk_tkl"
        ]
    )

    hist_pfr_rec_clean = pd.DataFrame({
        "season":
            hist_pfr_rec["season"],

        "pfr_id":
            hist_pfr_rec[
                hist_pfr_rec_id
            ],

        "Rec_Broken_Tackles":
            safe_numeric(
                hist_pfr_rec,
                hist_rec_broken_column,
                default=0
            )
    })

except Exception as error:
    print(
        f"Historical PFR receiving failed: {error}"
    )

    hist_pfr_rec_clean = pd.DataFrame(
        columns=[
            "season",
            "pfr_id",
            "Rec_Broken_Tackles"
        ]
    )

hist_pfr_tackles = hist_pfr_rush_clean.merge(
    hist_pfr_rec_clean,
    on=[
        "season",
        "pfr_id"
    ],
    how="outer"
)

hist_pfr_tackles["Total_Broken_Tackles"] = (
    hist_pfr_tackles[
        "Rush_Broken_Tackles"
    ].fillna(0)
    +
    hist_pfr_tackles[
        "Rec_Broken_Tackles"
    ].fillna(0)
)

hist_hb = hist_hb.merge(
    hist_pfr_tackles[
        [
            "season",
            "pfr_id",
            "Total_Broken_Tackles"
        ]
    ],
    on=[
        "season",
        "pfr_id"
    ],
    how="left"
)

hist_hb["Missed_Tackles_Forced_Per_Touch"] = (
    hist_hb["Total_Broken_Tackles"]
    /
    hist_hb["Total_Touches"].replace(0, np.nan)
)


# ==================================================
# FUTURE HISTORICAL ROUTE / BLOCKING DATA
# ==================================================

hist_hb["Routes"] = np.nan
hist_hb["Yards_Per_Route"] = np.nan
hist_hb["Pass_Block_Snaps"] = np.nan
hist_hb["Pass_Block_Efficiency"] = np.nan

# IMPORTANT FOR LATER:
# When Pass_Block_Efficiency becomes available, its historical comparison
# dataframe must be built from SKILL_POSITIONS = RB/FB/WR/TE only.


# ==================================================
# QUALIFICATION / CONFIDENCE
# ==================================================

hb_stats["Rush_Pct_Eligible"] = (
    hb_stats["Rush_Attempts"] >=
    RUSH_QUALIFIER
)

hb_stats["Receiving_Pct_Eligible"] = (
    hb_stats["Targets"] >=
    RECEIVING_QUALIFIER
)

hb_stats["Touch_Pct_Eligible"] = (
    hb_stats["Total_Touches"] >=
    TOUCH_QUALIFIER
)

hb_stats["Route_Pct_Eligible"] = False
hb_stats["Block_Pct_Eligible"] = False

hb_stats["Rush_Stat_Confidence"] = np.minimum(
    hb_stats["Rush_Attempts"] /
    RUSH_QUALIFIER,
    1.0
)

hb_stats["Receiving_Stat_Confidence"] = np.minimum(
    hb_stats["Targets"] /
    RECEIVING_QUALIFIER,
    1.0
)

hb_stats["Touch_Stat_Confidence"] = np.minimum(
    hb_stats["Total_Touches"] /
    TOUCH_QUALIFIER,
    1.0
)

hb_stats["Route_Stat_Confidence"] = np.nan
hb_stats["Block_Stat_Confidence"] = np.nan

hist_rush_eligible = (
    hist_hb["Rush_Attempts"] >=
    RUSH_QUALIFIER
)

hist_receiving_eligible = (
    hist_hb["Targets"].fillna(0) >=
    RECEIVING_QUALIFIER
)

hist_touch_eligible = (
    hist_hb["Total_Touches"].fillna(0) >=
    TOUCH_QUALIFIER
)


# ==================================================
# FIVE-YEAR HISTORICAL PERCENTILES
# ==================================================

add_historical_percentile(
    hb_stats,
    hist_hb,
    "EPA_per_Attempt",
    "EPA_Attempt_Pctl",
    hist_rush_eligible,
    True
)

add_historical_percentile(
    hb_stats,
    hist_hb,
    "Rush_Success_Rate",
    "Rush_Success_Pctl",
    hist_rush_eligible,
    True
)

add_historical_percentile(
    hb_stats,
    hist_hb,
    "RYOE_per_Attempt",
    "RYOE_Attempt_Pctl",
    hist_rush_eligible,
    True
)

add_historical_percentile(
    hb_stats,
    hist_hb,
    "Missed_Tackles_Forced_Per_Touch",
    "Missed_Tackles_Forced_Per_Touch_Pctl",
    hist_touch_eligible,
    True
)

add_historical_percentile(
    hb_stats,
    hist_hb,
    "First_Down_Rate_Over_Expected",
    "First_Down_Rate_Over_Expected_Pctl",
    hist_rush_eligible,
    True
)

add_historical_percentile(
    hb_stats,
    hist_hb,
    "Receiving_EPA_per_Attempt",
    "Receiving_EPA_Attempt_Pctl",
    hist_receiving_eligible,
    True
)

add_historical_percentile(
    hb_stats,
    hist_hb,
    "Receiving_Success_Rate",
    "Receiving_Success_Pctl",
    hist_receiving_eligible,
    True
)

# Route and blocking metrics remain blank until reliable sources are added.
hb_stats["Yards_Per_Route_Pctl"] = np.nan
hb_stats["Pass_Block_Efficiency_Pctl"] = np.nan


# ==================================================
# SAMPLE-SIZE REGRESSION
# ==================================================

RUSH_BASED_PERCENTILES = [
    "EPA_Attempt_Pctl",
    "Rush_Success_Pctl",
    "RYOE_Attempt_Pctl",
    "First_Down_Rate_Over_Expected_Pctl"
]

for column in RUSH_BASED_PERCENTILES:
    hb_stats[column + "_Raw"] = hb_stats[column]

    hb_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        hb_stats["Rush_Stat_Confidence"]
        *
        (
            hb_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )

hb_stats[
    "Missed_Tackles_Forced_Per_Touch_Pctl_Raw"
] = hb_stats[
    "Missed_Tackles_Forced_Per_Touch_Pctl"
]

hb_stats[
    "Missed_Tackles_Forced_Per_Touch_Pctl"
] = (
    NEUTRAL_PERCENTILE
    +
    hb_stats["Touch_Stat_Confidence"]
    *
    (
        hb_stats[
            "Missed_Tackles_Forced_Per_Touch_Pctl"
        ]
        -
        NEUTRAL_PERCENTILE
    )
)

RECEIVING_BASED_PERCENTILES = [
    "Receiving_EPA_Attempt_Pctl",
    "Receiving_Success_Pctl"
]

for column in RECEIVING_BASED_PERCENTILES:
    hb_stats[column + "_Raw"] = hb_stats[column]

    hb_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        hb_stats["Receiving_Stat_Confidence"]
        *
        (
            hb_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )

# Once routes are available, Yards_Per_Route_Pctl should be regressed using
# Route_Stat_Confidence. Once blocking snaps are available, Pass_Block_Efficiency
# should be regressed using Block_Stat_Confidence.


# ==================================================
# HB ADVANCED COMPONENT WEIGHTS — STARTING POINT
# ==================================================

RUSHING_WEIGHTS = {
    "EPA_Attempt_Pctl": 0.25,
    "Rush_Success_Pctl": 0.20,
    "RYOE_Attempt_Pctl": 0.25,
    "Missed_Tackles_Forced_Per_Touch_Pctl": 0.15,
    "First_Down_Rate_Over_Expected_Pctl": 0.15
}

RECEIVING_WEIGHTS = {
    "Receiving_EPA_Attempt_Pctl": 0.40,
    "Yards_Per_Route_Pctl": 0.35,
    "Receiving_Success_Pctl": 0.25
}

BLOCKING_WEIGHTS = {
    "Pass_Block_Efficiency_Pctl": 1.00
}

hb_stats["Rushing_Core_Pctl"] = hb_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=RUSHING_WEIGHTS
)

hb_stats["Receiving_Component_Pctl"] = hb_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=RECEIVING_WEIGHTS
)

hb_stats["Blocking_Component_Pctl"] = hb_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=BLOCKING_WEIGHTS
)


# ==================================================
# OPPORTUNITY SHARES
# ==================================================

hb_stats["Total_Opportunities"] = (
    hb_stats["Rush_Attempts"].fillna(0)
    +
    hb_stats["Targets"].fillna(0)
)

hb_stats["Rushing_Opportunity_Share"] = (
    hb_stats["Rush_Attempts"].fillna(0)
    /
    hb_stats["Total_Opportunities"].replace(0, np.nan)
)

hb_stats["Receiving_Opportunity_Share"] = (
    hb_stats["Targets"].fillna(0)
    /
    hb_stats["Total_Opportunities"].replace(0, np.nan)
)

# Future blocking usage denominator.
hb_stats["Pass_Block_Play_Share"] = np.nan


# ==================================================
# RECEIVING + BLOCKING ADJUSTMENTS
# ==================================================

hb_stats["Receiving_Adjustment"] = np.where(
    hb_stats["Receiving_Component_Pctl"].notna(),
    (
        hb_stats["Receiving_Component_Pctl"]
        -
        NEUTRAL_PERCENTILE
    )
    *
    hb_stats["Receiving_Opportunity_Share"].fillna(0)
    *
    RECEIVING_BOOST_MULTIPLIER,
    0.0
)

# Blocking is structurally ready but has no effect until both blocking quality
# and blocking usage data are available.
hb_stats["Blocking_Adjustment"] = np.where(
    hb_stats["Blocking_Component_Pctl"].notna()
    & hb_stats["Pass_Block_Play_Share"].notna(),
    (
        hb_stats["Blocking_Component_Pctl"]
        -
        NEUTRAL_PERCENTILE
    )
    *
    hb_stats["Pass_Block_Play_Share"].fillna(0)
    *
    BLOCKING_BOOST_MULTIPLIER,
    0.0
)

hb_stats["Advanced_Weighted_Pctl"] = (
    hb_stats["Rushing_Core_Pctl"]
    +
    hb_stats["Receiving_Adjustment"]
    +
    hb_stats["Blocking_Adjustment"]
)

hb_stats["Advanced_Weighted_Pctl"] = (
    hb_stats["Advanced_Weighted_Pctl"]
    .clip(lower=0, upper=100)
)


# ==================================================
# ADVANCED DATA COVERAGE
# ==================================================

hb_stats["Rushing_Available_Weight"] = hb_stats.apply(
    calculate_available_weight,
    axis=1,
    weights=RUSHING_WEIGHTS
)

hb_stats["Receiving_Available_Weight"] = hb_stats.apply(
    calculate_available_weight,
    axis=1,
    weights=RECEIVING_WEIGHTS
)

hb_stats["Blocking_Available_Weight"] = hb_stats.apply(
    calculate_available_weight,
    axis=1,
    weights=BLOCKING_WEIGHTS
)

hb_stats["Rushing_Data_Coverage"] = (
    hb_stats["Rushing_Available_Weight"]
    /
    sum(RUSHING_WEIGHTS.values())
)

hb_stats["Receiving_Data_Coverage"] = (
    hb_stats["Receiving_Available_Weight"]
    /
    sum(RECEIVING_WEIGHTS.values())
)

hb_stats["Blocking_Data_Coverage"] = (
    hb_stats["Blocking_Available_Weight"]
    /
    sum(BLOCKING_WEIGHTS.values())
)

hb_stats["Advanced_Data_Coverage"] = (
    hb_stats["Rushing_Data_Coverage"]
    * RUSHING_COVERAGE_WEIGHT
    +
    hb_stats["Receiving_Data_Coverage"]
    * RECEIVING_COVERAGE_WEIGHT
    +
    hb_stats["Blocking_Data_Coverage"]
    * BLOCKING_COVERAGE_WEIGHT
)

hb_stats["Advanced_Rating"] = (
    60
    +
    hb_stats["Advanced_Weighted_Pctl"]
    * 0.40
)


# ==================================================
# LOAD LEGACY DATABASE
# ==================================================

legacy = pd.read_excel(
    LEGACY_DB_FILE,
    sheet_name="Main"
)

legacy = legacy[
    [
        "Team",
        "Pos",
        "Player",
        "Age",
        "Madden",
        "PFF",
        "PFR"
    ]
].copy()

legacy["Name_Key"] = (
    legacy["Player"]
    .apply(normalize_player_name)
)

legacy["Pos_Key"] = (
    legacy["Pos"]
    .apply(normalize_legacy_position)
)

legacy["Team_Key"] = (
    legacy["Team"]
    .astype(str)
    .str.upper()
    .str.strip()
)

hb_stats["Name_Key"] = (
    hb_stats["Player"]
    .apply(normalize_player_name)
)

# Legacy database still uses "Kenneth Gainwell".
hb_stats.loc[
    hb_stats["Player_ID"] == "00-0036919",
    "Name_Key"
] = "kenneth gainwell"

hb_stats["Pos_Key"] = "HB"

hb_stats["Team_Key"] = (
    hb_stats["Team_2025"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ==================================================
# PRIMARY LEGACY MATCH
# NAME + POSITION + 2025 TEAM
# ==================================================

legacy_primary = (
    legacy
    .drop_duplicates(
        [
            "Name_Key",
            "Pos_Key",
            "Team_Key"
        ],
        keep=False
    )
    [
        [
            "Name_Key",
            "Pos_Key",
            "Team_Key",
            "Age",
            "Madden",
            "PFF",
            "PFR"
        ]
    ]
)

hb_stats = hb_stats.merge(
    legacy_primary,
    on=[
        "Name_Key",
        "Pos_Key",
        "Team_Key"
    ],
    how="left"
)

hb_stats["Legacy_Match_Status"] = np.where(
    hb_stats["Madden"].notna(),
    "NAME+POS+TEAM",
    "UNMATCHED"
)


# ==================================================
# FALLBACK LEGACY MATCH
# NAME + POSITION ONLY
# ==================================================

name_pos_counts = (
    legacy
    .groupby(
        [
            "Name_Key",
            "Pos_Key"
        ]
    )
    .size()
    .reset_index(
        name="Match_Count"
    )
)

legacy_fallback = legacy.merge(
    name_pos_counts,
    on=[
        "Name_Key",
        "Pos_Key"
    ],
    how="left"
)

legacy_fallback = legacy_fallback[
    legacy_fallback["Match_Count"] == 1
][
    [
        "Name_Key",
        "Pos_Key",
        "Age",
        "Madden",
        "PFF",
        "PFR"
    ]
].copy()

legacy_fallback = legacy_fallback.rename(
    columns={
        "Age": "Fallback_Age",
        "Madden": "Fallback_Madden",
        "PFF": "Fallback_PFF",
        "PFR": "Fallback_PFR"
    }
)

hb_stats = hb_stats.merge(
    legacy_fallback,
    on=[
        "Name_Key",
        "Pos_Key"
    ],
    how="left"
)

needs_fallback = hb_stats["Madden"].isna()

for column in [
    "Age",
    "Madden",
    "PFF",
    "PFR"
]:
    fallback_column = "Fallback_" + column

    hb_stats.loc[
        needs_fallback,
        column
    ] = hb_stats.loc[
        needs_fallback,
        fallback_column
    ]

fallback_success = (
    needs_fallback
    & hb_stats["Fallback_Madden"].notna()
)

hb_stats.loc[
    fallback_success,
    "Legacy_Match_Status"
] = "NAME+POS"


# ==================================================
# AGE FALLBACK FROM NFLVERSE BIRTH DATE
# ==================================================

AGE_REFERENCE_DATE = pd.Timestamp(
    "2026-09-09"
)

birth_dates = pd.to_datetime(
    hb_stats["birth_date"],
    errors="coerce"
)

calculated_age = (
    (
        AGE_REFERENCE_DATE
        -
        birth_dates
    ).dt.days
    /
    365.2425
).round(1)

hb_stats["Age_Source"] = np.select(
    [
        hb_stats["Age"].notna(),
        calculated_age.notna()
    ],
    [
        "LEGACY",
        "NFLVERSE_BIRTH_DATE"
    ],
    default="MISSING"
)

hb_stats["Age"] = (
    hb_stats["Age"]
    .fillna(calculated_age)
)

hb_stats = hb_stats.drop(
    columns=[
        "Name_Key",
        "Pos_Key",
        "Team_Key",
        "Fallback_Age",
        "Fallback_Madden",
        "Fallback_PFF",
        "Fallback_PFR"
    ],
    errors="ignore"
)


# ==================================================
# LEGACY RATING
# ==================================================

LEGACY_WEIGHTS = {
    "Madden": 0.5722,
    "PFF": 0.3611,
    "PFR": 0.0666
}


def calculate_legacy_rating(row):
    weighted_total = 0.0
    available_weight = 0.0

    if pd.notna(row["Madden"]):
        weighted_total += (
            row["Madden"]
            *
            LEGACY_WEIGHTS["Madden"]
        )
        available_weight += LEGACY_WEIGHTS["Madden"]

    if pd.notna(row["PFF"]):
        weighted_total += (
            row["PFF"]
            *
            1.1
            *
            LEGACY_WEIGHTS["PFF"]
        )
        available_weight += LEGACY_WEIGHTS["PFF"]

    if pd.notna(row["PFR"]):
        weighted_total += (
            row["PFR"]
            *
            5.2
            *
            LEGACY_WEIGHTS["PFR"]
        )
        available_weight += LEGACY_WEIGHTS["PFR"]

    if available_weight == 0:
        return np.nan

    return weighted_total / available_weight


hb_stats["Legacy_Rating"] = hb_stats.apply(
    calculate_legacy_rating,
    axis=1
)

hb_stats["Legacy_Fallback_Applied"] = (
    hb_stats["Legacy_Rating"].isna()
)

hb_stats["Legacy_Rating_Used"] = (
    hb_stats["Legacy_Rating"]
    .fillna(MISSING_LEGACY_RATING)
)

# ==================================================
# FINAL ADVANCED / LEGACY CONFIDENCE
# ==================================================

hb_stats["Opportunity_Confidence"] = np.minimum(
    hb_stats["Total_Opportunities"]
    /
    FULL_CONFIDENCE_OPPORTUNITIES,
    1.0
)

hb_stats["Actual_Advanced_Weight"] = (
    BASE_ADVANCED_OVR_WEIGHT
    *
    hb_stats["Opportunity_Confidence"]
    *
    hb_stats["Advanced_Data_Coverage"]
)

hb_stats["Actual_Legacy_Weight"] = (
    1
    -
    hb_stats["Actual_Advanced_Weight"]
)


def calculate_pre_bonus_overall(row):
    advanced = row["Advanced_Rating"]
    legacy = row["Legacy_Rating_Used"]
    advanced_weight = row["Actual_Advanced_Weight"]
    legacy_weight = row["Actual_Legacy_Weight"]

    if pd.notna(advanced) and pd.notna(legacy):
        return (
            advanced * advanced_weight
            +
            legacy * legacy_weight
        )

    if pd.notna(legacy):
        return legacy

    if pd.notna(advanced):
        return advanced

    return np.nan


hb_stats["Pre_Bonus_Overall"] = hb_stats.apply(
    calculate_pre_bonus_overall,
    axis=1
)


# ==================================================
# HISTORICAL RECORD BONUS
# ==================================================

# Intentionally zero for HB V1.
# Build HB-specific record categories only after the core model is behaving.
hb_stats["Historical_Record_Bonus"] = 0.0

hb_stats["Final_Overall"] = (
    hb_stats["Pre_Bonus_Overall"]
    +
    hb_stats["Historical_Record_Bonus"]
)


# ==================================================
# AGE-ADJUSTED VALUE
# ==================================================

hb_stats["Age_Adjusted_Overall"] = (
    hb_stats["Final_Overall"]
    *
    (
        (
            100
            +
            (
                HB_PRIME_AGE
                -
                hb_stats["Age"]
            )
        )
        /
        100
    )
)


# ==================================================
# FINAL OUTPUT
# ==================================================

output_columns = [
    "Player",
    "Player_ID",
    "Team_2025",
    "Position",
    "Age",
    "Age_Source",
    "Final_Overall",
    "Age_Adjusted_Overall",
    "Legacy_Rating",
    "Advanced_Rating",
    "Advanced_Weighted_Pctl",
    "Pre_Bonus_Overall",
    "Historical_Record_Bonus",

    "Rush_Attempts",
    "Targets",
    "Receptions",
    "Total_Touches",
    "Total_Opportunities",
    "Rushing_Yards",
    "Rushing_TDs",
    "Receiving_Yards",
    "Receiving_TDs",

    "Legacy_Match_Status",

    "Rush_Pct_Eligible",
    "Receiving_Pct_Eligible",
    "Touch_Pct_Eligible",
    "Route_Pct_Eligible",
    "Block_Pct_Eligible",

    "Rush_Stat_Confidence",
    "Receiving_Stat_Confidence",
    "Touch_Stat_Confidence",
    "Route_Stat_Confidence",
    "Block_Stat_Confidence",
    "Opportunity_Confidence",

    "Rushing_Opportunity_Share",
    "Receiving_Opportunity_Share",
    "Pass_Block_Play_Share",

    "EPA_per_Attempt",
    "EPA_Attempt_Pctl",
    "EPA_Attempt_Pctl_Raw",

    "Rush_Success_Rate",
    "Rush_Success_Pctl",
    "Rush_Success_Pctl_Raw",

    "RYOE_per_Attempt",
    "RYOE_Attempt_Pctl",
    "RYOE_Attempt_Pctl_Raw",

    "Missed_Tackles_Forced_Per_Touch",
    "Missed_Tackles_Forced_Per_Touch_Pctl",
    "Missed_Tackles_Forced_Per_Touch_Pctl_Raw",

    "Pass_Block_Snaps",
    "Pass_Block_Efficiency",
    "Pass_Block_Efficiency_Pctl",

    "Actual_First_Down_Rate",
    "Expected_First_Down_Rate",
    "First_Down_Rate_Over_Expected",
    "First_Down_Rate_Over_Expected_Pctl",
    "First_Down_Rate_Over_Expected_Pctl_Raw",

    "Receiving_EPA_per_Attempt",
    "Receiving_EPA_Attempt_Pctl",
    "Receiving_EPA_Attempt_Pctl_Raw",

    "Routes",
    "Yards_Per_Route",
    "Yards_Per_Route_Pctl",

    "Receiving_Success_Rate",
    "Receiving_Success_Pctl",
    "Receiving_Success_Pctl_Raw",

    "Rushing_Core_Pctl",
    "Receiving_Component_Pctl",
    "Blocking_Component_Pctl",
    "Receiving_Adjustment",
    "Blocking_Adjustment",

    "Rushing_Data_Coverage",
    "Receiving_Data_Coverage",
    "Blocking_Data_Coverage",
    "Advanced_Data_Coverage",
    "Actual_Advanced_Weight",
    "Actual_Legacy_Weight",

    "Total_Rushing_EPA",
    "Total_Receiving_EPA",

    "Madden",
    "PFF",
    "PFR",

    "pfr_id",
    "pff_id",
    "espn_id"
]

# Protect output against an optional source failing before its columns exist.
for column in output_columns:
    if column not in hb_stats.columns:
        hb_stats[column] = np.nan

missing_age_players = hb_stats[
    hb_stats["Age"].isna()
][
    [
        "Player",
        "Player_ID",
        "Team_2025"
    ]
]

if not missing_age_players.empty:
    print("\nPlayers still missing age:")
    print(
        missing_age_players.to_string(
            index=False
        )
    )

hb_output = hb_stats[
    output_columns
].copy()

hb_output = hb_output.round(
    {
        "Age": 1,
        "Legacy_Rating": 2,
        "Advanced_Weighted_Pctl": 2,
        "Advanced_Rating": 2,
        "Pre_Bonus_Overall": 2,
        "Historical_Record_Bonus": 2,
        "Final_Overall": 2,
        "Age_Adjusted_Overall": 2,

        "Rush_Stat_Confidence": 3,
        "Receiving_Stat_Confidence": 3,
        "Touch_Stat_Confidence": 3,
        "Route_Stat_Confidence": 3,
        "Block_Stat_Confidence": 3,
        "Opportunity_Confidence": 3,

        "Rushing_Opportunity_Share": 3,
        "Receiving_Opportunity_Share": 3,
        "Pass_Block_Play_Share": 3,

        "EPA_per_Attempt": 3,
        "EPA_Attempt_Pctl": 1,
        "EPA_Attempt_Pctl_Raw": 1,

        "Rush_Success_Rate": 2,
        "Rush_Success_Pctl": 1,
        "Rush_Success_Pctl_Raw": 1,

        "RYOE_per_Attempt": 3,
        "RYOE_Attempt_Pctl": 1,
        "RYOE_Attempt_Pctl_Raw": 1,

        "Missed_Tackles_Forced_Per_Touch": 3,
        "Missed_Tackles_Forced_Per_Touch_Pctl": 1,
        "Missed_Tackles_Forced_Per_Touch_Pctl_Raw": 1,

        "Pass_Block_Efficiency": 3,
        "Pass_Block_Efficiency_Pctl": 1,

        "Actual_First_Down_Rate": 2,
        "Expected_First_Down_Rate": 2,
        "First_Down_Rate_Over_Expected": 2,
        "First_Down_Rate_Over_Expected_Pctl": 1,
        "First_Down_Rate_Over_Expected_Pctl_Raw": 1,

        "Receiving_EPA_per_Attempt": 3,
        "Receiving_EPA_Attempt_Pctl": 1,
        "Receiving_EPA_Attempt_Pctl_Raw": 1,

        "Yards_Per_Route": 3,
        "Yards_Per_Route_Pctl": 1,

        "Receiving_Success_Rate": 2,
        "Receiving_Success_Pctl": 1,
        "Receiving_Success_Pctl_Raw": 1,

        "Rushing_Core_Pctl": 2,
        "Receiving_Component_Pctl": 2,
        "Blocking_Component_Pctl": 2,
        "Receiving_Adjustment": 2,
        "Blocking_Adjustment": 2,

        "Rushing_Data_Coverage": 3,
        "Receiving_Data_Coverage": 3,
        "Blocking_Data_Coverage": 3,
        "Advanced_Data_Coverage": 3,
        "Actual_Advanced_Weight": 3,
        "Actual_Legacy_Weight": 3,

        "Total_Rushing_EPA": 2,
        "Total_Receiving_EPA": 2,

        "Madden": 1,
        "PFF": 1,
        "PFR": 1
    }
)

hb_output = hb_output.sort_values(
    "Final_Overall",
    ascending=False,
    na_position="last"
)

top_25 = hb_output.head(25).copy()


# ==================================================
# SAVE PARQUET FOR FUTURE MASTER WORKBOOK
# ==================================================

hb_output.to_parquet(
    PARQUET_OUTPUT,
    index=False
)


# ==================================================
# EXCEL OUTPUT
# ==================================================

try:
    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        hb_output.to_excel(
            writer,
            sheet_name="All_HBs",
            index=False
        )

        top_25.to_excel(
            writer,
            sheet_name="Top_25_Check",
            index=False
        )

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = max(
                    (
                        len(str(cell.value))
                        if cell.value is not None
                        else 0
                    )
                    for cell in column_cells
                )

                column_letter = (
                    column_cells[0].column_letter
                )

                worksheet.column_dimensions[
                    column_letter
                ].width = min(
                    max_length + 2,
                    28
                )

    print(
        f"Created {OUTPUT_FILE} "
        f"with {len(hb_output)} HBs."
    )

    print(
        f"Created {PARQUET_OUTPUT} "
        "for the future master workbook."
    )

except PermissionError:
    print(
        f"ERROR: Close {OUTPUT_FILE} "
        "in Excel and run the script again."
    )
