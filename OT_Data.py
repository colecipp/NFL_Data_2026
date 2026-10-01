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

# Enough snaps for an individual OT metric to be compared confidently
# against the five-year historical pool.
SNAP_QUALIFIER = 300

# Enough offensive snaps for the current season to receive the maximum
# Advanced-vs-Legacy influence. We can tune this after seeing the first run.
FULL_CONFIDENCE_SNAPS = 750

OUTPUT_FILE = f"OT_Advanced_{SEASON}.xlsx"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PARQUET_OUTPUT = OUTPUT_DIR / f"OT_{SEASON}.parquet"

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

PBP_CACHE = CACHE_DIR / f"pbp_{SEASON}.parquet"
SNAP_CACHE = CACHE_DIR / f"snap_counts_{SEASON}.parquet"

HISTORICAL_SEASONS = list(
    range(SEASON - 4, SEASON + 1)
)

HIST_PBP_CACHE = (
    CACHE_DIR /
    f"pbp_{HISTORICAL_SEASONS[0]}_{HISTORICAL_SEASONS[-1]}.parquet"
)

HIST_SNAP_CACHE = (
    CACHE_DIR /
    f"snap_counts_{HISTORICAL_SEASONS[0]}_{HISTORICAL_SEASONS[-1]}.parquet"
)

LEGACY_DB_FILE = "2026_NFLActive.xlsx"

BASE_ADVANCED_OVR_WEIGHT = 0.60
NEUTRAL_PERCENTILE = 50.0
OT_PRIME_AGE = 29.0
MISSING_LEGACY_RATING = 65.0


# ==================================================
# OT ADVANCED WEIGHTS — STARTING POINT
# ==================================================
#
# Pass protection gets slightly more weight than run blocking,
# but the model still values complete tackles.
#
# Pancakes and penalties are normalized per 100 offensive snaps so
# full-time starters are not automatically helped/hurt simply by volume.
# ==================================================

OT_ADVANCED_WEIGHTS = {
    "Run_Block_Efficiency_Pctl": 0.24,
    "Pass_Block_Efficiency_Pctl": 0.27,
    "Avg_Time_To_Pressure_Pctl": 0.13,
    "Pressure_Rate_Over_Expected_Pctl": 0.14,
    "Pancake_Rate_Pctl": 0.07,
    "Penalty_Rate_Pctl": 0.08,
    "Run_Block_Disrupt_Rate_Pctl": 0.07,
}


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
    Every current OT with a raw value receives a percentile.

    The qualifier applies only to the historical comparison pool.
    Low-snap current players are later regressed toward 50.
    """

    current_df[percentile_column] = np.nan

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


def is_tackle_position(series):
    return (
        series
        .astype(str)
        .str.upper()
        .str.strip()
        .isin({"T", "OT", "LT", "RT"})
    )


def normalize_legacy_position(series):
    normalized = (
        series
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return normalized.replace({
        "T": "OT",
        "LT": "OT",
        "RT": "OT",
    })


# ==================================================
# PLAYER ID MAP
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

player_pfr_map = (
    players[
        [
            "gsis_id",
            "display_name",
            "position",
            "pfr_id",
            "pff_id",
            "espn_id",
            "birth_date"
        ]
    ]
    .dropna(subset=["pfr_id"])
    .drop_duplicates("pfr_id")
)


# ==================================================
# CURRENT SNAP COUNTS — BASE OT POPULATION
# ==================================================

if SNAP_CACHE.exists():
    snap_counts = pd.read_parquet(SNAP_CACHE)
else:
    snap_counts = nfl.load_snap_counts([SEASON]).to_pandas()
    snap_counts.to_parquet(
        SNAP_CACHE,
        index=False
    )

if "game_type" in snap_counts.columns:
    snap_counts = snap_counts[
        snap_counts["game_type"] == "REG"
    ].copy()

current_ot_snaps = snap_counts[
    is_tackle_position(snap_counts["position"])
].copy()

current_ot_snaps["offense_snaps"] = pd.to_numeric(
    current_ot_snaps["offense_snaps"],
    errors="coerce"
).fillna(0)

sort_columns = [
    column
    for column in ["week", "game_id"]
    if column in current_ot_snaps.columns
]

if sort_columns:
    current_ot_snaps = current_ot_snaps.sort_values(sort_columns)

current_ot = (
    current_ot_snaps
    .groupby("pfr_player_id", dropna=False)
    .agg(
        Snap_Name=("player", "last"),
        Team_2025=("team", "last"),
        Offense_Snaps=("offense_snaps", "sum")
    )
    .reset_index()
    .rename(
        columns={
            "pfr_player_id": "pfr_id"
        }
    )
)

current_ot = current_ot[
    current_ot["Offense_Snaps"] > 0
].copy()

current_ot = current_ot.merge(
    player_pfr_map,
    on="pfr_id",
    how="left",
    suffixes=("", "_player_map")
)

ot_stats = pd.DataFrame({
    "Player_ID": current_ot["gsis_id"],
    "Player": current_ot["display_name"].fillna(current_ot["Snap_Name"]),
    "Team_2025": current_ot["Team_2025"],
    "Position": "OT",
    "Offense_Snaps": current_ot["Offense_Snaps"],
    "pfr_id": current_ot["pfr_id"],
    "pff_id": current_ot["pff_id"],
    "espn_id": current_ot["espn_id"],
    "birth_date": current_ot["birth_date"],
})


# ==================================================
# CURRENT PLAY-BY-PLAY — PENALTIES
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

penalty_player_column = first_existing_column(
    pbp,
    [
        "penalty_player_id",
        "penalty_player_id_1"
    ]
)

if penalty_player_column is not None:
    penalty_plays = pbp[
        pbp[penalty_player_column].notna()
    ].copy()

    if (
        "penalty_team" in penalty_plays.columns
        and "posteam" in penalty_plays.columns
    ):
        penalty_plays = penalty_plays[
            penalty_plays["penalty_team"] ==
            penalty_plays["posteam"]
        ].copy()

    if "penalty" in penalty_plays.columns:
        penalty_flag = pd.to_numeric(
            penalty_plays["penalty"],
            errors="coerce"
        )

        if penalty_flag.notna().any():
            penalty_plays = penalty_plays[
                penalty_flag == 1
            ].copy()

    current_penalties = (
        penalty_plays
        .groupby(penalty_player_column)
        .size()
        .reset_index(name="Penalties")
        .rename(
            columns={
                penalty_player_column: "Player_ID"
            }
        )
    )

    ot_stats = ot_stats.merge(
        current_penalties,
        on="Player_ID",
        how="left"
    )
else:
    ot_stats["Penalties"] = np.nan

ot_stats["Penalties"] = (
    pd.to_numeric(
        ot_stats["Penalties"],
        errors="coerce"
    )
    .fillna(0)
)

ot_stats["Penalty_Rate"] = (
    ot_stats["Penalties"]
    /
    ot_stats["Offense_Snaps"].replace(0, np.nan)
    *
    100
)


# ==================================================
# METRICS NOT CLEANLY AVAILABLE FROM NFLREADPY V1
# ==================================================
#
# These columns are intentional placeholders. We do not substitute team-level
# pressure/rushing outcomes because those would look like individual OT stats
# without actually identifying the responsible blocker.
# ==================================================

ot_stats["Run_Block_Efficiency"] = np.nan
ot_stats["Pass_Block_Efficiency"] = np.nan
ot_stats["Avg_Time_To_Pressure"] = np.nan
ot_stats["Pressure_Rate_Over_Expected"] = np.nan
ot_stats["Pancakes"] = np.nan
ot_stats["Pancakes_Per_100_Snaps"] = np.nan
ot_stats["Run_Block_Disrupt_Rate"] = np.nan


# ==================================================
# CURRENT QUALIFICATION / CONFIDENCE
# ==================================================

ot_stats["Snap_Pct_Eligible"] = (
    ot_stats["Offense_Snaps"] >= SNAP_QUALIFIER
)

ot_stats["Snap_Stat_Confidence"] = np.minimum(
    ot_stats["Offense_Snaps"] /
    SNAP_QUALIFIER,
    1.0
)


# ==================================================
# FIVE-YEAR HISTORICAL SNAP COUNTS
# ==================================================

if HIST_SNAP_CACHE.exists():
    hist_snap_counts = pd.read_parquet(HIST_SNAP_CACHE)
else:
    hist_snap_counts = nfl.load_snap_counts(
        HISTORICAL_SEASONS
    ).to_pandas()

    hist_snap_counts.to_parquet(
        HIST_SNAP_CACHE,
        index=False
    )

if "game_type" in hist_snap_counts.columns:
    hist_snap_counts = hist_snap_counts[
        hist_snap_counts["game_type"] == "REG"
    ].copy()

hist_ot_snaps = hist_snap_counts[
    is_tackle_position(hist_snap_counts["position"])
].copy()

hist_ot_snaps["offense_snaps"] = pd.to_numeric(
    hist_ot_snaps["offense_snaps"],
    errors="coerce"
).fillna(0)

hist_ot = (
    hist_ot_snaps
    .groupby(
        [
            "season",
            "pfr_player_id"
        ],
        dropna=False
    )
    .agg(
        Offense_Snaps=("offense_snaps", "sum")
    )
    .reset_index()
    .rename(
        columns={
            "pfr_player_id": "pfr_id"
        }
    )
)

hist_ot = hist_ot.merge(
    player_pfr_map[
        [
            "gsis_id",
            "pfr_id"
        ]
    ],
    on="pfr_id",
    how="left"
)

hist_ot = hist_ot.rename(
    columns={
        "gsis_id": "Player_ID"
    }
)


# ==================================================
# FIVE-YEAR HISTORICAL PENALTIES
# ==================================================

if HIST_PBP_CACHE.exists():
    hist_pbp = pd.read_parquet(HIST_PBP_CACHE)
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

hist_penalty_player_column = first_existing_column(
    hist_pbp,
    [
        "penalty_player_id",
        "penalty_player_id_1"
    ]
)

if hist_penalty_player_column is not None:
    hist_penalty_plays = hist_pbp[
        hist_pbp[hist_penalty_player_column].notna()
    ].copy()

    if (
        "penalty_team" in hist_penalty_plays.columns
        and "posteam" in hist_penalty_plays.columns
    ):
        hist_penalty_plays = hist_penalty_plays[
            hist_penalty_plays["penalty_team"] ==
            hist_penalty_plays["posteam"]
        ].copy()

    if "penalty" in hist_penalty_plays.columns:
        hist_penalty_flag = pd.to_numeric(
            hist_penalty_plays["penalty"],
            errors="coerce"
        )

        if hist_penalty_flag.notna().any():
            hist_penalty_plays = hist_penalty_plays[
                hist_penalty_flag == 1
            ].copy()

    hist_penalties = (
        hist_penalty_plays
        .groupby(
            [
                "season",
                hist_penalty_player_column
            ]
        )
        .size()
        .reset_index(name="Penalties")
        .rename(
            columns={
                hist_penalty_player_column: "Player_ID"
            }
        )
    )

    hist_ot = hist_ot.merge(
        hist_penalties,
        on=[
            "season",
            "Player_ID"
        ],
        how="left"
    )
else:
    hist_ot["Penalties"] = np.nan

hist_ot["Penalties"] = (
    pd.to_numeric(
        hist_ot["Penalties"],
        errors="coerce"
    )
    .fillna(0)
)

hist_ot["Penalty_Rate"] = (
    hist_ot["Penalties"]
    /
    hist_ot["Offense_Snaps"].replace(0, np.nan)
    *
    100
)

# Future historical metrics.
hist_ot["Run_Block_Efficiency"] = np.nan
hist_ot["Pass_Block_Efficiency"] = np.nan
hist_ot["Avg_Time_To_Pressure"] = np.nan
hist_ot["Pressure_Rate_Over_Expected"] = np.nan
hist_ot["Pancakes"] = np.nan
hist_ot["Pancakes_Per_100_Snaps"] = np.nan
hist_ot["Run_Block_Disrupt_Rate"] = np.nan

hist_snap_eligible = (
    hist_ot["Offense_Snaps"] >= SNAP_QUALIFIER
)


# ==================================================
# FIVE-YEAR HISTORICAL PERCENTILES
# ==================================================

add_historical_percentile(
    ot_stats,
    hist_ot,
    "Penalty_Rate",
    "Penalty_Rate_Pctl",
    hist_snap_eligible,
    False
)

# Future metrics. These columns are ready for source additions later.
for percentile_column in [
    "Run_Block_Efficiency_Pctl",
    "Pass_Block_Efficiency_Pctl",
    "Avg_Time_To_Pressure_Pctl",
    "Pressure_Rate_Over_Expected_Pctl",
    "Pancake_Rate_Pctl",
    "Run_Block_Disrupt_Rate_Pctl",
]:
    ot_stats[percentile_column] = np.nan


# ==================================================
# SAMPLE-SIZE REGRESSION
# ==================================================

SNAP_BASED_PERCENTILES = [
    "Run_Block_Efficiency_Pctl",
    "Pass_Block_Efficiency_Pctl",
    "Avg_Time_To_Pressure_Pctl",
    "Pressure_Rate_Over_Expected_Pctl",
    "Pancake_Rate_Pctl",
    "Penalty_Rate_Pctl",
    "Run_Block_Disrupt_Rate_Pctl",
]

for column in SNAP_BASED_PERCENTILES:
    ot_stats[column + "_Raw"] = ot_stats[column]

    ot_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        ot_stats["Snap_Stat_Confidence"]
        *
        (
            ot_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )


# ==================================================
# ADVANCED SCORE + DATA COVERAGE
# ==================================================

ot_stats["Advanced_Weighted_Pctl"] = ot_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=OT_ADVANCED_WEIGHTS
)

TOTAL_ADVANCED_WEIGHT = sum(
    OT_ADVANCED_WEIGHTS.values()
)


def calculate_available_advanced_weight(row):
    available_weight = 0.0

    for metric, weight in OT_ADVANCED_WEIGHTS.items():
        if pd.notna(row.get(metric)):
            available_weight += weight

    return available_weight


ot_stats["Advanced_Available_Weight"] = (
    ot_stats.apply(
        calculate_available_advanced_weight,
        axis=1
    )
)

ot_stats["Advanced_Data_Coverage"] = (
    ot_stats["Advanced_Available_Weight"]
    /
    TOTAL_ADVANCED_WEIGHT
)

ot_stats["Advanced_Rating"] = (
    60
    +
    ot_stats["Advanced_Weighted_Pctl"] * 0.40
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

legacy["Pos_Key"] = normalize_legacy_position(
    legacy["Pos"]
)

legacy["Team_Key"] = (
    legacy["Team"]
    .astype(str)
    .str.upper()
    .str.strip()
)

ot_stats["Name_Key"] = (
    ot_stats["Player"]
    .apply(normalize_player_name)
)

ot_stats["Pos_Key"] = "OT"

ot_stats["Team_Key"] = (
    ot_stats["Team_2025"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ==================================================
# PRIMARY LEGACY MATCH — NAME + POSITION + TEAM
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

ot_stats = ot_stats.merge(
    legacy_primary,
    on=[
        "Name_Key",
        "Pos_Key",
        "Team_Key"
    ],
    how="left"
)

ot_stats["Legacy_Match_Status"] = np.where(
    ot_stats["Madden"].notna(),
    "NAME+POS+TEAM",
    "UNMATCHED"
)


# ==================================================
# FALLBACK LEGACY MATCH — NAME + POSITION ONLY
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
    .reset_index(name="Match_Count")
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

ot_stats = ot_stats.merge(
    legacy_fallback,
    on=[
        "Name_Key",
        "Pos_Key"
    ],
    how="left"
)

needs_fallback = ot_stats["Madden"].isna()

for column in [
    "Age",
    "Madden",
    "PFF",
    "PFR"
]:
    fallback_column = "Fallback_" + column

    ot_stats.loc[
        needs_fallback,
        column
    ] = ot_stats.loc[
        needs_fallback,
        fallback_column
    ]

fallback_success = (
    needs_fallback
    &
    ot_stats["Fallback_Madden"].notna()
)

ot_stats.loc[
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
    ot_stats["birth_date"],
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

ot_stats["Age_Source"] = np.select(
    [
        ot_stats["Age"].notna(),
        calculated_age.notna()
    ],
    [
        "LEGACY",
        "NFLVERSE_BIRTH_DATE"
    ],
    default="MISSING"
)

ot_stats["Age"] = (
    ot_stats["Age"]
    .fillna(calculated_age)
)

ot_stats = ot_stats.drop(
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


ot_stats["Legacy_Rating"] = ot_stats.apply(
    calculate_legacy_rating,
    axis=1
)

ot_stats["Legacy_Fallback_Applied"] = (
    ot_stats["Legacy_Rating"].isna()
)

ot_stats["Legacy_Rating_Used"] = (
    ot_stats["Legacy_Rating"]
    .fillna(MISSING_LEGACY_RATING)
)


# ==================================================
# FINAL ADVANCED / LEGACY CONFIDENCE
# ==================================================

ot_stats["Opportunity_Confidence"] = np.minimum(
    ot_stats["Offense_Snaps"]
    /
    FULL_CONFIDENCE_SNAPS,
    1.0
)

ot_stats["Actual_Advanced_Weight"] = (
    BASE_ADVANCED_OVR_WEIGHT
    *
    ot_stats["Opportunity_Confidence"]
    *
    ot_stats["Advanced_Data_Coverage"]
)

ot_stats["Actual_Legacy_Weight"] = (
    1
    -
    ot_stats["Actual_Advanced_Weight"]
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


ot_stats["Pre_Bonus_Overall"] = ot_stats.apply(
    calculate_pre_bonus_overall,
    axis=1
)

# No OT-specific historical record bonus in V1.
ot_stats["Historical_Record_Bonus"] = 0.0

ot_stats["Final_Overall"] = (
    ot_stats["Pre_Bonus_Overall"]
    +
    ot_stats["Historical_Record_Bonus"]
)


# ==================================================
# AGE-ADJUSTED VALUE
# ==================================================

ot_stats["Age_Adjusted_Overall"] = (
    ot_stats["Final_Overall"]
    *
    (
        (
            100
            +
            (
                OT_PRIME_AGE
                -
                ot_stats["Age"]
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
    "Legacy_Rating_Used",
    "Legacy_Fallback_Applied",
    "Advanced_Rating",
    "Advanced_Weighted_Pctl",
    "Pre_Bonus_Overall",
    "Historical_Record_Bonus",
    "Legacy_Match_Status",

    "Offense_Snaps",
    "Snap_Pct_Eligible",
    "Snap_Stat_Confidence",
    "Opportunity_Confidence",
    "Advanced_Data_Coverage",
    "Actual_Advanced_Weight",
    "Actual_Legacy_Weight",

    "Run_Block_Efficiency",
    "Run_Block_Efficiency_Pctl",
    "Run_Block_Efficiency_Pctl_Raw",

    "Pass_Block_Efficiency",
    "Pass_Block_Efficiency_Pctl",
    "Pass_Block_Efficiency_Pctl_Raw",

    "Avg_Time_To_Pressure",
    "Avg_Time_To_Pressure_Pctl",
    "Avg_Time_To_Pressure_Pctl_Raw",

    "Pressure_Rate_Over_Expected",
    "Pressure_Rate_Over_Expected_Pctl",
    "Pressure_Rate_Over_Expected_Pctl_Raw",

    "Pancakes",
    "Pancakes_Per_100_Snaps",
    "Pancake_Rate_Pctl",
    "Pancake_Rate_Pctl_Raw",

    "Penalties",
    "Penalty_Rate",
    "Penalty_Rate_Pctl",
    "Penalty_Rate_Pctl_Raw",

    "Run_Block_Disrupt_Rate",
    "Run_Block_Disrupt_Rate_Pctl",
    "Run_Block_Disrupt_Rate_Pctl_Raw",

    "Madden",
    "PFF",
    "PFR",
    "pfr_id",
    "pff_id",
    "espn_id",
]

for column in output_columns:
    if column not in ot_stats.columns:
        ot_stats[column] = np.nan

ot_output = ot_stats[
    output_columns
].copy()

ot_output = ot_output.round({
    "Age": 1,
    "Final_Overall": 2,
    "Age_Adjusted_Overall": 2,
    "Legacy_Rating": 2,
    "Legacy_Rating_Used": 2,
    "Advanced_Rating": 2,
    "Advanced_Weighted_Pctl": 2,
    "Pre_Bonus_Overall": 2,
    "Historical_Record_Bonus": 2,
    "Snap_Stat_Confidence": 3,
    "Opportunity_Confidence": 3,
    "Advanced_Data_Coverage": 3,
    "Actual_Advanced_Weight": 3,
    "Actual_Legacy_Weight": 3,
    "Run_Block_Efficiency": 3,
    "Run_Block_Efficiency_Pctl": 1,
    "Run_Block_Efficiency_Pctl_Raw": 1,
    "Pass_Block_Efficiency": 3,
    "Pass_Block_Efficiency_Pctl": 1,
    "Pass_Block_Efficiency_Pctl_Raw": 1,
    "Avg_Time_To_Pressure": 3,
    "Avg_Time_To_Pressure_Pctl": 1,
    "Avg_Time_To_Pressure_Pctl_Raw": 1,
    "Pressure_Rate_Over_Expected": 3,
    "Pressure_Rate_Over_Expected_Pctl": 1,
    "Pressure_Rate_Over_Expected_Pctl_Raw": 1,
    "Pancakes": 1,
    "Pancakes_Per_100_Snaps": 3,
    "Pancake_Rate_Pctl": 1,
    "Pancake_Rate_Pctl_Raw": 1,
    "Penalties": 1,
    "Penalty_Rate": 3,
    "Penalty_Rate_Pctl": 1,
    "Penalty_Rate_Pctl_Raw": 1,
    "Run_Block_Disrupt_Rate": 3,
    "Run_Block_Disrupt_Rate_Pctl": 1,
    "Run_Block_Disrupt_Rate_Pctl_Raw": 1,
    "Madden": 1,
    "PFF": 1,
    "PFR": 1,
})

ot_output = ot_output.sort_values(
    "Final_Overall",
    ascending=False,
    na_position="last"
)

top_25 = ot_output.head(25).copy()

missing_age_players = ot_stats[
    ot_stats["Age"].isna()
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


# ==================================================
# SAVE PARQUET FOR FUTURE MASTER WORKBOOK
# ==================================================

ot_output.to_parquet(
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

        ot_output.to_excel(
            writer,
            sheet_name="All_OTs",
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
        f"with {len(ot_output)} OTs."
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
