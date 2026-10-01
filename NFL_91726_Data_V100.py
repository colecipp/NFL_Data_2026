from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd


# ==================================================
# SETTINGS
# ==================================================

SEASON = 2025

PASSING_QUALIFIER = 200
SCRAMBLE_QUALIFIER = 20
RUSH_QUALIFIER = 25

OUTPUT_FILE = f"QB_Advanced_{SEASON}.xlsx"

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

PBP_CACHE = CACHE_DIR / f"pbp_{SEASON}.parquet"

QBR_URL = (
    "https://github.com/nflverse/nflverse-data/"
    "releases/download/espn_data/qbr_season_level.csv"
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def add_percentile(
    df,
    value_column,
    percentile_column,
    eligible,
    higher_is_better=True
):
    """
    Add percentile only for players who meet the qualification threshold.
    Players below the threshold keep their raw stat but receive no percentile.
    """

    df[percentile_column] = np.nan

    valid = (
        eligible &
        df[value_column].notna()
    )

    if valid.sum() == 0:
        return

    df.loc[valid, percentile_column] = (
        df.loc[valid, value_column]
        .rank(
            pct=True,
            ascending=higher_is_better
        ) * 100
    )


def normalize_numeric_id(series):
    """
    Convert IDs such as 3918298.0 into '3918298'
    so different data sources merge correctly.
    """

    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
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


# Regular season only
pbp = pbp[
    pbp["season_type"] == "REG"
].copy()

pbp = pbp.sort_values(
    ["week", "game_id", "play_id"]
)


# ==================================================
# IDENTIFY QB ON EACH DROPBACK
# ==================================================

# Normally passer_player_id identifies the QB.
pbp["qb_id"] = pbp["passer_player_id"]
pbp["qb_name"] = pbp["passer_player_name"]

# On scrambles, fall back to the rusher ID if needed.
scramble_missing_qb = (
    (pbp["qb_scramble"] == 1) &
    (pbp["qb_id"].isna())
)

pbp.loc[
    scramble_missing_qb,
    "qb_id"
] = pbp.loc[
    scramble_missing_qb,
    "rusher_player_id"
]

pbp.loc[
    scramble_missing_qb,
    "qb_name"
] = pbp.loc[
    scramble_missing_qb,
    "rusher_player_name"
]


# Prefer qb_epa for QB evaluation.
# Fall back to normal EPA if necessary.
pbp["qb_epa_value"] = (
    pbp["qb_epa"]
    .fillna(pbp["epa"])
)


# ==================================================
# DROPBACK METRICS
# ==================================================

dropbacks = pbp[
    (pbp["qb_dropback"] == 1) &
    (pbp["qb_id"].notna())
].copy()


qb_stats = (
    dropbacks
    .groupby("qb_id")
    .agg(
        Short_Name=("qb_name", "last"),
        Team_2025=("posteam", "last"),

        Dropbacks=("qb_id", "size"),

        EPA_per_Dropback=(
            "qb_epa_value",
            "mean"
        ),

        CPOE=(
            "cpoe",
            "mean"
        ),

        Dropback_Success_Rate=(
            "success",
            "mean"
        ),

        Sacks=(
            "sack",
            "sum"
        )
    )
    .reset_index()
    .rename(
        columns={
            "qb_id": "Player_ID"
        }
    )
)


qb_stats["Sack_Rate"] = (
    qb_stats["Sacks"] /
    qb_stats["Dropbacks"]
)


# ==================================================
# SCRAMBLE METRICS
# ==================================================

scrambles = pbp[
    (pbp["qb_scramble"] == 1) &
    (pbp["rusher_player_id"].notna()) &
    (pbp["epa"].notna())
].copy()


scramble_stats = (
    scrambles
    .groupby("rusher_player_id")
    .agg(
        Scrambles=("epa", "count"),

        Scramble_EPA_per_Attempt=(
            "epa",
            "mean"
        ),

        Scramble_Success_Rate=(
            "success",
            "mean"
        )
    )
    .reset_index()
    .rename(
        columns={
            "rusher_player_id": "Player_ID"
        }
    )
)


qb_stats = qb_stats.merge(
    scramble_stats,
    on="Player_ID",
    how="left"
)


# ==================================================
# DESIGNED QB RUSH METRICS
# ==================================================

# We separate designed runs from scrambles.
# This prevents the scrambling metric and rushing metric
# from counting the exact same plays twice.

qb_ids = set(
    qb_stats["Player_ID"]
)


designed_runs = pbp[
    (pbp["rush_attempt"] == 1) &
    (pbp["rusher_player_id"].isin(qb_ids)) &
    (pbp["qb_scramble"].fillna(0) != 1) &
    (pbp["qb_kneel"].fillna(0) != 1) &
    (pbp["epa"].notna())
].copy()


rush_stats = (
    designed_runs
    .groupby("rusher_player_id")
    .agg(
        Designed_Rush_Attempts=(
            "epa",
            "count"
        ),

        Designed_Rush_EPA_per_Attempt=(
            "epa",
            "mean"
        )
    )
    .reset_index()
    .rename(
        columns={
            "rusher_player_id": "Player_ID"
        }
    )
)


qb_stats = qb_stats.merge(
    rush_stats,
    on="Player_ID",
    how="left"
)


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
    "espn_id"
]

players = (
    players[player_columns]
    .drop_duplicates("gsis_id")
)


qb_stats = qb_stats.merge(
    players,
    left_on="Player_ID",
    right_on="gsis_id",
    how="left"
)


qb_stats["Player"] = (
    qb_stats["display_name"]
    .fillna(qb_stats["Short_Name"])
)


# Remove non-QBs who threw a trick-play pass
qb_stats = qb_stats[
    (qb_stats["position"] == "QB") |
    (qb_stats["position"].isna())
].copy()


qb_stats["Position"] = "QB"


# ==================================================
# PRO FOOTBALL REFERENCE
# ON-TARGET %
# ==================================================

try:

    pfr = nfl.load_pfr_advstats(
        seasons=True,
        stat_type="pass",
        summary_level="season"
    ).to_pandas()

    pfr = pfr[
        pfr["season"] == SEASON
    ].copy()

    # nflverse has used both names in different tables.
    if "pfr_id" in pfr.columns:
        pfr_id_column = "pfr_id"
    else:
        pfr_id_column = "pfr_player_id"

    # If multiple rows exist, keep the row with
    # the largest season pass-attempt sample.
    if "pass_attempts" in pfr.columns:

        pfr = pfr.sort_values(
            "pass_attempts",
            ascending=False
        )

    pfr = (
        pfr
        .drop_duplicates(pfr_id_column)
        [[
            pfr_id_column,
            "on_tgt_pct"
        ]]
        .rename(
            columns={
                pfr_id_column: "PFR_ID",
                "on_tgt_pct": "On_Target_Pct"
            }
        )
    )

    qb_stats = qb_stats.merge(
        pfr,
        left_on="pfr_id",
        right_on="PFR_ID",
        how="left"
    )

except Exception as error:

    print(
        f"PFR on-target data failed: {error}"
    )

    qb_stats["On_Target_Pct"] = np.nan


# ==================================================
# ESPN TOTAL QBR
# ==================================================

try:

    qbr = pd.read_csv(QBR_URL)

    qbr = qbr[
        (qbr["season"] == SEASON) &
        (qbr["season_type"] == "Regular")
    ].copy()

    qbr["ESPN_ID_Key"] = (
        normalize_numeric_id(
            qbr["player_id"]
        )
    )

    # Normally one season row per player.
    # This protects us against duplicate rows.
    if "qb_plays" in qbr.columns:

        qbr = qbr.sort_values(
            "qb_plays",
            ascending=False
        )

    qbr = (
        qbr
        .drop_duplicates("ESPN_ID_Key")
        [[
            "ESPN_ID_Key",
            "qbr_total"
        ]]
        .rename(
            columns={
                "qbr_total": "QBR"
            }
        )
    )

    qb_stats["ESPN_ID_Key"] = (
        normalize_numeric_id(
            qb_stats["espn_id"]
        )
    )

    qb_stats = qb_stats.merge(
        qbr,
        on="ESPN_ID_Key",
        how="left"
    )

except Exception as error:

    print(
        f"QBR data failed: {error}"
    )

    qb_stats["QBR"] = np.nan


# ==================================================
# PFF TURNOVER-WORTHY PLAY RATE
# ==================================================

# PFF-specific metric.
# Do NOT approximate this with interceptions.

qb_stats["TWP_Rate"] = np.nan


# ==================================================
# DISPLAY-FRIENDLY RATE CONVERSIONS
# ==================================================

# nflfastR CPOE is stored as a proportion.
# Convert to percentage points.

# qb_stats["CPOE"] = (
#     qb_stats["CPOE"] * 100
# )

qb_stats["Dropback_Success_Rate"] = (
    qb_stats["Dropback_Success_Rate"] * 100
)

qb_stats["Sack_Rate"] = (
    qb_stats["Sack_Rate"] * 100
)

qb_stats["Scramble_Success_Rate"] = (
    qb_stats["Scramble_Success_Rate"] * 100
)


# ==================================================
# QUALIFICATION GROUPS
# ==================================================

passing_eligible = (
    qb_stats["Dropbacks"] >=
    PASSING_QUALIFIER
)

scramble_eligible = (
    qb_stats["Scrambles"] >=
    SCRAMBLE_QUALIFIER
)

rush_eligible = (
    qb_stats["Designed_Rush_Attempts"] >=
    RUSH_QUALIFIER
)


qb_stats["Passing_Pct_Eligible"] = (
    passing_eligible
)

qb_stats["Scramble_Pct_Eligible"] = (
    scramble_eligible
)

qb_stats["Rush_Pct_Eligible"] = (
    rush_eligible
)


# ==================================================
# 2025 POSITIONAL PERCENTILES
# ==================================================

add_percentile(
    qb_stats,
    "QBR",
    "QBR_Pctl",
    passing_eligible,
    True
)

add_percentile(
    qb_stats,
    "EPA_per_Dropback",
    "EPA_DB_Pctl",
    passing_eligible,
    True
)

add_percentile(
    qb_stats,
    "CPOE",
    "CPOE_Pctl",
    passing_eligible,
    True
)

add_percentile(
    qb_stats,
    "Dropback_Success_Rate",
    "Success_Rate_Pctl",
    passing_eligible,
    True
)

add_percentile(
    qb_stats,
    "Sack_Rate",
    "Sack_Rate_Pctl",
    passing_eligible,
    False
)

add_percentile(
    qb_stats,
    "On_Target_Pct",
    "On_Target_Pctl",
    passing_eligible,
    True
)

add_percentile(
    qb_stats,
    "Scramble_EPA_per_Attempt",
    "Scramble_EPA_Pctl",
    scramble_eligible,
    True
)

add_percentile(
    qb_stats,
    "Scramble_Success_Rate",
    "Scramble_Success_Pctl",
    scramble_eligible,
    True
)

add_percentile(
    qb_stats,
    "Designed_Rush_EPA_per_Attempt",
    "Rush_EPA_Pctl",
    rush_eligible,
    True
)


# TWP percentile stays blank until PFF data exists
qb_stats["TWP_Pctl"] = np.nan

# ==================================================
# QB ADVANCED WEIGHTS
# ==================================================

# EDIT THESE AS YOU TEST THE MODEL.
# These should eventually total 1.00.

QB_WEIGHTS = {
    "QBR_Pctl": 0.16,
    "EPA_DB_Pctl": 0.18,
    "CPOE_Pctl": 0.12,
    "Success_Rate_Pctl": 0.10,
    "TWP_Pctl": 0.16,
    "Sack_Rate_Pctl": 0.10,
    "On_Target_Pctl": 0.08,
    "Scramble_Pctl": 0.05,
    "Rush_EPA_Pctl": 0.05
}


# ==================================================
# COMBINE SCRAMBLE METRICS
# ==================================================

qb_stats["Scramble_Pctl"] = (
    qb_stats[
        [
            "Scramble_EPA_Pctl",
            "Scramble_Success_Pctl"
        ]
    ]
    .mean(axis=1)
)


# ==================================================
# CALCULATE WEIGHTED ADVANCED SCORE
# ==================================================

def calculate_weighted_score(row, weights):
    """
    Calculates a player's weighted advanced percentile.

    If a metric is unavailable or the player did not qualify
    for that metric, its weight is removed from the denominator.

    Example:
    If rushing is worth 5% but a QB doesn't qualify,
    the remaining available metrics are automatically
    reweighted to total 100%.
    """

    weighted_total = 0
    available_weight = 0

    for metric, weight in weights.items():

        value = row.get(metric)

        if pd.notna(value) and weight > 0:
            weighted_total += value * weight
            available_weight += weight

    if available_weight == 0:
        return np.nan

    return weighted_total / available_weight


qb_stats["Advanced_Weighted_Pctl"] = qb_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=QB_WEIGHTS
)


# ==================================================
# OPTIONAL ADVANCED RATING CONVERSION
# ==================================================

# Leave this simple for now.
# Replace the formula later once we settle on
# the proper percentile-to-rating normalization.

qb_stats["Advanced_Rating"] = (
    60 +
    qb_stats["Advanced_Weighted_Pctl"] * 0.40
)


# ==================================================
# LOAD LEGACY DATABASE RATINGS
# ==================================================

LEGACY_DB_FILE = "2026_NFLActive.xlsx"

legacy = pd.read_excel(
    LEGACY_DB_FILE,
    sheet_name="Main"
)

# We only need these fields from the old database.
# Pandas will call the SECOND duplicate Madden column "Madden.1",
# so "Madden" correctly refers to Column G.
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


# ==================================================
# NORMALIZE PLAYER NAMES FOR MATCHING
# ==================================================

import re
import unicodedata


def normalize_player_name(name):

    if pd.isna(name):
        return ""

    name = str(name)

    # Remove accents
    name = unicodedata.normalize(
        "NFKD",
        name
    ).encode(
        "ascii",
        "ignore"
    ).decode()

    # Lowercase
    name = name.lower()

    # Remove punctuation
    name = re.sub(
        r"[^a-z0-9 ]",
        "",
        name
    )

    # Remove common suffixes
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


legacy["Name_Key"] = (
    legacy["Player"]
    .apply(normalize_player_name)
)

legacy["Pos_Key"] = (
    legacy["Pos"]
    .astype(str)
    .str.upper()
    .str.strip()
)

legacy["Team_Key"] = (
    legacy["Team"]
    .astype(str)
    .str.upper()
    .str.strip()
)


qb_stats["Name_Key"] = (
    qb_stats["Player"]
    .apply(normalize_player_name)
)

qb_stats["Pos_Key"] = "QB"

qb_stats["Team_Key"] = (
    qb_stats["Team_2025"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ==================================================
# PRIMARY MATCH
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


qb_stats = qb_stats.merge(
    legacy_primary,
    on=[
        "Name_Key",
        "Pos_Key",
        "Team_Key"
    ],
    how="left"
)


qb_stats["Legacy_Match_Status"] = np.where(
    qb_stats["Madden"].notna(),
    "NAME+POS+TEAM",
    "UNMATCHED"
)


# ==================================================
# FALLBACK MATCH
# NAME + POSITION ONLY
# ==================================================

# Only allow fallback if that player/position combination
# appears exactly ONCE in the old database.

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


qb_stats = qb_stats.merge(
    legacy_fallback,
    on=[
        "Name_Key",
        "Pos_Key"
    ],
    how="left"
)


needs_fallback = (
    qb_stats["Madden"].isna()
)


qb_stats.loc[
    needs_fallback,
    "Age"
] = qb_stats.loc[
    needs_fallback,
    "Fallback_Age"
]


qb_stats.loc[
    needs_fallback,
    "Madden"
] = qb_stats.loc[
    needs_fallback,
    "Fallback_Madden"
]


qb_stats.loc[
    needs_fallback,
    "PFF"
] = qb_stats.loc[
    needs_fallback,
    "Fallback_PFF"
]


qb_stats.loc[
    needs_fallback,
    "PFR"
] = qb_stats.loc[
    needs_fallback,
    "Fallback_PFR"
]


fallback_success = (
    needs_fallback &
    qb_stats["Fallback_Madden"].notna()
)


qb_stats.loc[
    fallback_success,
    "Legacy_Match_Status"
] = "NAME+POS"


# ==================================================
# CLEAN TEMPORARY MATCHING COLUMNS
# ==================================================

qb_stats = qb_stats.drop(
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

qb_stats["Legacy_Rating"] = (
    qb_stats["Madden"] * 0.5722 +
    qb_stats["PFF"] * 1.1 * 0.3611 +
    qb_stats["PFR"] * 5.2 * 0.0666
)

# ==================================================
# FINAL OVERALL WEIGHTS
# ==================================================

# EDIT THESE while testing.
# They should total 1.00.

ADVANCED_OVR_WEIGHT = 0.60
LEGACY_OVR_WEIGHT = 0.40


qb_stats["Final_Overall"] = (
    qb_stats["Advanced_Rating"] * ADVANCED_OVR_WEIGHT +
    qb_stats["Legacy_Rating"] * LEGACY_OVR_WEIGHT
)

qb_stats["Age_Adjusted_Overall"] = (
    qb_stats["Final_Overall"] *
    (
        (100 + (29 - qb_stats["Age"])) /
        100
    )
)

# ==================================================
# FINAL COLUMN ORDER
# ==================================================


qb_output = qb_stats[
    [
        "Player_ID",
        "Player",
        "Team_2025",
        "Position",
        "Age",
        "Dropbacks",
        "Legacy_Rating",
        "Final_Overall",
        "Age_Adjusted_Overall",
        "Advanced_Weighted_Pctl",
        "Advanced_Rating",

        "Legacy_Match_Status",
        "pfr_id",
        "pff_id",
        "espn_id",
        
        "Passing_Pct_Eligible",
        "QBR",
        "QBR_Pctl",

        "EPA_per_Dropback",
        "EPA_DB_Pctl",

        "CPOE",
        "CPOE_Pctl",

        "Dropback_Success_Rate",
        "Success_Rate_Pctl",

        "TWP_Rate",
        "TWP_Pctl",

        "Sacks",
        "Sack_Rate",
        "Sack_Rate_Pctl",

        "On_Target_Pct",
        "On_Target_Pctl",

        "Scrambles",
        "Scramble_Pct_Eligible",
        "Scramble_Pctl",

        "Scramble_EPA_per_Attempt",
        "Scramble_EPA_Pctl",

        "Scramble_Success_Rate",
        "Scramble_Success_Pctl",

        "Designed_Rush_Attempts",
        "Rush_Pct_Eligible",

        "Designed_Rush_EPA_per_Attempt",
        "Rush_EPA_Pctl",

        "Madden",
        "PFF",
        "PFR"

    ]
].copy()


# ==================================================
# ROUND FOR DISPLAY
# ==================================================

qb_output = qb_output.round(
    {
        "QBR": 1,
        "QBR_Pctl": 1,

        "EPA_per_Dropback": 3,
        "EPA_DB_Pctl": 1,

        "CPOE": 2,
        "CPOE_Pctl": 1,

        "Dropback_Success_Rate": 2,
        "Success_Rate_Pctl": 1,

        "TWP_Rate": 2,
        "TWP_Pctl": 1,

        "Sack_Rate": 2,
        "Sack_Rate_Pctl": 1,

        "On_Target_Pct": 2,
        "On_Target_Pctl": 1,

        "Scramble_EPA_per_Attempt": 3,
        "Scramble_EPA_Pctl": 1,

        "Scramble_Success_Rate": 2,
        "Scramble_Success_Pctl": 1,
        "Scramble_Pctl": 1,        

        "Designed_Rush_EPA_per_Attempt": 3,
        "Rush_EPA_Pctl": 1,


        "Advanced_Weighted_Pctl": 2,
        "Advanced_Rating": 2,

        "Madden": 1,
        "PFF": 1,
        "PFR": 1,
        "Legacy_Rating": 2,

        "Final_Overall": 2,
        "Age_Adjusted_Overall": 2
    }
)


qb_output = qb_output.sort_values(
    "Dropbacks",
    ascending=False
)


# ==================================================
# EXCEL OUTPUT
# ==================================================

josh_allen = qb_output[
    qb_output["Player"] == "Josh Allen"
].copy()


try:

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        qb_output.to_excel(
            writer,
            sheet_name="All_QBs",
            index=False
        )

        josh_allen.to_excel(
            writer,
            sheet_name="Josh_Allen_Check",
            index=False
        )

        # Basic visual formatting
        for worksheet in writer.book.worksheets:

            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:

                max_length = max(
                    len(str(cell.value))
                    if cell.value is not None
                    else 0
                    for cell in column_cells
                )

                column_letter = (
                    column_cells[0]
                    .column_letter
                )

                worksheet.column_dimensions[
                    column_letter
                ].width = min(
                    max_length + 2,
                    28
                )

    print(
        f"Created {OUTPUT_FILE} "
        f"with {len(qb_output)} QBs."
    )

except PermissionError:

    print(
        f"ERROR: Close {OUTPUT_FILE} "
        "in Excel and run the script again."
    )