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

# ==================================================
# CACHE SETTINGS
# ==================================================

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

PBP_CACHE = CACHE_DIR / f"pbp_{SEASON}.parquet"

# Five-year historical comparison pool
HISTORICAL_SEASONS = list(
    range(SEASON - 4, SEASON + 1)
)

HIST_PBP_CACHE = (
    CACHE_DIR /
    f"pbp_{HISTORICAL_SEASONS[0]}_{HISTORICAL_SEASONS[-1]}.parquet"
)

# ==================================================
# MODEL SETTINGS — V2
# ==================================================

FULL_CONFIDENCE_DROPBACKS = 600

BASE_ADVANCED_OVR_WEIGHT = 0.60

NEUTRAL_PERCENTILE = 50.0

MAX_RECORD_BONUS = 1.50

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
# FIVE-YEAR HISTORICAL PERCENTILE REFERENCE
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


# --------------------------------------------------
# IDENTIFY QB
# --------------------------------------------------

hist_pbp["qb_id"] = (
    hist_pbp["passer_player_id"]
)

hist_pbp["qb_name"] = (
    hist_pbp["passer_player_name"]
)


hist_scramble_missing_qb = (
    (hist_pbp["qb_scramble"] == 1) &
    (hist_pbp["qb_id"].isna())
)


hist_pbp.loc[
    hist_scramble_missing_qb,
    "qb_id"
] = hist_pbp.loc[
    hist_scramble_missing_qb,
    "rusher_player_id"
]


hist_pbp.loc[
    hist_scramble_missing_qb,
    "qb_name"
] = hist_pbp.loc[
    hist_scramble_missing_qb,
    "rusher_player_name"
]


hist_pbp["qb_epa_value"] = (
    hist_pbp["qb_epa"]
    .fillna(hist_pbp["epa"])
)


# ==================================================
# HISTORICAL DROPBACK METRICS
# ==================================================

hist_dropbacks = hist_pbp[
    (hist_pbp["qb_dropback"] == 1) &
    (hist_pbp["qb_id"].notna())
].copy()


hist_qb = (
    hist_dropbacks
    .groupby(
        [
            "season",
            "qb_id"
        ]
    )
    .agg(
        Dropbacks=(
            "qb_id",
            "size"
        ),

        Total_QB_EPA=(
            "qb_epa_value",
            "sum"
        ),

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


hist_qb["Sack_Rate"] = (
    hist_qb["Sacks"] /
    hist_qb["Dropbacks"] *
    100
)


hist_qb["Dropback_Success_Rate"] = (
    hist_qb["Dropback_Success_Rate"] *
    100
)


# ==================================================
# HISTORICAL SCRAMBLE METRICS
# ==================================================

hist_scrambles = hist_pbp[
    (hist_pbp["qb_scramble"] == 1) &
    (hist_pbp["rusher_player_id"].notna()) &
    (hist_pbp["epa"].notna())
].copy()


hist_scramble_stats = (
    hist_scrambles
    .groupby(
        [
            "season",
            "rusher_player_id"
        ]
    )
    .agg(
        Scrambles=(
            "epa",
            "count"
        ),

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
            "rusher_player_id":
            "Player_ID"
        }
    )
)


hist_scramble_stats[
    "Scramble_Success_Rate"
] *= 100


hist_qb = hist_qb.merge(
    hist_scramble_stats,
    on=[
        "season",
        "Player_ID"
    ],
    how="left"
)


# ==================================================
# HISTORICAL DESIGNED QB RUNS
# ==================================================

hist_qb_ids = set(
    hist_qb["Player_ID"]
)


hist_designed_runs = hist_pbp[
    (hist_pbp["rush_attempt"] == 1) &
    (
        hist_pbp["rusher_player_id"]
        .isin(hist_qb_ids)
    ) &
    (
        hist_pbp["qb_scramble"]
        .fillna(0) != 1
    ) &
    (
        hist_pbp["qb_kneel"]
        .fillna(0) != 1
    ) &
    (hist_pbp["epa"].notna())
].copy()


hist_rush_stats = (
    hist_designed_runs
    .groupby(
        [
            "season",
            "rusher_player_id"
        ]
    )
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
            "rusher_player_id":
            "Player_ID"
        }
    )
)


hist_qb = hist_qb.merge(
    hist_rush_stats,
    on=[
        "season",
        "Player_ID"
    ],
    how="left"
)


# ==================================================
# ADD PFR + ESPN IDS
# ==================================================

hist_qb = hist_qb.merge(
    players[
        [
            "gsis_id",
            "pfr_id",
            "espn_id"
        ]
    ],
    left_on="Player_ID",
    right_on="gsis_id",
    how="left"
)


# ==================================================
# HISTORICAL PFR ON-TARGET %
# ==================================================

hist_pfr = nfl.load_pfr_advstats(
    seasons=HISTORICAL_SEASONS,
    stat_type="pass",
    summary_level="season"
).to_pandas()


hist_pfr_id_column = (
    "pfr_id"
    if "pfr_id" in hist_pfr.columns
    else "pfr_player_id"
)


hist_pfr = (
    hist_pfr[
        [
            "season",
            hist_pfr_id_column,
            "on_tgt_pct"
        ]
    ]
    .rename(
        columns={
            hist_pfr_id_column:
            "PFR_ID",

            "on_tgt_pct":
            "On_Target_Pct"
        }
    )
)


hist_qb = hist_qb.merge(
    hist_pfr,
    left_on=[
        "season",
        "pfr_id"
    ],
    right_on=[
        "season",
        "PFR_ID"
    ],
    how="left"
)


# ==================================================
# HISTORICAL QBR
# ==================================================

qbr_all = pd.read_csv(
    QBR_URL
)


qbr_hist = qbr_all[
    (
        qbr_all["season"]
        .isin(HISTORICAL_SEASONS)
    ) &
    (
        qbr_all["season_type"]
        == "Regular"
    )
].copy()


qbr_hist["ESPN_ID_Key"] = (
    normalize_numeric_id(
        qbr_hist["player_id"]
    )
)


hist_qb["ESPN_ID_Key"] = (
    normalize_numeric_id(
        hist_qb["espn_id"]
    )
)


qbr_hist = qbr_hist[
    [
        "season",
        "ESPN_ID_Key",
        "qbr_total"
    ]
].rename(
    columns={
        "qbr_total": "QBR"
    }
)


hist_qb = hist_qb.merge(
    qbr_hist,
    on=[
        "season",
        "ESPN_ID_Key"
    ],
    how="left"
)


# ==================================================
# HISTORICAL QUALIFICATION
# ==================================================

hist_qb["Passing_Eligible"] = (
    hist_qb["Dropbacks"] >=
    PASSING_QUALIFIER
)


hist_qb["Scramble_Eligible"] = (
    hist_qb["Scrambles"]
    .fillna(0) >=
    SCRAMBLE_QUALIFIER
)


hist_qb["Rush_Eligible"] = (
    hist_qb[
        "Designed_Rush_Attempts"
    ]
    .fillna(0) >=
    RUSH_QUALIFIER
)


# ==================================================
# HISTORICAL PERCENTILE FUNCTION
# ==================================================

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

        below = (
            pool < value
        ).sum()

        equal = (
            pool == value
        ).sum()

    else:

        below = (
            pool > value
        ).sum()

        equal = (
            pool == value
        ).sum()

    return (
        (
            below +
            0.5 * equal
        ) /
        len(pool) *
        100
    )


def add_historical_percentile(
    current_df,
    historical_df,
    value_column,
    percentile_column,
    current_eligible,
    historical_eligible,
    higher_is_better=True
):

    current_df[
        percentile_column
    ] = np.nan

    pool = historical_df.loc[
        historical_eligible,
        value_column
    ]

    valid = (
        current_eligible &
        current_df[
            value_column
        ].notna()
    )

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


# ==================================================
# NEW 5-YEAR PERCENTILES
# ==================================================

add_historical_percentile(
    qb_stats,
    hist_qb,
    "QBR",
    "QBR_Pctl",
    passing_eligible,
    hist_qb["Passing_Eligible"],
    True
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "EPA_per_Dropback",
    "EPA_DB_Pctl",
    passing_eligible,
    hist_qb["Passing_Eligible"],
    True
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "CPOE",
    "CPOE_Pctl",
    passing_eligible,
    hist_qb["Passing_Eligible"],
    True
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "Dropback_Success_Rate",
    "Success_Rate_Pctl",
    passing_eligible,
    hist_qb["Passing_Eligible"],
    True
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "Sack_Rate",
    "Sack_Rate_Pctl",
    passing_eligible,
    hist_qb["Passing_Eligible"],
    False
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "On_Target_Pct",
    "On_Target_Pctl",
    passing_eligible,
    hist_qb["Passing_Eligible"],
    True
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "Scramble_EPA_per_Attempt",
    "Scramble_EPA_Pctl",
    scramble_eligible,
    hist_qb[
        "Scramble_Eligible"
    ],
    True
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "Scramble_Success_Rate",
    "Scramble_Success_Pctl",
    scramble_eligible,
    hist_qb[
        "Scramble_Eligible"
    ],
    True
)

add_historical_percentile(
    qb_stats,
    hist_qb,
    "Designed_Rush_EPA_per_Attempt",
    "Rush_EPA_Pctl",
    rush_eligible,
    hist_qb[
        "Rush_Eligible"
    ],
    True
)


# Still blank until PFF TWP data is added
qb_stats["TWP_Pctl"] = np.nan

# ==================================================
# QB ADVANCED WEIGHTS
# ==================================================

# ==================================================
# PASSING METRIC WEIGHTS
# ==================================================

# These weights apply WITHIN the passing component.
# Missing metrics are reweighted automatically.

QB_PASSING_WEIGHTS = {
    "QBR_Pctl": 0.16,
    "EPA_DB_Pctl": 0.18,
    "CPOE_Pctl": 0.12,
    "Success_Rate_Pctl": 0.10,
    "TWP_Pctl": 0.16,
    "Sack_Rate_Pctl": 0.10,
    "On_Target_Pctl": 0.08
}


def calculate_weighted_score(
    row,
    weights
):

    weighted_total = 0
    available_weight = 0

    for metric, weight in (
        weights.items()
    ):

        value = row.get(metric)

        if (
            pd.notna(value) and
            weight > 0
        ):

            weighted_total += (
                value * weight
            )

            available_weight += weight

    if available_weight == 0:
        return np.nan

    return (
        weighted_total /
        available_weight
    )


# ==================================================
# PASSING CORE
# ==================================================

qb_stats[
    "Passing_Core_Pctl"
] = qb_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=QB_PASSING_WEIGHTS
)


# ==================================================
# SCRAMBLE COMPONENT
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


# If the QB doesn't have enough scrambles
# to trust the sample, use neutral rather
# than deleting the category and boosting
# all of his other statistics.

qb_stats[
    "Scramble_Component_Pctl"
] = np.where(
    scramble_eligible,
    qb_stats["Scramble_Pctl"],
    NEUTRAL_PERCENTILE
)


# ==================================================
# DESIGNED-RUN COMPONENT
# ==================================================

qb_stats[
    "Rush_Component_Pctl"
] = np.where(
    rush_eligible,
    qb_stats["Rush_EPA_Pctl"],
    NEUTRAL_PERCENTILE
)


# ==================================================
# OPPORTUNITY WEIGHTING
# ==================================================

# Scrambles are already included in Dropbacks.
# Designed runs are not.
#
# Therefore:
#
# Total QB plays =
# Dropbacks + Designed Runs

qb_stats[
    "Total_QB_Plays"
] = (
    qb_stats["Dropbacks"] +
    qb_stats[
        "Designed_Rush_Attempts"
    ].fillna(0)
)


qb_stats[
    "Scramble_Play_Share"
] = (
    qb_stats["Scrambles"]
    .fillna(0) /
    qb_stats["Total_QB_Plays"]
)


qb_stats[
    "Designed_Rush_Play_Share"
] = (
    qb_stats[
        "Designed_Rush_Attempts"
    ].fillna(0) /
    qb_stats["Total_QB_Plays"]
)


qb_stats[
    "Passing_Play_Share"
] = (
    1 -
    qb_stats[
        "Scramble_Play_Share"
    ] -
    qb_stats[
        "Designed_Rush_Play_Share"
    ]
)


# ==================================================
# OPPORTUNITY-WEIGHTED ADVANCED SCORE
# ==================================================

# ==================================================
# MOBILITY BONUS
# ==================================================

# Passing establishes the QB's baseline.
# Mobility adds/subtracts value depending on:
# 1. How good the QB is at it
# 2. How frequently he uses it
#
# The multiplier controls how valuable mobility is
# relative to passing.

MOBILITY_BOOST_MULTIPLIER = 1.50


qb_stats["Scramble_Adjustment"] = np.where(
    scramble_eligible,
    (
        qb_stats["Scramble_Pctl"] -
        NEUTRAL_PERCENTILE
    )
    * qb_stats["Scramble_Play_Share"]
    * MOBILITY_BOOST_MULTIPLIER,
    0
)


qb_stats["Rush_Adjustment"] = np.where(
    rush_eligible,
    (
        qb_stats["Rush_EPA_Pctl"] -
        NEUTRAL_PERCENTILE
    )
    * qb_stats["Designed_Rush_Play_Share"]
    * MOBILITY_BOOST_MULTIPLIER,
    0
)


qb_stats["Mobility_Adjustment"] = (
    qb_stats["Scramble_Adjustment"] +
    qb_stats["Rush_Adjustment"]
)


qb_stats["Advanced_Weighted_Pctl"] = (
    qb_stats["Passing_Core_Pctl"] +
    qb_stats["Mobility_Adjustment"]
)


# Prevent impossible percentile-style values
qb_stats["Advanced_Weighted_Pctl"] = (
    qb_stats["Advanced_Weighted_Pctl"]
    .clip(lower=0, upper=100)
)


qb_stats["Advanced_Rating"] = (
    60 +
    qb_stats["Advanced_Weighted_Pctl"] * 0.40
)


# Temporary percentile -> rating conversion.
# We can replace this later.

qb_stats[
    "Advanced_Rating"
] = (
    60 +
    qb_stats[
        "Advanced_Weighted_Pctl"
    ] * 0.40
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

# ==================================================
# DROPBACK CONFIDENCE
# ==================================================

qb_stats[
    "Dropback_Confidence"
] = np.minimum(
    qb_stats["Dropbacks"] /
    FULL_CONFIDENCE_DROPBACKS,
    1.0
)


# At 600+ DB:
# Advanced = 60%
# Legacy   = 40%
#
# At 300 DB:
# Advanced = 30%
# Legacy   = 70%

qb_stats[
    "Actual_Advanced_Weight"
] = (
    BASE_ADVANCED_OVR_WEIGHT *
    qb_stats[
        "Dropback_Confidence"
    ]
)


qb_stats[
    "Actual_Legacy_Weight"
] = (
    1 -
    qb_stats[
        "Actual_Advanced_Weight"
    ]
)


qb_stats[
    "Pre_Bonus_Overall"
] = (
    qb_stats["Advanced_Rating"] *
    qb_stats[
        "Actual_Advanced_Weight"
    ]

    +

    qb_stats["Legacy_Rating"] *
    qb_stats[
        "Actual_Legacy_Weight"
    ]
)

# ==================================================
# CURRENT-SEASON BROAD PRODUCTION
# ==================================================

season_stats = nfl.load_player_stats(
    [SEASON],
    summary_level="reg"
).to_pandas()


qb_season_stats = (
    season_stats[
        season_stats[
            "position"
        ] == "QB"
    ]
    .groupby("player_id")
    .agg(
        Passing_Yards=(
            "passing_yards",
            "sum"
        ),

        Passing_TDs=(
            "passing_tds",
            "sum"
        ),

        Total_Passing_EPA=(
            "passing_epa",
            "sum"
        )
    )
    .reset_index()
    .rename(
        columns={
            "player_id":
            "Player_ID"
        }
    )
)


qb_stats = qb_stats.merge(
    qb_season_stats,
    on="Player_ID",
    how="left"
)

# ==================================================
# HISTORIC-SEASON BONUS
# ==================================================

# Small bonuses only.
#
# Current 2025 all-time thresholds:
#
# Passing yards:
# Top 5  = 5,235+
# Top 10 = 5,109+
# Top 20 = 4,933+
#
# Passing TD:
# Top 5  = 48+
# Top 10 = 44+
# Top 20 = 40+
#
# Update these thresholds in future seasons
# if the historical leaderboard changes.

def record_threshold_bonus(
    value,
    top_5,
    top_10,
    top_20
):

    if pd.isna(value):
        return 0.0

    if value >= top_5:
        return 0.50

    if value >= top_10:
        return 0.35

    if value >= top_20:
        return 0.20

    return 0.0


qb_stats[
    "Pass_Yards_Record_Bonus"
] = qb_stats[
    "Passing_Yards"
].apply(
    lambda x:
    record_threshold_bonus(
        x,
        top_5=5235,
        top_10=5109,
        top_20=4933
    )
)


qb_stats[
    "Pass_TD_Record_Bonus"
] = qb_stats[
    "Passing_TDs"
].apply(
    lambda x:
    record_threshold_bonus(
        x,
        top_5=48,
        top_10=44,
        top_20=40
    )
)

# ==================================================
# HISTORIC QBR RANK
# ==================================================

qbr_record_pool = qbr_all[
    (
        qbr_all["season_type"]
        == "Regular"
    ) &
    (
        qbr_all["qbr_total"]
        .notna()
    )
].copy()


# Use meaningful seasons only
if "qb_plays" in qbr_record_pool.columns:

    qbr_record_pool = (
        qbr_record_pool[
            qbr_record_pool[
                "qb_plays"
            ] >= PASSING_QUALIFIER
        ]
    )


def historical_rank(
    value,
    pool
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

    return int(
        1 +
        (pool > value).sum()
    )


qb_stats[
    "QBR_Historical_Rank"
] = qb_stats["QBR"].apply(
    lambda x:
    historical_rank(
        x,
        qbr_record_pool[
            "qbr_total"
        ]
    )
)


def rank_bonus(rank):

    if pd.isna(rank):
        return 0.0

    if rank <= 5:
        return 0.50

    if rank <= 10:
        return 0.35

    if rank <= 20:
        return 0.20

    return 0.0


qb_stats[
    "QBR_Record_Bonus"
] = qb_stats[
    "QBR_Historical_Rank"
].apply(
    rank_bonus
)


# ==================================================
# TOTAL RECORD BONUS
# ==================================================

qb_stats[
    "Historical_Record_Bonus"
] = (
    qb_stats[
        "Pass_Yards_Record_Bonus"
    ] +

    qb_stats[
        "Pass_TD_Record_Bonus"
    ] +

    qb_stats[
        "QBR_Record_Bonus"
    ]
).clip(
    upper=MAX_RECORD_BONUS
)


qb_stats[
    "Final_Overall"
] = (
    qb_stats[
        "Pre_Bonus_Overall"
    ] +
    qb_stats[
        "Historical_Record_Bonus"
    ]
)



qb_stats[
    "Age_Adjusted_Overall"
] = (
    qb_stats[
        "Final_Overall"
    ] *
    (
        (
            100 +
            (
                29 -
                qb_stats["Age"]
            )
        ) /
        100
    )
)


# ==================================================
# FINAL COLUMN ORDER
# ==================================================


qb_output = qb_stats[
    [
        "Player",
        "Player_ID",
        "Team_2025",
        "Position",
        "Age",
        "Dropbacks",
        "Legacy_Rating",
        "Final_Overall",
        "Age_Adjusted_Overall",
        "Pre_Bonus_Overall",
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

        "Passing_Core_Pctl",

        "Total_QB_Plays",
        "Passing_Play_Share",
        "Scramble_Play_Share",
        "Designed_Rush_Play_Share",

        "Scramble_Component_Pctl",
        "Rush_Component_Pctl",

        "Advanced_Weighted_Pctl",
        "Advanced_Rating",

        "Dropback_Confidence",
        "Actual_Advanced_Weight",
        "Actual_Legacy_Weight",

        "Madden",
        "PFF",
        "PFR",
        "Legacy_Rating",

        "Passing_Yards",
        "Passing_TDs",
        "Total_Passing_EPA",

        "Pass_Yards_Record_Bonus",
        "Pass_TD_Record_Bonus",
        "QBR_Historical_Rank",
        "QBR_Record_Bonus",
        "Historical_Record_Bonus",

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

        "Passing_Core_Pctl": 2,

        "Passing_Play_Share": 3,
        "Scramble_Play_Share": 3,
        "Designed_Rush_Play_Share": 3,

        "Scramble_Component_Pctl": 2,
        "Rush_Component_Pctl": 2,

        "Advanced_Weighted_Pctl": 2,
        "Advanced_Rating": 2,

        "Dropback_Confidence": 3,
        "Actual_Advanced_Weight": 3,
        "Actual_Legacy_Weight": 3,

        "Total_Passing_EPA": 2,

        "Pass_Yards_Record_Bonus": 2,
        "Pass_TD_Record_Bonus": 2,
        "QBR_Record_Bonus": 2,
        "Historical_Record_Bonus": 2,

        "Pre_Bonus_Overall": 2,
        "Final_Overall": 2,
        "Age_Adjusted_Overall": 2,

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