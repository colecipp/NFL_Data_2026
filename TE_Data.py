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

# Full-confidence thresholds for the individual TE metrics.
TARGET_QUALIFIER = 40
ROUTE_QUALIFIER = 150
BLOCK_QUALIFIER = 100

# Final advanced-vs-legacy confidence.
# This is intentionally higher than TARGET_QUALIFIER, mirroring the QB model:
# a stat can be trustworthy enough for its percentile before the entire season
# is trustworthy enough to receive the maximum Advanced weight.
FULL_CONFIDENCE_TARGETS = 80
FULL_CONFIDENCE_ROLE_SNAPS = 500

OUTPUT_FILE = f"TE_Advanced_{SEASON}.xlsx"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PARQUET_OUTPUT = OUTPUT_DIR / f"TE_{SEASON}.parquet"

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
TE_PRIME_AGE = 29.0
MISSING_LEGACY_RATING = 65.0

# Intended share of the complete TE advanced model.
# Receiving remains the baseline; blocking adds/subtracts value by role usage.
RECEIVING_MODEL_SHARE = 0.80
BLOCKING_MODEL_SHARE = 0.20
BLOCKING_BOOST_MULTIPLIER = 0.60


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
    Every current TE with a raw value receives a percentile.

    The qualifier applies ONLY to the historical comparison pool.
    Low-volume current players are later regressed toward 50.
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


def calculate_weighted_score(
    row,
    weights
):
    weighted_total = 0.0
    available_weight = 0.0

    for metric, weight in weights.items():

        value = row.get(metric)

        if (
            pd.notna(value)
            and weight > 0
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


def first_existing_column(
    df,
    candidates
):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def safe_numeric(
    df,
    column,
    default=np.nan
):
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


# ==================================================
# LOAD / CACHE PLAY-BY-PLAY
# ==================================================

if PBP_CACHE.exists():

    pbp = pd.read_parquet(
        PBP_CACHE
    )

else:

    pbp = nfl.load_pbp(
        [SEASON]
    ).to_pandas()

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
# BASE TE POPULATION
# ==================================================

season_stats = nfl.load_player_stats(
    [SEASON],
    summary_level="reg"
).to_pandas()


current_te = season_stats[
    season_stats["position"] == "TE"
].copy()


# Protect against an unexpected duplicate player row.
current_te = (
    current_te
    .sort_values(
        "targets",
        ascending=False
    )
    .drop_duplicates(
        "player_id"
    )
)


# Season-level player stats do not always expose a team column
# consistently across nflreadpy/nflverse versions. Use it when
# available; otherwise leave it blank and let the PBP target data
# below supply the player's 2025 team.
if "team" in current_te.columns:
    current_te_team = current_te["team"]
elif "recent_team" in current_te.columns:
    current_te_team = current_te["recent_team"]
else:
    current_te_team = pd.Series(
        np.nan,
        index=current_te.index,
        dtype="object"
    )


te_stats = pd.DataFrame({
    "Player_ID":
        current_te["player_id"],

    "Short_Name":
        current_te["player_name"],

    "Player":
        current_te[
            "player_display_name"
        ].fillna(
            current_te["player_name"]
        ),

    "Team_2025":
        current_te_team,

    "Position":
        "TE",

    "Season_Targets":
        pd.to_numeric(
            current_te["targets"],
            errors="coerce"
        ),

    "Receptions":
        pd.to_numeric(
            current_te["receptions"],
            errors="coerce"
        ),

    "Receiving_Yards":
        pd.to_numeric(
            current_te["receiving_yards"],
            errors="coerce"
        ),

    "Receiving_TDs":
        pd.to_numeric(
            current_te["receiving_tds"],
            errors="coerce"
        ),

    "Total_Receiving_EPA":
        pd.to_numeric(
            current_te["receiving_epa"],
            errors="coerce"
        )
})


# ==================================================
# CURRENT TARGET-LEVEL METRICS
# ==================================================

target_plays = pbp[
    pbp["receiver_player_id"].notna()
].copy()


target_stats = (
    target_plays
    .groupby(
        "receiver_player_id"
    )
    .agg(
        PBP_Targets=(
            "receiver_player_id",
            "size"
        ),

        Target_EPA_Total=(
            "epa",
            "sum"
        ),

        EPA_per_Target=(
            "epa",
            "mean"
        ),

        Target_Success_Rate=(
            "success",
            "mean"
        ),

        Catch_Rate_Over_Expected=(
            "cpoe",
            "mean"
        ),

        Team_From_PBP=(
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


te_stats = te_stats.merge(
    target_stats,
    on="Player_ID",
    how="left"
)


te_stats["Targets"] = (
    te_stats["PBP_Targets"]
    .fillna(
        te_stats["Season_Targets"]
    )
    .fillna(0)
)


# PBP is the better 2025 team indicator for receivers who were traded.
te_stats["Team_2025"] = (
    te_stats["Team_From_PBP"]
    .fillna(
        te_stats["Team_2025"]
    )
)


te_stats["Target_Success_Rate"] = (
    te_stats[
        "Target_Success_Rate"
    ] * 100
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
    "espn_id",
    "birth_date"
]

players = (
    players[player_columns]
    .drop_duplicates("gsis_id")
)


te_stats = te_stats.merge(
    players,
    left_on="Player_ID",
    right_on="gsis_id",
    how="left",
    suffixes=("", "_player_map")
)


te_stats["Player"] = (
    te_stats["display_name"]
    .fillna(
        te_stats["Player"]
    )
)


# ==================================================
# NEXT GEN STATS — RECEIVING
# CURRENT SEASON
#
# We use Avg Separation as the current nflverse proxy
# for the user's future "Separation Rate" category.
# ==================================================

try:

    ngs = nfl.load_nextgen_stats(
        seasons=[SEASON],
        stat_type="receiving"
    ).to_pandas()

    if "season_type" in ngs.columns:

        ngs = ngs[
            ngs["season_type"] == "REG"
        ].copy()

    if "player_position" in ngs.columns:

        ngs = ngs[
            ngs["player_position"] == "TE"
        ].copy()

    ngs["targets"] = pd.to_numeric(
        ngs["targets"],
        errors="coerce"
    ).fillna(0)

    ngs["receptions"] = pd.to_numeric(
        ngs["receptions"],
        errors="coerce"
    ).fillna(0)

    ngs["Sep_Weighted"] = (
        pd.to_numeric(
            ngs["avg_separation"],
            errors="coerce"
        )
        *
        ngs["targets"]
    )

    ngs["YACOE_Weighted"] = (
        pd.to_numeric(
            ngs[
                "avg_yac_above_expectation"
            ],
            errors="coerce"
        )
        *
        ngs["receptions"]
    )

    ngs_current = (
        ngs
        .groupby(
            "player_gsis_id"
        )
        .agg(
            NGS_Targets=(
                "targets",
                "sum"
            ),

            NGS_Receptions=(
                "receptions",
                "sum"
            ),

            Sep_Weighted=(
                "Sep_Weighted",
                "sum"
            ),

            YACOE_Weighted=(
                "YACOE_Weighted",
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

    ngs_current["Avg_Separation"] = (
        ngs_current["Sep_Weighted"]
        /
        ngs_current[
            "NGS_Targets"
        ].replace(0, np.nan)
    )

    ngs_current[
        "YAC_Over_Expected"
    ] = (
        ngs_current["YACOE_Weighted"]
        /
        ngs_current[
            "NGS_Receptions"
        ].replace(0, np.nan)
    )

    ngs_current = ngs_current[
        [
            "Player_ID",
            "Avg_Separation",
            "YAC_Over_Expected"
        ]
    ]

    te_stats = te_stats.merge(
        ngs_current,
        on="Player_ID",
        how="left"
    )

except Exception as error:

    print(
        f"NGS receiving data failed: {error}"
    )

    te_stats["Avg_Separation"] = np.nan
    te_stats[
        "YAC_Over_Expected"
    ] = np.nan


# ==================================================
# PFR ADVANCED RECEIVING
# CURRENT SEASON
#
# Drop rate is used directly when available.
# Broken tackles / reception is used as a temporary
# proxy for Missed Tackles Forced / Touch.
# ==================================================

try:

    pfr_rec = nfl.load_pfr_advstats(
        seasons=[SEASON],
        stat_type="rec",
        summary_level="season"
    ).to_pandas()

    pfr_id_column = (
        "pfr_id"
        if "pfr_id" in pfr_rec.columns
        else "pfr_player_id"
    )

    drop_pct_column = first_existing_column(
        pfr_rec,
        [
            "drop_pct",
            "receiving_drop_pct"
        ]
    )

    drops_column = first_existing_column(
        pfr_rec,
        [
            "drops",
            "receiving_drop"
        ]
    )

    broken_tackles_column = first_existing_column(
        pfr_rec,
        [
            "brk_tkl",
            "broken_tackles",
            "receiving_broken_tackles",
            "receiving_brk_tkl"
        ]
    )

    receptions_column = first_existing_column(
        pfr_rec,
        [
            "receptions",
            "rec"
        ]
    )

    targets_column = first_existing_column(
        pfr_rec,
        [
            "targets",
            "tgt"
        ]
    )

    pfr_current = pd.DataFrame({
        "pfr_id":
            pfr_rec[pfr_id_column],

        "PFR_Drop_Rate":
            safe_numeric(
                pfr_rec,
                drop_pct_column
            ),

        "PFR_Drops":
            safe_numeric(
                pfr_rec,
                drops_column
            ),

        "PFR_Broken_Tackles":
            safe_numeric(
                pfr_rec,
                broken_tackles_column
            ),

        "PFR_Receptions":
            safe_numeric(
                pfr_rec,
                receptions_column
            ),

        "PFR_Targets":
            safe_numeric(
                pfr_rec,
                targets_column
            )
    })

    if (
        drop_pct_column is None
        and drops_column is not None
        and targets_column is not None
    ):

        pfr_current[
            "PFR_Drop_Rate"
        ] = (
            pfr_current["PFR_Drops"]
            /
            pfr_current[
                "PFR_Targets"
            ].replace(0, np.nan)
            *
            100
        )

    pfr_current[
        "Missed_Tackles_Per_Touch"
    ] = (
        pfr_current[
            "PFR_Broken_Tackles"
        ]
        /
        pfr_current[
            "PFR_Receptions"
        ].replace(0, np.nan)
    )

    pfr_current = (
        pfr_current
        .drop_duplicates("pfr_id")
    )

    te_stats = te_stats.merge(
        pfr_current[
            [
                "pfr_id",
                "PFR_Drop_Rate",
                "PFR_Drops",
                "PFR_Broken_Tackles",
                "Missed_Tackles_Per_Touch"
            ]
        ],
        on="pfr_id",
        how="left"
    )

    te_stats["Drop_Rate"] = (
        te_stats["PFR_Drop_Rate"]
    )

except Exception as error:

    print(
        f"PFR receiving data failed: {error}"
    )

    te_stats["Drop_Rate"] = np.nan
    te_stats[
        "Missed_Tackles_Per_Touch"
    ] = np.nan


# ==================================================
# FUTURE WEBSITE / ROUTE-CHARTING METRICS
#
# Columns are intentionally present now so they can
# be plugged into the model later without rebuilding it.
# ==================================================

te_stats["Routes"] = np.nan
te_stats["Yards_Per_Route"] = np.nan
te_stats["Targets_Per_Route"] = np.nan
te_stats["Route_Win_Rate"] = np.nan
te_stats[
    "Contested_Catch_Win_Rate"
] = np.nan
te_stats["Press_Win_Rate"] = np.nan

# Blocking metrics/snaps are explicit placeholders for now.
# They live outside the receiving core so a blocking specialist can
# eventually add meaningful value based on both quality and usage.
te_stats["Run_Block_Snaps"] = np.nan
te_stats["Pass_Block_Snaps"] = np.nan
te_stats["Run_Block_Efficiency"] = np.nan
te_stats["Pass_Block_Efficiency"] = np.nan

te_stats["Total_Blocking_Snaps"] = (
    te_stats[[
        "Run_Block_Snaps",
        "Pass_Block_Snaps"
    ]].sum(axis=1, min_count=1)
)

# The requested long-term metric is "Separation Rate".
# For V1, Avg_Separation is the available NGS proxy.
te_stats["Separation_Rate"] = (
    te_stats["Avg_Separation"]
)


# ==================================================
# QUALIFICATION / CONFIDENCE FLAGS
# ==================================================

te_stats["Target_Pct_Eligible"] = (
    te_stats["Targets"] >=
    TARGET_QUALIFIER
)

te_stats["Route_Pct_Eligible"] = (
    te_stats["Routes"].fillna(0) >=
    ROUTE_QUALIFIER
)

te_stats["Block_Pct_Eligible"] = (
    te_stats["Total_Blocking_Snaps"]
    .fillna(0) >=
    BLOCK_QUALIFIER
)

te_stats["Target_Stat_Confidence"] = (
    np.minimum(
        te_stats["Targets"]
        /
        TARGET_QUALIFIER,
        1.0
    )
)

te_stats["Route_Stat_Confidence"] = np.where(
    te_stats["Routes"].notna(),
    np.minimum(
        te_stats["Routes"] /
        ROUTE_QUALIFIER,
        1.0
    ),
    np.nan
)

te_stats["Block_Stat_Confidence"] = np.where(
    te_stats["Total_Blocking_Snaps"].notna(),
    np.minimum(
        te_stats["Total_Blocking_Snaps"] /
        BLOCK_QUALIFIER,
        1.0
    ),
    np.nan
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


hist_targets = hist_pbp[
    hist_pbp[
        "receiver_player_id"
    ].notna()
].copy()


hist_te = (
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

        EPA_per_Target=(
            "epa",
            "mean"
        ),

        Target_Success_Rate=(
            "success",
            "mean"
        ),

        Catch_Rate_Over_Expected=(
            "cpoe",
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


hist_te[
    "Target_Success_Rate"
] *= 100


# Add player position and PFR IDs so RB/TE targets
# do not enter TE historical percentiles.
hist_te = hist_te.merge(
    players[
        [
            "gsis_id",
            "position",
            "pfr_id"
        ]
    ],
    left_on="Player_ID",
    right_on="gsis_id",
    how="left"
)


hist_te = hist_te[
    hist_te["position"] == "TE"
].copy()


# ==================================================
# HISTORICAL NEXT GEN RECEIVING
# ==================================================

try:

    hist_ngs = nfl.load_nextgen_stats(
        seasons=HISTORICAL_SEASONS,
        stat_type="receiving"
    ).to_pandas()

    if "season_type" in hist_ngs.columns:

        hist_ngs = hist_ngs[
            hist_ngs[
                "season_type"
            ] == "REG"
        ].copy()

    if "player_position" in hist_ngs.columns:

        hist_ngs = hist_ngs[
            hist_ngs[
                "player_position"
            ] == "TE"
        ].copy()

    hist_ngs["targets"] = pd.to_numeric(
        hist_ngs["targets"],
        errors="coerce"
    ).fillna(0)

    hist_ngs["receptions"] = pd.to_numeric(
        hist_ngs["receptions"],
        errors="coerce"
    ).fillna(0)

    hist_ngs["Sep_Weighted"] = (
        pd.to_numeric(
            hist_ngs["avg_separation"],
            errors="coerce"
        )
        *
        hist_ngs["targets"]
    )

    hist_ngs["YACOE_Weighted"] = (
        pd.to_numeric(
            hist_ngs[
                "avg_yac_above_expectation"
            ],
            errors="coerce"
        )
        *
        hist_ngs["receptions"]
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
            NGS_Targets=(
                "targets",
                "sum"
            ),

            NGS_Receptions=(
                "receptions",
                "sum"
            ),

            Sep_Weighted=(
                "Sep_Weighted",
                "sum"
            ),

            YACOE_Weighted=(
                "YACOE_Weighted",
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

    hist_ngs_season[
        "Separation_Rate"
    ] = (
        hist_ngs_season[
            "Sep_Weighted"
        ]
        /
        hist_ngs_season[
            "NGS_Targets"
        ].replace(0, np.nan)
    )

    hist_ngs_season[
        "YAC_Over_Expected"
    ] = (
        hist_ngs_season[
            "YACOE_Weighted"
        ]
        /
        hist_ngs_season[
            "NGS_Receptions"
        ].replace(0, np.nan)
    )

    hist_te = hist_te.merge(
        hist_ngs_season[
            [
                "season",
                "Player_ID",
                "Separation_Rate",
                "YAC_Over_Expected"
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
        f"Historical NGS receiving data failed: {error}"
    )

    hist_te["Separation_Rate"] = np.nan
    hist_te["YAC_Over_Expected"] = np.nan


# ==================================================
# HISTORICAL PFR ADVANCED RECEIVING
# ==================================================

try:

    hist_pfr = nfl.load_pfr_advstats(
        seasons=HISTORICAL_SEASONS,
        stat_type="rec",
        summary_level="season"
    ).to_pandas()

    hist_pfr_id_column = (
        "pfr_id"
        if "pfr_id" in hist_pfr.columns
        else "pfr_player_id"
    )

    hist_drop_pct_column = first_existing_column(
        hist_pfr,
        [
            "drop_pct",
            "receiving_drop_pct"
        ]
    )

    hist_drops_column = first_existing_column(
        hist_pfr,
        [
            "drops",
            "receiving_drop"
        ]
    )

    hist_broken_tackles_column = first_existing_column(
        hist_pfr,
        [
            "brk_tkl",
            "broken_tackles",
            "receiving_broken_tackles",
            "receiving_brk_tkl"
        ]
    )

    hist_receptions_column = first_existing_column(
        hist_pfr,
        [
            "receptions",
            "rec"
        ]
    )

    hist_targets_column = first_existing_column(
        hist_pfr,
        [
            "targets",
            "tgt"
        ]
    )

    hist_pfr_clean = pd.DataFrame({
        "season":
            hist_pfr["season"],

        "pfr_id":
            hist_pfr[
                hist_pfr_id_column
            ],

        "Drop_Rate":
            safe_numeric(
                hist_pfr,
                hist_drop_pct_column
            ),

        "Drops":
            safe_numeric(
                hist_pfr,
                hist_drops_column
            ),

        "Broken_Tackles":
            safe_numeric(
                hist_pfr,
                hist_broken_tackles_column
            ),

        "PFR_Receptions":
            safe_numeric(
                hist_pfr,
                hist_receptions_column
            ),

        "PFR_Targets":
            safe_numeric(
                hist_pfr,
                hist_targets_column
            )
    })

    if (
        hist_drop_pct_column is None
        and hist_drops_column is not None
        and hist_targets_column is not None
    ):

        hist_pfr_clean[
            "Drop_Rate"
        ] = (
            hist_pfr_clean[
                "Drops"
            ]
            /
            hist_pfr_clean[
                "PFR_Targets"
            ].replace(0, np.nan)
            *
            100
        )

    hist_pfr_clean[
        "Missed_Tackles_Per_Touch"
    ] = (
        hist_pfr_clean[
            "Broken_Tackles"
        ]
        /
        hist_pfr_clean[
            "PFR_Receptions"
        ].replace(0, np.nan)
    )

    hist_te = hist_te.merge(
        hist_pfr_clean[
            [
                "season",
                "pfr_id",
                "Drop_Rate",
                "Missed_Tackles_Per_Touch"
            ]
        ],
        on=[
            "season",
            "pfr_id"
        ],
        how="left"
    )

except Exception as error:

    print(
        f"Historical PFR receiving data failed: {error}"
    )

    hist_te["Drop_Rate"] = np.nan
    hist_te[
        "Missed_Tackles_Per_Touch"
    ] = np.nan


# Future historical metrics.
hist_te["Routes"] = np.nan
hist_te["Yards_Per_Route"] = np.nan
hist_te["Targets_Per_Route"] = np.nan
hist_te["Route_Win_Rate"] = np.nan
hist_te[
    "Contested_Catch_Win_Rate"
] = np.nan
hist_te["Press_Win_Rate"] = np.nan
hist_te["Run_Block_Snaps"] = np.nan
hist_te["Pass_Block_Snaps"] = np.nan
hist_te["Run_Block_Efficiency"] = np.nan
hist_te["Pass_Block_Efficiency"] = np.nan
hist_te["Total_Blocking_Snaps"] = (
    hist_te[[
        "Run_Block_Snaps",
        "Pass_Block_Snaps"
    ]].sum(axis=1, min_count=1)
)


hist_target_eligible = (
    hist_te["Targets"] >=
    TARGET_QUALIFIER
)

hist_route_eligible = (
    hist_te["Routes"]
    .fillna(0) >=
    ROUTE_QUALIFIER
)

hist_block_eligible = (
    hist_te["Total_Blocking_Snaps"]
    .fillna(0) >=
    BLOCK_QUALIFIER
)


# ==================================================
# FIVE-YEAR HISTORICAL PERCENTILES
# ==================================================

add_historical_percentile(
    te_stats,
    hist_te,
    "EPA_per_Target",
    "EPA_Target_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    te_stats,
    hist_te,
    "Separation_Rate",
    "Separation_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    te_stats,
    hist_te,
    "Target_Success_Rate",
    "Target_Success_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    te_stats,
    hist_te,
    "Catch_Rate_Over_Expected",
    "Catch_Rate_Over_Expected_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    te_stats,
    hist_te,
    "YAC_Over_Expected",
    "YAC_Over_Expected_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    te_stats,
    hist_te,
    "Missed_Tackles_Per_Touch",
    "Missed_Tackles_Per_Touch_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    te_stats,
    hist_te,
    "Drop_Rate",
    "Drop_Rate_Pctl",
    hist_target_eligible,
    False
)


# Route-based and later charting percentiles.
for percentile_column in [
    "Yards_Per_Route_Pctl",
    "Targets_Per_Route_Pctl",
    "Route_Win_Rate_Pctl",
    "Contested_Catch_Win_Rate_Pctl",
    "Press_Win_Rate_Pctl",
    "Run_Block_Efficiency_Pctl",
    "Pass_Block_Efficiency_Pctl"
]:
    te_stats[percentile_column] = np.nan


# ==================================================
# SAMPLE-SIZE REGRESSION
# ==================================================

TARGET_BASED_PERCENTILES = [
    "EPA_Target_Pctl",
    "Separation_Pctl",
    "Target_Success_Pctl",
    "Catch_Rate_Over_Expected_Pctl",
    "YAC_Over_Expected_Pctl",
    "Missed_Tackles_Per_Touch_Pctl",
    "Drop_Rate_Pctl"
]


for column in TARGET_BASED_PERCENTILES:

    te_stats[
        column + "_Raw"
    ] = te_stats[column]

    te_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        te_stats[
            "Target_Stat_Confidence"
        ]
        *
        (
            te_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )


# Route-based metrics regress toward neutral using route volume once available.
ROUTE_BASED_PERCENTILES = [
    "Yards_Per_Route_Pctl",
    "Targets_Per_Route_Pctl",
    "Route_Win_Rate_Pctl",
    "Press_Win_Rate_Pctl"
]

for column in ROUTE_BASED_PERCENTILES:
    if column in te_stats.columns:
        te_stats[column + "_Raw"] = te_stats[column]
        if te_stats["Route_Stat_Confidence"].notna().any():
            te_stats[column] = (
                NEUTRAL_PERCENTILE +
                te_stats["Route_Stat_Confidence"] *
                (te_stats[column] - NEUTRAL_PERCENTILE)
            )

# Blocking metrics regress toward neutral using blocking snaps once available.
BLOCK_BASED_PERCENTILES = [
    "Run_Block_Efficiency_Pctl",
    "Pass_Block_Efficiency_Pctl"
]

for column in BLOCK_BASED_PERCENTILES:
    if column in te_stats.columns:
        te_stats[column + "_Raw"] = te_stats[column]
        if te_stats["Block_Stat_Confidence"].notna().any():
            te_stats[column] = (
                NEUTRAL_PERCENTILE +
                te_stats["Block_Stat_Confidence"] *
                (te_stats[column] - NEUTRAL_PERCENTILE)
            )


# ==================================================
# TE ADVANCED WEIGHTS — STARTING POINT
#
# Missing metrics are automatically ignored and the
# available weights are renormalized.
# We will tune these after looking at the first rankings.
# ==================================================

TE_RECEIVING_WEIGHTS = {
    # TE receiving is judged against TE history, not WR history.
    # Separation/YAC matter, but less than for WRs because alignment and role
    # make those categories structurally harder for many tight ends.
    "EPA_Target_Pctl": 0.17,
    "Separation_Pctl": 0.06,
    "Target_Success_Pctl": 0.12,
    "Yards_Per_Route_Pctl": 0.15,
    "Targets_Per_Route_Pctl": 0.10,
    "Catch_Rate_Over_Expected_Pctl": 0.10,
    "Route_Win_Rate_Pctl": 0.08,
    "YAC_Over_Expected_Pctl": 0.05,
    "Missed_Tackles_Per_Touch_Pctl": 0.04,
    "Drop_Rate_Pctl": 0.04,
    "Contested_Catch_Win_Rate_Pctl": 0.08,
    "Press_Win_Rate_Pctl": 0.01
}

TE_BLOCKING_WEIGHTS = {
    "Run_Block_Efficiency_Pctl": 0.65,
    "Pass_Block_Efficiency_Pctl": 0.35
}

te_stats["Receiving_Core_Pctl"] = te_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=TE_RECEIVING_WEIGHTS
)

te_stats["Blocking_Component_Pctl"] = te_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=TE_BLOCKING_WEIGHTS
)

TOTAL_RECEIVING_WEIGHT = sum(
    TE_RECEIVING_WEIGHTS.values()
)
TOTAL_BLOCKING_WEIGHT = sum(
    TE_BLOCKING_WEIGHTS.values()
)


def calculate_available_weight(row, weights):
    return sum(
        weight
        for metric, weight in weights.items()
        if pd.notna(row.get(metric)) and weight > 0
    )


te_stats["Receiving_Available_Weight"] = te_stats.apply(
    calculate_available_weight,
    axis=1,
    weights=TE_RECEIVING_WEIGHTS
)

te_stats["Blocking_Available_Weight"] = te_stats.apply(
    calculate_available_weight,
    axis=1,
    weights=TE_BLOCKING_WEIGHTS
)

te_stats["Receiving_Data_Coverage"] = (
    te_stats["Receiving_Available_Weight"] /
    TOTAL_RECEIVING_WEIGHT
)

te_stats["Blocking_Data_Coverage"] = (
    te_stats["Blocking_Available_Weight"] /
    TOTAL_BLOCKING_WEIGHT
)

# Missing blocking data reduces how much authority the advanced model gets,
# rather than pretending the incomplete receiving-only model is complete.
te_stats["Advanced_Data_Coverage"] = (
    RECEIVING_MODEL_SHARE *
    te_stats["Receiving_Data_Coverage"]
    +
    BLOCKING_MODEL_SHARE *
    te_stats["Blocking_Data_Coverage"]
)

# Future role usage: routes versus actual run/pass-block snaps.
te_stats["Total_TE_Role_Snaps"] = (
    te_stats[["Routes", "Total_Blocking_Snaps"]]
    .sum(axis=1, min_count=1)
)

te_stats["Blocking_Play_Share"] = np.where(
    te_stats["Total_TE_Role_Snaps"].fillna(0) > 0,
    te_stats["Total_Blocking_Snaps"].fillna(0) /
    te_stats["Total_TE_Role_Snaps"],
    np.nan
)

te_stats["Receiving_Role_Share"] = np.where(
    te_stats["Total_TE_Role_Snaps"].fillna(0) > 0,
    te_stats["Routes"].fillna(0) /
    te_stats["Total_TE_Role_Snaps"],
    np.nan
)

# Receiving establishes the TE baseline. Blocking can materially add or subtract
# value based on both blocking quality and how often the TE is actually used there.
te_stats["Blocking_Adjustment"] = np.where(
    te_stats["Blocking_Component_Pctl"].notna() &
    te_stats["Blocking_Play_Share"].notna(),
    (
        te_stats["Blocking_Component_Pctl"] -
        NEUTRAL_PERCENTILE
    ) *
    te_stats["Blocking_Play_Share"] *
    BLOCKING_BOOST_MULTIPLIER,
    0.0
)

te_stats["Advanced_Weighted_Pctl"] = (
    te_stats["Receiving_Core_Pctl"].fillna(NEUTRAL_PERCENTILE) +
    te_stats["Blocking_Adjustment"]
).clip(lower=0, upper=100)

te_stats["Advanced_Rating"] = (
    60 +
    te_stats["Advanced_Weighted_Pctl"] * 0.40
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
    .apply(
        normalize_player_name
    )
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


te_stats["Name_Key"] = (
    te_stats["Player"]
    .apply(
        normalize_player_name
    )
)

te_stats["Pos_Key"] = "TE"

te_stats["Team_Key"] = (
    te_stats["Team_2025"]
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


te_stats = te_stats.merge(
    legacy_primary,
    on=[
        "Name_Key",
        "Pos_Key",
        "Team_Key"
    ],
    how="left"
)


te_stats[
    "Legacy_Match_Status"
] = np.where(
    te_stats["Madden"].notna(),
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
    legacy_fallback[
        "Match_Count"
    ] == 1
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


legacy_fallback = (
    legacy_fallback
    .rename(
        columns={
            "Age":
                "Fallback_Age",

            "Madden":
                "Fallback_Madden",

            "PFF":
                "Fallback_PFF",

            "PFR":
                "Fallback_PFR"
        }
    )
)


te_stats = te_stats.merge(
    legacy_fallback,
    on=[
        "Name_Key",
        "Pos_Key"
    ],
    how="left"
)


needs_fallback = (
    te_stats["Madden"].isna()
)


for column in [
    "Age",
    "Madden",
    "PFF",
    "PFR"
]:

    fallback_column = (
        "Fallback_" + column
    )

    te_stats.loc[
        needs_fallback,
        column
    ] = te_stats.loc[
        needs_fallback,
        fallback_column
    ]


fallback_success = (
    needs_fallback
    &
    te_stats[
        "Fallback_Madden"
    ].notna()
)


te_stats.loc[
    fallback_success,
    "Legacy_Match_Status"
] = "NAME+POS"

# ==================================================
# AGE FALLBACK FROM NFLVERSE BIRTH DATE
# ==================================================

# 2026 regular season begins September 9, 2026.
# Keep legacy Age when available.
# Only calculate age when legacy Age is missing.

AGE_REFERENCE_DATE = pd.Timestamp(
    "2026-09-09"
)

birth_dates = pd.to_datetime(
    te_stats["birth_date"],
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


te_stats["Age_Source"] = np.select(
    [
        te_stats["Age"].notna(),
        calculated_age.notna()
    ],
    [
        "LEGACY",
        "NFLVERSE_BIRTH_DATE"
    ],
    default="MISSING"
)


te_stats["Age"] = (
    te_stats["Age"]
    .fillna(
        calculated_age
    )
)

te_stats = te_stats.drop(
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
#
# Same legacy architecture as the QB model, but if
# one source is missing the remaining sources are
# proportionally reweighted instead of blanking the TE.
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
            LEGACY_WEIGHTS[
                "Madden"
            ]
        )

        available_weight += (
            LEGACY_WEIGHTS[
                "Madden"
            ]
        )

    if pd.notna(row["PFF"]):

        weighted_total += (
            row["PFF"]
            *
            1.1
            *
            LEGACY_WEIGHTS[
                "PFF"
            ]
        )

        available_weight += (
            LEGACY_WEIGHTS[
                "PFF"
            ]
        )

    if pd.notna(row["PFR"]):

        weighted_total += (
            row["PFR"]
            *
            5.2
            *
            LEGACY_WEIGHTS[
                "PFR"
            ]
        )

        available_weight += (
            LEGACY_WEIGHTS[
                "PFR"
            ]
        )

    if available_weight == 0:
        return np.nan

    return (
        weighted_total
        /
        available_weight
    )


te_stats["Legacy_Rating"] = (
    te_stats.apply(
        calculate_legacy_rating,
        axis=1
    )
)

te_stats["Legacy_Fallback_Applied"] = (
    te_stats["Legacy_Rating"].isna()
)

te_stats["Legacy_Rating_Used"] = (
    te_stats["Legacy_Rating"]
    .fillna(MISSING_LEGACY_RATING)
)

# ==================================================
# FINAL ADVANCED / LEGACY CONFIDENCE
# ==================================================

te_stats["Receiving_Opportunity_Confidence"] = np.minimum(
    te_stats["Targets"] /
    FULL_CONFIDENCE_TARGETS,
    1.0
)

te_stats["Role_Snap_Confidence"] = np.where(
    te_stats["Total_TE_Role_Snaps"].notna(),
    np.minimum(
        te_stats["Total_TE_Role_Snaps"] /
        FULL_CONFIDENCE_ROLE_SNAPS,
        1.0
    ),
    0.0
)

# For now this is target-driven because routes/block snaps are unavailable.
# Later, a true blocking specialist can reach full confidence through role snaps.
te_stats["Opportunity_Confidence"] = np.maximum(
    te_stats["Receiving_Opportunity_Confidence"],
    te_stats["Role_Snap_Confidence"]
)


te_stats[
    "Actual_Advanced_Weight"
] = (
    BASE_ADVANCED_OVR_WEIGHT
    *
    te_stats[
        "Opportunity_Confidence"
    ]
    *
    te_stats[
        "Advanced_Data_Coverage"
    ]
)


te_stats[
    "Actual_Legacy_Weight"
] = (
    1
    -
    te_stats[
        "Actual_Advanced_Weight"
    ]
)


def calculate_pre_bonus_overall(row):

    advanced = row[
        "Advanced_Rating"
    ]

    legacy = row[
        "Legacy_Rating_Used"
    ]

    advanced_weight = row[
        "Actual_Advanced_Weight"
    ]

    legacy_weight = row[
        "Actual_Legacy_Weight"
    ]

    if (
        pd.notna(advanced)
        and pd.notna(legacy)
    ):

        return (
            advanced
            *
            advanced_weight
            +
            legacy
            *
            legacy_weight
        )

    if pd.notna(legacy):
        return legacy

    if pd.notna(advanced):
        return advanced

    return np.nan


te_stats[
    "Pre_Bonus_Overall"
] = te_stats.apply(
    calculate_pre_bonus_overall,
    axis=1
)


# ==================================================
# HISTORICAL RECORD BONUS
#
# Intentionally zero for TE V1.
# We will build TE-specific record categories only
# after the core ranking model is behaving correctly.
# ==================================================

te_stats[
    "Historical_Record_Bonus"
] = 0.0


te_stats[
    "Final_Overall"
] = (
    te_stats[
        "Pre_Bonus_Overall"
    ]
    +
    te_stats[
        "Historical_Record_Bonus"
    ]
)


# ==================================================
# AGE-ADJUSTED VALUE
# ==================================================

te_stats[
    "Age_Adjusted_Overall"
] = (
    te_stats[
        "Final_Overall"
    ]
    *
    (
        (
            100
            +
            (
                TE_PRIME_AGE
                -
                te_stats["Age"]
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
    "Final_Overall",
    "Age_Adjusted_Overall",    
    "Legacy_Rating",
    "Legacy_Rating_Used",
    "Legacy_Fallback_Applied",
    "Advanced_Rating",

    "Targets",
    "Receptions",
    "Receiving_Yards",
    "Receiving_TDs",

    "Legacy_Match_Status",  
    "Advanced_Weighted_Pctl",
    "Pre_Bonus_Overall",
    "Historical_Record_Bonus",


    "Target_Pct_Eligible",
    "Route_Pct_Eligible",
    "Block_Pct_Eligible",
    "Target_Stat_Confidence",
    "Route_Stat_Confidence",
    "Block_Stat_Confidence",
    "Receiving_Opportunity_Confidence",
    "Role_Snap_Confidence",
    "Opportunity_Confidence",
    "Actual_Advanced_Weight",
    "Actual_Legacy_Weight",

    "EPA_per_Target",
    "EPA_Target_Pctl",
    "EPA_Target_Pctl_Raw",

    "Separation_Rate",
    "Separation_Pctl",
    "Separation_Pctl_Raw",

    "Target_Success_Rate",
    "Target_Success_Pctl",
    "Target_Success_Pctl_Raw",

    "Routes",
    "Yards_Per_Route",
    "Yards_Per_Route_Pctl",

    "Targets_Per_Route",
    "Targets_Per_Route_Pctl",

    "Catch_Rate_Over_Expected",
    "Catch_Rate_Over_Expected_Pctl",
    "Catch_Rate_Over_Expected_Pctl_Raw",

    "Route_Win_Rate",
    "Route_Win_Rate_Pctl",

    "YAC_Over_Expected",
    "YAC_Over_Expected_Pctl",
    "YAC_Over_Expected_Pctl_Raw",

    "Missed_Tackles_Per_Touch",
    "Missed_Tackles_Per_Touch_Pctl",
    "Missed_Tackles_Per_Touch_Pctl_Raw",

    "Drop_Rate",
    "Drop_Rate_Pctl",
    "Drop_Rate_Pctl_Raw",

    "Contested_Catch_Win_Rate",
    "Contested_Catch_Win_Rate_Pctl",

    "Press_Win_Rate",
    "Press_Win_Rate_Pctl",

    "Run_Block_Snaps",
    "Run_Block_Efficiency",
    "Run_Block_Efficiency_Pctl",

    "Pass_Block_Snaps",
    "Pass_Block_Efficiency",
    "Pass_Block_Efficiency_Pctl",

    "Total_Blocking_Snaps",
    "Total_TE_Role_Snaps",
    "Receiving_Role_Share",
    "Blocking_Play_Share",
    "Receiving_Core_Pctl",
    "Blocking_Component_Pctl",
    "Blocking_Adjustment",
    "Receiving_Data_Coverage",
    "Blocking_Data_Coverage",
    "Advanced_Data_Coverage",

    "Total_Receiving_EPA",

    "Madden",
    "PFF",
    "PFR",
    "Age_Source",

    "pfr_id",
    "pff_id",
    "espn_id"
]

missing_age_players = te_stats[
    te_stats["Age"].isna()
][
    [
        "Player",
        "Player_ID",
        "Team_2025"
    ]
]


if not missing_age_players.empty:

    print(
        "\nPlayers still missing age:"
    )

    print(
        missing_age_players.to_string(
            index=False
        )
    )
    
# Protect V1 output against an optional source failing
# before its columns are created.
for column in output_columns:

    if column not in te_stats.columns:
        te_stats[column] = np.nan


te_output = te_stats[
    output_columns
].copy()


te_output = te_output.round(
    {
        "Age": 1,

        "Legacy_Rating": 2,
        "Advanced_Weighted_Pctl": 2,
        "Advanced_Rating": 2,
        "Pre_Bonus_Overall": 2,
        "Historical_Record_Bonus": 2,
        "Final_Overall": 2,
        "Age_Adjusted_Overall": 2,

        "Target_Stat_Confidence": 3,
        "Route_Stat_Confidence": 3,
        "Block_Stat_Confidence": 3,
        "Receiving_Opportunity_Confidence": 3,
        "Role_Snap_Confidence": 3,
        "Opportunity_Confidence": 3,
        "Actual_Advanced_Weight": 3,
        "Actual_Legacy_Weight": 3,

        "EPA_per_Target": 3,
        "EPA_Target_Pctl": 1,
        "EPA_Target_Pctl_Raw": 1,

        "Separation_Rate": 3,
        "Separation_Pctl": 1,
        "Separation_Pctl_Raw": 1,

        "Target_Success_Rate": 2,
        "Target_Success_Pctl": 1,
        "Target_Success_Pctl_Raw": 1,

        "Yards_Per_Route": 3,
        "Yards_Per_Route_Pctl": 1,

        "Targets_Per_Route": 3,
        "Targets_Per_Route_Pctl": 1,

        "Catch_Rate_Over_Expected": 2,
        "Catch_Rate_Over_Expected_Pctl": 1,
        "Catch_Rate_Over_Expected_Pctl_Raw": 1,

        "Route_Win_Rate": 2,
        "Route_Win_Rate_Pctl": 1,

        "YAC_Over_Expected": 3,
        "YAC_Over_Expected_Pctl": 1,
        "YAC_Over_Expected_Pctl_Raw": 1,

        "Missed_Tackles_Per_Touch": 3,
        "Missed_Tackles_Per_Touch_Pctl": 1,
        "Missed_Tackles_Per_Touch_Pctl_Raw": 1,

        "Drop_Rate": 2,
        "Drop_Rate_Pctl": 1,
        "Drop_Rate_Pctl_Raw": 1,

        "Contested_Catch_Win_Rate": 2,
        "Contested_Catch_Win_Rate_Pctl": 1,

        "Press_Win_Rate": 2,
        "Press_Win_Rate_Pctl": 1,

        "Run_Block_Efficiency": 2,
        "Run_Block_Efficiency_Pctl": 1,
        "Pass_Block_Efficiency": 2,
        "Pass_Block_Efficiency_Pctl": 1,
        "Receiving_Role_Share": 3,
        "Blocking_Play_Share": 3,
        "Receiving_Core_Pctl": 2,
        "Blocking_Component_Pctl": 2,
        "Blocking_Adjustment": 2,
        "Receiving_Data_Coverage": 3,
        "Blocking_Data_Coverage": 3,
        "Advanced_Data_Coverage": 3,

        "Total_Receiving_EPA": 2,

        "Madden": 1,
        "PFF": 1,
        "PFR": 1
    }
)


te_output = te_output.sort_values(
    "Final_Overall",
    ascending=False,
    na_position="last"
)


top_25 = te_output.head(25).copy()


# ==================================================
# SAVE PARQUET FOR FUTURE MASTER WORKBOOK
# ==================================================

te_output.to_parquet(
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
    ) as teiter:

        te_output.to_excel(
            teiter,
            sheet_name="All_TEs",
            index=False
        )

        top_25.to_excel(
            teiter,
            sheet_name="Top_25_Check",
            index=False
        )

        for worksheet in (
            teiter.book.worksheets
        ):

            worksheet.freeze_panes = "A2"

            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

            for column_cells in (
                worksheet.columns
            ):

                max_length = max(
                    (
                        len(
                            str(cell.value)
                        )
                        if cell.value
                        is not None
                        else 0
                    )
                    for cell
                    in column_cells
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
        f"with {len(te_output)} TEs."
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
