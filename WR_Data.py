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

# Full-confidence thresholds for the individual WR metrics.
TARGET_QUALIFIER = 50
ROUTE_QUALIFIER = 200

# Final advanced-vs-legacy confidence.
# This is intentionally higher than TARGET_QUALIFIER, mirroring the QB model:
# a stat can be trustworthy enough for its percentile before the entire season
# is trustworthy enough to receive the maximum Advanced weight.
FULL_CONFIDENCE_TARGETS = 100

OUTPUT_FILE = f"WR_Advanced_{SEASON}.xlsx"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PARQUET_OUTPUT = OUTPUT_DIR / f"WR_{SEASON}.parquet"

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
WR_PRIME_AGE = 29.0
MISSING_LEGACY_RATING = 65.0


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
    Every current WR with a raw value receives a percentile.

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
# BASE WR POPULATION
# ==================================================

season_stats = nfl.load_player_stats(
    [SEASON],
    summary_level="reg"
).to_pandas()


current_wr = season_stats[
    season_stats["position"] == "WR"
].copy()


# Protect against an unexpected duplicate player row.
current_wr = (
    current_wr
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
if "team" in current_wr.columns:
    current_wr_team = current_wr["team"]
elif "recent_team" in current_wr.columns:
    current_wr_team = current_wr["recent_team"]
else:
    current_wr_team = pd.Series(
        np.nan,
        index=current_wr.index,
        dtype="object"
    )


wr_stats = pd.DataFrame({
    "Player_ID":
        current_wr["player_id"],

    "Short_Name":
        current_wr["player_name"],

    "Player":
        current_wr[
            "player_display_name"
        ].fillna(
            current_wr["player_name"]
        ),

    "Team_2025":
        current_wr_team,

    "Position":
        "WR",

    "Season_Targets":
        pd.to_numeric(
            current_wr["targets"],
            errors="coerce"
        ),

    "Receptions":
        pd.to_numeric(
            current_wr["receptions"],
            errors="coerce"
        ),

    "Receiving_Yards":
        pd.to_numeric(
            current_wr["receiving_yards"],
            errors="coerce"
        ),

    "Receiving_TDs":
        pd.to_numeric(
            current_wr["receiving_tds"],
            errors="coerce"
        ),

    "Total_Receiving_EPA":
        pd.to_numeric(
            current_wr["receiving_epa"],
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


wr_stats = wr_stats.merge(
    target_stats,
    on="Player_ID",
    how="left"
)


wr_stats["Targets"] = (
    wr_stats["PBP_Targets"]
    .fillna(
        wr_stats["Season_Targets"]
    )
    .fillna(0)
)


# PBP is the better 2025 team indicator for receivers who were traded.
wr_stats["Team_2025"] = (
    wr_stats["Team_From_PBP"]
    .fillna(
        wr_stats["Team_2025"]
    )
)


wr_stats["Target_Success_Rate"] = (
    wr_stats[
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


wr_stats = wr_stats.merge(
    players,
    left_on="Player_ID",
    right_on="gsis_id",
    how="left",
    suffixes=("", "_player_map")
)


wr_stats["Player"] = (
    wr_stats["display_name"]
    .fillna(
        wr_stats["Player"]
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
            ngs["player_position"] == "WR"
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

    wr_stats = wr_stats.merge(
        ngs_current,
        on="Player_ID",
        how="left"
    )

except Exception as error:

    print(
        f"NGS receiving data failed: {error}"
    )

    wr_stats["Avg_Separation"] = np.nan
    wr_stats[
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

    wr_stats = wr_stats.merge(
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

    wr_stats["Drop_Rate"] = (
        wr_stats["PFR_Drop_Rate"]
    )

except Exception as error:

    print(
        f"PFR receiving data failed: {error}"
    )

    wr_stats["Drop_Rate"] = np.nan
    wr_stats[
        "Missed_Tackles_Per_Touch"
    ] = np.nan


# ==================================================
# FUTURE WEBSITE / ROUTE-CHARTING METRICS
#
# Columns are intentionally present now so they can
# be plugged into the model later without rebuilding it.
# ==================================================

wr_stats["Routes"] = np.nan
wr_stats["Yards_Per_Route"] = np.nan
wr_stats["Targets_Per_Route"] = np.nan
wr_stats["Route_Win_Rate"] = np.nan
wr_stats[
    "Contested_Catch_Win_Rate"
] = np.nan
wr_stats["Press_Win_Rate"] = np.nan
wr_stats[
    "Run_Block_Efficiency"
] = np.nan

# The requested long-term metric is "Separation Rate".
# For V1, Avg_Separation is the available NGS proxy.
wr_stats["Separation_Rate"] = (
    wr_stats["Avg_Separation"]
)


# ==================================================
# QUALIFICATION / CONFIDENCE FLAGS
# ==================================================

wr_stats["Target_Pct_Eligible"] = (
    wr_stats["Targets"] >=
    TARGET_QUALIFIER
)

wr_stats["Route_Pct_Eligible"] = False

wr_stats["Target_Stat_Confidence"] = (
    np.minimum(
        wr_stats["Targets"]
        /
        TARGET_QUALIFIER,
        1.0
    )
)

# When route counts become available, replace this
# with min(Routes / ROUTE_QUALIFIER, 1.0).
wr_stats["Route_Stat_Confidence"] = np.nan


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


hist_wr = (
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


hist_wr[
    "Target_Success_Rate"
] *= 100


# Add player position and PFR IDs so RB/TE targets
# do not enter WR historical percentiles.
hist_wr = hist_wr.merge(
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


hist_wr = hist_wr[
    hist_wr["position"] == "WR"
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
            ] == "WR"
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

    hist_wr = hist_wr.merge(
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

    hist_wr["Separation_Rate"] = np.nan
    hist_wr["YAC_Over_Expected"] = np.nan


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

    hist_wr = hist_wr.merge(
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

    hist_wr["Drop_Rate"] = np.nan
    hist_wr[
        "Missed_Tackles_Per_Touch"
    ] = np.nan


# Future historical metrics.
hist_wr["Routes"] = np.nan
hist_wr["Yards_Per_Route"] = np.nan
hist_wr["Targets_Per_Route"] = np.nan
hist_wr["Route_Win_Rate"] = np.nan
hist_wr[
    "Contested_Catch_Win_Rate"
] = np.nan
hist_wr["Press_Win_Rate"] = np.nan
hist_wr[
    "Run_Block_Efficiency"
] = np.nan


hist_target_eligible = (
    hist_wr["Targets"] >=
    TARGET_QUALIFIER
)

hist_route_eligible = (
    hist_wr["Routes"]
    .fillna(0) >=
    ROUTE_QUALIFIER
)


# ==================================================
# FIVE-YEAR HISTORICAL PERCENTILES
# ==================================================

add_historical_percentile(
    wr_stats,
    hist_wr,
    "EPA_per_Target",
    "EPA_Target_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    wr_stats,
    hist_wr,
    "Separation_Rate",
    "Separation_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    wr_stats,
    hist_wr,
    "Target_Success_Rate",
    "Target_Success_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    wr_stats,
    hist_wr,
    "Catch_Rate_Over_Expected",
    "Catch_Rate_Over_Expected_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    wr_stats,
    hist_wr,
    "YAC_Over_Expected",
    "YAC_Over_Expected_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    wr_stats,
    hist_wr,
    "Missed_Tackles_Per_Touch",
    "Missed_Tackles_Per_Touch_Pctl",
    hist_target_eligible,
    True
)

add_historical_percentile(
    wr_stats,
    hist_wr,
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
    "Run_Block_Efficiency_Pctl"
]:
    wr_stats[percentile_column] = np.nan


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

    wr_stats[
        column + "_Raw"
    ] = wr_stats[column]

    wr_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        wr_stats[
            "Target_Stat_Confidence"
        ]
        *
        (
            wr_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )


# When route counts are added later, route-based metrics
# should use Route_Stat_Confidence in exactly the same way.


# ==================================================
# WR ADVANCED WEIGHTS — STARTING POINT
#
# Missing metrics are automatically ignored and the
# available weights are renormalized.
# We will tune these after looking at the first rankings.
# ==================================================

WR_ADVANCED_WEIGHTS = {
    "EPA_Target_Pctl": 0.16,
    "Separation_Pctl": 0.10,
    "Target_Success_Pctl": 0.12,
    "Yards_Per_Route_Pctl": 0.14,
    "Targets_Per_Route_Pctl": 0.10,
    "Catch_Rate_Over_Expected_Pctl": 0.10,
    "Route_Win_Rate_Pctl": 0.08,
    "YAC_Over_Expected_Pctl": 0.07,
    "Missed_Tackles_Per_Touch_Pctl": 0.04,
    "Drop_Rate_Pctl": 0.04,
    "Contested_Catch_Win_Rate_Pctl": 0.03,
    "Press_Win_Rate_Pctl": 0.01,
    "Run_Block_Efficiency_Pctl": 0.01
}


wr_stats[
    "Advanced_Weighted_Pctl"
] = wr_stats.apply(
    calculate_weighted_score,
    axis=1,
    weights=WR_ADVANCED_WEIGHTS
)

TOTAL_ADVANCED_WEIGHT = sum(
    WR_ADVANCED_WEIGHTS.values()
)


def calculate_available_advanced_weight(row):

    available_weight = 0.0

    for metric, weight in WR_ADVANCED_WEIGHTS.items():

        if pd.notna(row.get(metric)):
            available_weight += weight

    return available_weight


wr_stats["Advanced_Available_Weight"] = (
    wr_stats.apply(
        calculate_available_advanced_weight,
        axis=1
    )
)


wr_stats["Advanced_Data_Coverage"] = (
    wr_stats["Advanced_Available_Weight"]
    /
    TOTAL_ADVANCED_WEIGHT
)

wr_stats["Advanced_Rating"] = (
    60
    +
    wr_stats[
        "Advanced_Weighted_Pctl"
    ] * 0.40
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


wr_stats["Name_Key"] = (
    wr_stats["Player"]
    .apply(
        normalize_player_name
    )
)

wr_stats["Pos_Key"] = "WR"

wr_stats["Team_Key"] = (
    wr_stats["Team_2025"]
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


wr_stats = wr_stats.merge(
    legacy_primary,
    on=[
        "Name_Key",
        "Pos_Key",
        "Team_Key"
    ],
    how="left"
)


wr_stats[
    "Legacy_Match_Status"
] = np.where(
    wr_stats["Madden"].notna(),
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


wr_stats = wr_stats.merge(
    legacy_fallback,
    on=[
        "Name_Key",
        "Pos_Key"
    ],
    how="left"
)


needs_fallback = (
    wr_stats["Madden"].isna()
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

    wr_stats.loc[
        needs_fallback,
        column
    ] = wr_stats.loc[
        needs_fallback,
        fallback_column
    ]


fallback_success = (
    needs_fallback
    &
    wr_stats[
        "Fallback_Madden"
    ].notna()
)


wr_stats.loc[
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
    wr_stats["birth_date"],
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


wr_stats["Age_Source"] = np.select(
    [
        wr_stats["Age"].notna(),
        calculated_age.notna()
    ],
    [
        "LEGACY",
        "NFLVERSE_BIRTH_DATE"
    ],
    default="MISSING"
)


wr_stats["Age"] = (
    wr_stats["Age"]
    .fillna(
        calculated_age
    )
)

wr_stats = wr_stats.drop(
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
# proportionally reweighted instead of blanking the WR.
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


wr_stats["Legacy_Rating"] = (
    wr_stats.apply(
        calculate_legacy_rating,
        axis=1
    )
)

wr_stats["Legacy_Fallback_Applied"] = (
    wr_stats["Legacy_Rating"].isna()
)

wr_stats["Legacy_Rating_Used"] = (
    wr_stats["Legacy_Rating"]
    .fillna(MISSING_LEGACY_RATING)
)

# ==================================================
# FINAL ADVANCED / LEGACY CONFIDENCE
# ==================================================

wr_stats[
    "Opportunity_Confidence"
] = np.minimum(
    wr_stats["Targets"]
    /
    FULL_CONFIDENCE_TARGETS,
    1.0
)


wr_stats[
    "Actual_Advanced_Weight"
] = (
    BASE_ADVANCED_OVR_WEIGHT
    *
    wr_stats[
        "Opportunity_Confidence"
    ]
    *
    wr_stats[
        "Advanced_Data_Coverage"
    ]
)


wr_stats[
    "Actual_Legacy_Weight"
] = (
    1
    -
    wr_stats[
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


wr_stats[
    "Pre_Bonus_Overall"
] = wr_stats.apply(
    calculate_pre_bonus_overall,
    axis=1
)


# ==================================================
# HISTORICAL RECORD BONUS
#
# Intentionally zero for WR V1.
# We will build WR-specific record categories only
# after the core ranking model is behaving correctly.
# ==================================================

wr_stats[
    "Historical_Record_Bonus"
] = 0.0


wr_stats[
    "Final_Overall"
] = (
    wr_stats[
        "Pre_Bonus_Overall"
    ]
    +
    wr_stats[
        "Historical_Record_Bonus"
    ]
)


# ==================================================
# AGE-ADJUSTED VALUE
# ==================================================

wr_stats[
    "Age_Adjusted_Overall"
] = (
    wr_stats[
        "Final_Overall"
    ]
    *
    (
        (
            100
            +
            (
                WR_PRIME_AGE
                -
                wr_stats["Age"]
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
    "Target_Stat_Confidence",
    "Route_Stat_Confidence",
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

    "Run_Block_Efficiency",
    "Run_Block_Efficiency_Pctl",

    "Total_Receiving_EPA",

    "Madden",
    "PFF",
    "PFR",
    "Age_Source",

    "pfr_id",
    "pff_id",
    "espn_id"
]

missing_age_players = wr_stats[
    wr_stats["Age"].isna()
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

    if column not in wr_stats.columns:
        wr_stats[column] = np.nan


wr_output = wr_stats[
    output_columns
].copy()


wr_output = wr_output.round(
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

        "Total_Receiving_EPA": 2,

        "Madden": 1,
        "PFF": 1,
        "PFR": 1
    }
)


wr_output = wr_output.sort_values(
    "Final_Overall",
    ascending=False,
    na_position="last"
)


top_25 = wr_output.head(25).copy()


# ==================================================
# SAVE PARQUET FOR FUTURE MASTER WORKBOOK
# ==================================================

wr_output.to_parquet(
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

        wr_output.to_excel(
            writer,
            sheet_name="All_WRs",
            index=False
        )

        top_25.to_excel(
            writer,
            sheet_name="Top_25_Check",
            index=False
        )

        for worksheet in (
            writer.book.worksheets
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
        f"with {len(wr_output)} WRs."
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
