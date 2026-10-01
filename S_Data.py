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

# User-selected confidence thresholds.
TARGET_QUALIFIER = 30
DEFENSIVE_SNAP_QUALIFIER = 250
COVERAGE_SNAP_QUALIFIER = 200

# A full-time corner usually clears this comfortably. This only controls
# the maximum Advanced-vs-Legacy influence; the individual metric
# thresholds above control percentile sample-size regression.
FULL_CONFIDENCE_DEFENSIVE_SNAPS = 500

OUTPUT_FILE = f"S_Advanced_{SEASON}.xlsx"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PARQUET_OUTPUT = OUTPUT_DIR / f"S_{SEASON}.parquet"

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SNAP_CACHE = CACHE_DIR / f"snap_counts_{SEASON}.parquet"
ROSTER_CACHE = CACHE_DIR / f"rosters_{SEASON}.parquet"
PFR_DEF_CACHE = CACHE_DIR / f"pfr_def_{SEASON}.parquet"

HISTORICAL_SEASONS = list(
    range(SEASON - 4, SEASON + 1)
)

HIST_SNAP_CACHE = (
    CACHE_DIR /
    f"snap_counts_{HISTORICAL_SEASONS[0]}_{HISTORICAL_SEASONS[-1]}.parquet"
)

HIST_ROSTER_CACHE = (
    CACHE_DIR /
    f"rosters_{HISTORICAL_SEASONS[0]}_{HISTORICAL_SEASONS[-1]}.parquet"
)

HIST_PFR_DEF_CACHE = (
    CACHE_DIR /
    f"pfr_def_{HISTORICAL_SEASONS[0]}_{HISTORICAL_SEASONS[-1]}.parquet"
)

LEGACY_DB_FILE = "2026_NFLActive.xlsx"

BASE_ADVANCED_OVR_WEIGHT = 0.60
NEUTRAL_PERCENTILE = 50.0
S_PRIME_AGE = 28.0
MISSING_LEGACY_RATING = 65.0

CORNER_LABELS = {
    "CB",
    "LCB",
    "RCB",
    "NB",
    "NCB",
    "SCB",
}

SAFETY_LABELS = {
    "S",
    "FS",
    "SS",
    "SAF",
    "SAFETY",
}
def resolve_secondary_role(row):

    depth = normalize_position_label(
        row.get("Depth_Chart_Position")
    )

    snap = normalize_position_label(
        row.get("Snap_Position")
    )

    roster = normalize_position_label(
        row.get("Roster_Position")
    )

    for position in [depth, snap, roster]:

        if position in CORNER_LABELS:
            return "CB"

        if position in SAFETY_LABELS:
            return "S"

    return "UNRESOLVED"

# ==================================================
# SAFETY ADVANCED WEIGHTS — STARTING POINT
# ==================================================
#
# Safety is still primarily a coverage position, but the job includes more
# alley/run support and open-field tackling than safety. Explosive-play
# prevention also matters more because safeties are frequently the final
# layer of the defense.
#
# Coverage / explosive prevention: 77%
# Run defense / tackling:          23%
# ==================================================

S_ADVANCED_WEIGHTS = {
    "QBR_When_Targeted_Pctl": 0.10,
    "EPA_Per_Target_Pctl": 0.13,
    "Yards_Per_Coverage_Snap_Pctl": 0.10,
    "Coverage_Success_Rate_Pctl": 0.08,
    "Targets_Per_Coverage_Snap_Pctl": 0.05,
    "Completion_Pct_Over_Expected_Pctl": 0.07,
    "Forced_Incompletion_Rate_Pctl": 0.07,
    "YAC_Allowed_Pctl": 0.05,
    "Missed_Tackle_Rate_Pctl": 0.10,
    "Run_Stop_Rate_Over_Expected_Pctl": 0.13,
    "Explosive_Play_Rate_Allowed_Pctl": 0.12,
}

# ==================================================
# POSITION / ROLE LABELS
# ==================================================
#
# Generic DB is intentionally ambiguous and does not qualify by itself,
# because that would mix corners into the safety comparison pool.
# A specific S / FS / SS label in any available source is enough to qualify.
# ==================================================

DIRECT_SAFETY_LABELS = {
    "S",
    "FS",
    "SS",
    "SAF",
    "SAFETY",
}

DIRECT_CB_LABELS = {
    "CB",
    "LCB",
    "RCB",
    "NB",
    "NCB",
    "SCB",
    "SLOT",
}

LEGACY_SAFETY_LABELS = {
    "S",
    "FS",
    "SS",
    "SAF",
    "SAFETY",
    "DB",
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


def normalize_position_label(value):
    if pd.isna(value):
        return ""

    return str(value).upper().strip()


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
    Every current safety with a raw value receives a percentile.

    The qualifier applies only to the historical comparison pool.
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


def last_nonblank(series):
    clean = series.dropna()

    if clean.empty:
        return np.nan

    clean = clean.astype(str)
    clean = clean[
        clean.str.strip() != ""
    ]

    if clean.empty:
        return np.nan

    return clean.iloc[-1]


def classify_safety_role(row):
    """
    Current V1 safety classifier.

    A specific safety label in any available position source qualifies.
    Generic DB alone is not enough. A clear corner label excludes the player
    unless another source specifically identifies the player as a safety.
    """

    labels = {
        normalize_position_label(row.get("Snap_Position")),
        normalize_position_label(row.get("Roster_Position")),
        normalize_position_label(row.get("Depth_Chart_Position")),
    }

    labels.discard("")

    if labels.intersection(DIRECT_SAFETY_LABELS):
        return True

    if labels.intersection(DIRECT_CB_LABELS):
        return False

    return False


def safety_role_source(row):
    evidence = []

    snap_position = normalize_position_label(
        row.get("Snap_Position")
    )
    roster_position = normalize_position_label(
        row.get("Roster_Position")
    )
    depth_position = normalize_position_label(
        row.get("Depth_Chart_Position")
    )

    if snap_position in DIRECT_SAFETY_LABELS:
        evidence.append("SNAP")

    if roster_position in DIRECT_SAFETY_LABELS:
        evidence.append("ROSTER")

    if depth_position in DIRECT_SAFETY_LABELS:
        evidence.append("DEPTH")

    if not evidence:
        return "UNRESOLVED"

    return "+".join(evidence)


def calculate_missed_tackle_rate(
    df,
    missed_pct_column,
    missed_column,
    tackles_column
):
    if missed_pct_column is not None:
        return safe_numeric(
            df,
            missed_pct_column
        )

    if (
        missed_column is not None
        and tackles_column is not None
    ):
        missed = safe_numeric(
            df,
            missed_column
        )

        tackles = safe_numeric(
            df,
            tackles_column
        )

        return (
            missed
            /
            (
                tackles + missed
            ).replace(0, np.nan)
            *
            100
        )

    return pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
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

player_columns = [
    column
    for column in player_columns
    if column in players.columns
]

players = (
    players[player_columns]
    .drop_duplicates("gsis_id")
)

player_pfr_map = (
    players
    .dropna(subset=["pfr_id"])
    .drop_duplicates("pfr_id")
)


# ==================================================
# CURRENT ROSTERS — SECONDARY ROLE EVIDENCE
# ==================================================

if ROSTER_CACHE.exists():
    current_rosters = pd.read_parquet(
        ROSTER_CACHE
    )
else:
    current_rosters = nfl.load_rosters(
        [SEASON]
    ).to_pandas()

    current_rosters.to_parquet(
        ROSTER_CACHE,
        index=False
    )

if "season" in current_rosters.columns:
    current_rosters = current_rosters[
        current_rosters["season"] == SEASON
    ].copy()

roster_pfr_column = first_existing_column(
    current_rosters,
    [
        "pfr_id",
        "pfr_player_id"
    ]
)

roster_position_column = first_existing_column(
    current_rosters,
    [
        "position",
        "pos"
    ]
)

depth_position_column = first_existing_column(
    current_rosters,
    [
        "depth_chart_position",
        "depth_position"
    ]
)

roster_gsis_column = first_existing_column(
    current_rosters,
    [
        "gsis_id",
        "player_id"
    ]
)

roster_pff_column = first_existing_column(
    current_rosters,
    [
        "pff_id"
    ]
)

roster_espn_column = first_existing_column(
    current_rosters,
    [
        "espn_id"
    ]
)

roster_birth_column = first_existing_column(
    current_rosters,
    [
        "birth_date"
    ]
)

if roster_pfr_column is not None:
    current_roster_map = pd.DataFrame({
        "pfr_id": current_rosters[roster_pfr_column],
        "Roster_Position": (
            current_rosters[roster_position_column]
            if roster_position_column is not None
            else np.nan
        ),
        "Depth_Chart_Position": (
            current_rosters[depth_position_column]
            if depth_position_column is not None
            else np.nan
        ),
        "Roster_GSIS_ID": (
            current_rosters[roster_gsis_column]
            if roster_gsis_column is not None
            else np.nan
        ),
        "Roster_PFF_ID": (
            current_rosters[roster_pff_column]
            if roster_pff_column is not None
            else np.nan
        ),
        "Roster_ESPN_ID": (
            current_rosters[roster_espn_column]
            if roster_espn_column is not None
            else np.nan
        ),
        "Roster_Birth_Date": (
            current_rosters[roster_birth_column]
            if roster_birth_column is not None
            else np.nan
        ),
    })

    current_roster_map = (
        current_roster_map
        .dropna(subset=["pfr_id"])
        .drop_duplicates("pfr_id", keep="last")
    )
else:
    current_roster_map = pd.DataFrame({
        "pfr_id": pd.Series(dtype="object"),
        "Roster_Position": pd.Series(dtype="object"),
        "Depth_Chart_Position": pd.Series(dtype="object"),
        "Roster_GSIS_ID": pd.Series(dtype="object"),
        "Roster_PFF_ID": pd.Series(dtype="object"),
        "Roster_ESPN_ID": pd.Series(dtype="object"),
        "Roster_Birth_Date": pd.Series(dtype="object"),
    })


# ==================================================
# CURRENT SNAP COUNTS — BASE DEFENSIVE POPULATION
# ==================================================

if SNAP_CACHE.exists():
    snap_counts = pd.read_parquet(
        SNAP_CACHE
    )
else:
    snap_counts = nfl.load_snap_counts(
        [SEASON]
    ).to_pandas()

    snap_counts.to_parquet(
        SNAP_CACHE,
        index=False
    )

if "game_type" in snap_counts.columns:
    snap_counts = snap_counts[
        snap_counts["game_type"] == "REG"
    ].copy()

pfr_snap_column = first_existing_column(
    snap_counts,
    [
        "pfr_player_id",
        "pfr_id"
    ]
)

if pfr_snap_column is None:
    raise KeyError(
        "Snap counts do not contain a PFR player ID column."
    )

defensive_snap_column = first_existing_column(
    snap_counts,
    [
        "defense_snaps",
        "def_snaps",
        "defensive_snaps"
    ]
)

if defensive_snap_column is None:
    raise KeyError(
        "Snap counts do not contain a defensive snap column."
    )

snap_position_column = first_existing_column(
    snap_counts,
    [
        "position",
        "pos"
    ]
)

snap_name_column = first_existing_column(
    snap_counts,
    [
        "player",
        "player_name"
    ]
)

snap_team_column = first_existing_column(
    snap_counts,
    [
        "team",
        "club"
    ]
)

snap_counts["_Defensive_Snaps"] = safe_numeric(
    snap_counts,
    defensive_snap_column,
    default=0
).fillna(0)

current_def_snap_rows = snap_counts[
    snap_counts["_Defensive_Snaps"] > 0
].copy()

current_def_snap_rows = current_def_snap_rows.dropna(
    subset=[pfr_snap_column]
)

sort_columns = [
    column
    for column in [
        "week",
        "game_id"
    ]
    if column in current_def_snap_rows.columns
]

if sort_columns:
    current_def_snap_rows = (
        current_def_snap_rows
        .sort_values(sort_columns)
    )

current_def = (
    current_def_snap_rows
    .groupby(
        pfr_snap_column,
        dropna=False
    )
    .agg(
        Snap_Name=(
            snap_name_column,
            last_nonblank
        ) if snap_name_column is not None else (
            pfr_snap_column,
            "last"
        ),
        Team_2025=(
            snap_team_column,
            last_nonblank
        ) if snap_team_column is not None else (
            pfr_snap_column,
            "last"
        ),
        Snap_Position=(
            snap_position_column,
            last_nonblank
        ) if snap_position_column is not None else (
            pfr_snap_column,
            "last"
        ),
        Defensive_Snaps=(
            "_Defensive_Snaps",
            "sum"
        )
    )
    .reset_index()
    .rename(
        columns={
            pfr_snap_column: "pfr_id"
        }
    )
)

current_def = current_def.merge(
    current_roster_map,
    on="pfr_id",
    how="left"
)

current_def["Is_S"] = current_def.apply(
    classify_safety_role,
    axis=1
)

current_def["Safety_Role_Source"] = current_def.apply(
    safety_role_source,
    axis=1
)
current_def["Resolved_Secondary_Role"] = (
    current_def.apply(
        resolve_secondary_role,
        axis=1
    )
)

current_def["Is_S"] = (
    current_def["Resolved_Secondary_Role"]
    == "S"
)

current_s = current_def[
    current_def["Is_S"]
].copy()

current_s = current_s.merge(
    player_pfr_map,
    on="pfr_id",
    how="left",
    suffixes=("", "_player_map")
)

s_stats = pd.DataFrame({
    "Player_ID": (
        current_s["gsis_id"]
        .fillna(
            current_s["Roster_GSIS_ID"]
        )
    ),
    "Player": (
        current_s["display_name"]
        .fillna(
            current_s["Snap_Name"]
        )
    ),
    "Team_2025": current_s["Team_2025"],
    "Position": "S",
    "Snap_Position": current_s["Snap_Position"],
    "Roster_Position": current_s["Roster_Position"],
    "Depth_Chart_Position": current_s["Depth_Chart_Position"],
    "Safety_Role_Source": current_s["Safety_Role_Source"],
    "Defensive_Snaps": current_s["Defensive_Snaps"],
    "pfr_id": current_s["pfr_id"],
    "pff_id": (
        current_s["pff_id"]
        .fillna(
            current_s["Roster_PFF_ID"]
        )
    ),
    "espn_id": (
        current_s["espn_id"]
        .fillna(
            current_s["Roster_ESPN_ID"]
        )
    ),
    "birth_date": (
        current_s["birth_date"]
        .fillna(
            current_s["Roster_Birth_Date"]
        )
    ),
})


# ==================================================
# CURRENT PFR ADVANCED DEFENSE
# ==================================================
#
# Use any coverage-target / passer-rating / completion / YAC and missed-
# tackle fields exposed by the installed nflverse PFR schema. We do not
# fabricate coverage snaps, EPA/target, expected completion, run-stop over
# expectation, or explosive-play rate when those fields are unavailable.
# ==================================================

try:
    if PFR_DEF_CACHE.exists():
        pfr_def = pd.read_parquet(PFR_DEF_CACHE)
    else:
        pfr_def = nfl.load_pfr_advstats(
            seasons=[SEASON],
            stat_type="def",
            summary_level="season"
        ).to_pandas()
        pfr_def.to_parquet(PFR_DEF_CACHE, index=False)

    pfr_def_id_column = first_existing_column(
        pfr_def, ["pfr_id", "pfr_player_id"]
    )
    coverage_targets_column = first_existing_column(
        pfr_def, ["targets", "tgt", "def_targets", "coverage_targets", "pass_targets"]
    )
    qbr_targeted_column = first_existing_column(
        pfr_def, ["qbr_when_targeted", "passer_rating", "passer_rating_allowed",
                  "qb_rating", "def_passer_rating", "coverage_passer_rating"]
    )
    completions_column = first_existing_column(
        pfr_def, ["completions", "cmp", "def_completions",
                  "coverage_completions", "pass_completions_allowed"]
    )
    yac_column = first_existing_column(
        pfr_def, ["yac", "yards_after_catch", "yac_allowed",
                  "def_yac", "coverage_yac"]
    )
    missed_pct_column = first_existing_column(
        pfr_def, [
            "missed_tackle_pct", "missed_tackles_pct", "tkl_miss_pct",
            "def_missed_tackle_pct", "def_missed_tackles_pct"
        ]
    )
    missed_column = first_existing_column(
        pfr_def, [
            "missed_tackles", "missed_tackle", "tkl_miss",
            "tackles_missed", "def_missed_tackles", "def_tkl_miss"
        ]
    )
    tackles_column = first_existing_column(
        pfr_def, [
            "tackles", "combined_tackles", "tkl_comb",
            "def_tackles", "def_tackles_combined"
        ]
    )

    if pfr_def_id_column is None:
        raise KeyError(
            "PFR advanced defense does not contain a PFR player ID."
        )

    pfr_current = pd.DataFrame({
        "pfr_id": pfr_def[pfr_def_id_column],
        "Coverage_Targets": safe_numeric(pfr_def, coverage_targets_column),
        "QBR_When_Targeted": safe_numeric(pfr_def, qbr_targeted_column),
        "PFR_Completions_Allowed": safe_numeric(pfr_def, completions_column),
        "YAC_Allowed": safe_numeric(pfr_def, yac_column),
        "Missed_Tackle_Rate": calculate_missed_tackle_rate(
            pfr_def,
            missed_pct_column,
            missed_column,
            tackles_column
        ),
        "PFR_Missed_Tackles": safe_numeric(pfr_def, missed_column),
        "PFR_Tackles": safe_numeric(pfr_def, tackles_column),
    })

    pfr_current = pfr_current.drop_duplicates("pfr_id")
    s_stats = s_stats.merge(
        pfr_current,
        on="pfr_id",
        how="left"
    )

except Exception as error:
    print(f"PFR advanced defense failed: {error}")
    s_stats["Coverage_Targets"] = np.nan
    s_stats["QBR_When_Targeted"] = np.nan
    s_stats["PFR_Completions_Allowed"] = np.nan
    s_stats["YAC_Allowed"] = np.nan
    s_stats["Missed_Tackle_Rate"] = np.nan
    s_stats["PFR_Missed_Tackles"] = np.nan
    s_stats["PFR_Tackles"] = np.nan

# ==================================================
# FUTURE COVERAGE / RUN-DEFENSE METRICS
# ==================================================
#
# These are intentionally present now so charting data can be plugged into
# the same model later without rebuilding the safety architecture.
# ==================================================

s_stats["Coverage_Snaps"] = np.nan
s_stats["EPA_Per_Target"] = np.nan
s_stats["Yards_Per_Coverage_Snap"] = np.nan
s_stats["Coverage_Success_Rate"] = np.nan
s_stats["Targets_Per_Coverage_Snap"] = np.nan
s_stats["Completion_Pct_Over_Expected"] = np.nan
s_stats["Forced_Incompletion_Rate"] = np.nan
s_stats["Run_Stop_Rate_Over_Expected"] = np.nan
s_stats["Explosive_Play_Rate_Allowed"] = np.nan

# ==================================================
# QUALIFICATION / CONFIDENCE FLAGS
# ==================================================

s_stats["Target_Pct_Eligible"] = (
    s_stats["Coverage_Targets"].fillna(0) >= TARGET_QUALIFIER
)

s_stats["Coverage_Snap_Pct_Eligible"] = (
    s_stats["Coverage_Snaps"].fillna(0) >= COVERAGE_SNAP_QUALIFIER
)

s_stats["Snap_Pct_Eligible"] = (
    s_stats["Defensive_Snaps"] >= DEFENSIVE_SNAP_QUALIFIER
)

s_stats["Target_Stat_Confidence"] = np.where(
    s_stats["Coverage_Targets"].notna(),
    np.minimum(s_stats["Coverage_Targets"] / TARGET_QUALIFIER, 1.0),
    np.nan
)

s_stats["Coverage_Snap_Stat_Confidence"] = np.where(
    s_stats["Coverage_Snaps"].notna(),
    np.minimum(s_stats["Coverage_Snaps"] / COVERAGE_SNAP_QUALIFIER, 1.0),
    np.nan
)

s_stats["Snap_Stat_Confidence"] = np.minimum(
    s_stats["Defensive_Snaps"] / DEFENSIVE_SNAP_QUALIFIER,
    1.0
)


# ==================================================
# HISTORICAL ROSTERS — ROLE EVIDENCE
# ==================================================

if HIST_ROSTER_CACHE.exists():
    hist_rosters = pd.read_parquet(
        HIST_ROSTER_CACHE
    )
else:
    hist_rosters = nfl.load_rosters(
        HISTORICAL_SEASONS
    ).to_pandas()

    hist_rosters.to_parquet(
        HIST_ROSTER_CACHE,
        index=False
    )

hist_roster_pfr_column = first_existing_column(
    hist_rosters,
    [
        "pfr_id",
        "pfr_player_id"
    ]
)

hist_roster_position_column = first_existing_column(
    hist_rosters,
    [
        "position",
        "pos"
    ]
)

hist_depth_position_column = first_existing_column(
    hist_rosters,
    [
        "depth_chart_position",
        "depth_position"
    ]
)

if hist_roster_pfr_column is not None:
    hist_roster_map = pd.DataFrame({
        "season": (
            hist_rosters["season"]
            if "season" in hist_rosters.columns
            else np.nan
        ),
        "pfr_id": hist_rosters[
            hist_roster_pfr_column
        ],
        "Roster_Position": (
            hist_rosters[
                hist_roster_position_column
            ]
            if hist_roster_position_column is not None
            else np.nan
        ),
        "Depth_Chart_Position": (
            hist_rosters[
                hist_depth_position_column
            ]
            if hist_depth_position_column is not None
            else np.nan
        ),
    })

    hist_roster_map = (
        hist_roster_map
        .dropna(subset=["pfr_id"])
        .drop_duplicates(
            [
                "season",
                "pfr_id"
            ],
            keep="last"
        )
    )
else:
    hist_roster_map = pd.DataFrame({
        "season": pd.Series(dtype="float64"),
        "pfr_id": pd.Series(dtype="object"),
        "Roster_Position": pd.Series(dtype="object"),
        "Depth_Chart_Position": pd.Series(dtype="object"),
    })


# ==================================================
# HISTORICAL SNAP COUNTS — SAFETY COMPARISON POPULATION
# ==================================================

if HIST_SNAP_CACHE.exists():
    hist_snaps = pd.read_parquet(
        HIST_SNAP_CACHE
    )
else:
    hist_snaps = nfl.load_snap_counts(
        HISTORICAL_SEASONS
    ).to_pandas()

    hist_snaps.to_parquet(
        HIST_SNAP_CACHE,
        index=False
    )

if "game_type" in hist_snaps.columns:
    hist_snaps = hist_snaps[
        hist_snaps["game_type"] == "REG"
    ].copy()

hist_pfr_snap_column = first_existing_column(
    hist_snaps,
    [
        "pfr_player_id",
        "pfr_id"
    ]
)

hist_defensive_snap_column = first_existing_column(
    hist_snaps,
    [
        "defense_snaps",
        "def_snaps",
        "defensive_snaps"
    ]
)

hist_snap_position_column = first_existing_column(
    hist_snaps,
    [
        "position",
        "pos"
    ]
)

if (
    hist_pfr_snap_column is None
    or hist_defensive_snap_column is None
):
    raise KeyError(
        "Historical snap counts are missing required PFR ID or defensive snap columns."
    )

hist_snaps["_Defensive_Snaps"] = safe_numeric(
    hist_snaps,
    hist_defensive_snap_column,
    default=0
).fillna(0)

hist_snap_rows = hist_snaps[
    hist_snaps["_Defensive_Snaps"] > 0
].copy()

hist_snap_rows = hist_snap_rows.dropna(
    subset=[hist_pfr_snap_column]
)

hist_s_base = (
    hist_snap_rows
    .groupby(
        [
            "season",
            hist_pfr_snap_column
        ],
        dropna=False
    )
    .agg(
        Snap_Position=(
            hist_snap_position_column,
            last_nonblank
        ) if hist_snap_position_column is not None else (
            hist_pfr_snap_column,
            "last"
        ),
        Defensive_Snaps=(
            "_Defensive_Snaps",
            "sum"
        )
    )
    .reset_index()
    .rename(
        columns={
            hist_pfr_snap_column: "pfr_id"
        }
    )
)

hist_s_base = hist_s_base.merge(
    hist_roster_map,
    on=[
        "season",
        "pfr_id"
    ],
    how="left"
)

hist_s_base["Is_S"] = hist_s_base.apply(
    classify_safety_role,
    axis=1
)

hist_s = hist_s_base[
    hist_s_base["Is_S"]
].copy()


# ==================================================
# HISTORICAL PFR ADVANCED DEFENSE
# ==================================================

try:
    if HIST_PFR_DEF_CACHE.exists():
        hist_pfr_def = pd.read_parquet(HIST_PFR_DEF_CACHE)
    else:
        hist_pfr_def = nfl.load_pfr_advstats(
            seasons=HISTORICAL_SEASONS,
            stat_type="def",
            summary_level="season"
        ).to_pandas()
        hist_pfr_def.to_parquet(HIST_PFR_DEF_CACHE, index=False)

    hist_pfr_id_column = first_existing_column(
        hist_pfr_def, ["pfr_id", "pfr_player_id"]
    )
    hist_targets_column = first_existing_column(
        hist_pfr_def, ["targets", "tgt", "def_targets", "coverage_targets", "pass_targets"]
    )
    hist_qbr_column = first_existing_column(
        hist_pfr_def, ["qbr_when_targeted", "passer_rating", "passer_rating_allowed",
                       "qb_rating", "def_passer_rating", "coverage_passer_rating"]
    )
    hist_completions_column = first_existing_column(
        hist_pfr_def, ["completions", "cmp", "def_completions",
                       "coverage_completions", "pass_completions_allowed"]
    )
    hist_yac_column = first_existing_column(
        hist_pfr_def, ["yac", "yards_after_catch", "yac_allowed",
                       "def_yac", "coverage_yac"]
    )
    hist_missed_pct_column = first_existing_column(
        hist_pfr_def, [
            "missed_tackle_pct", "missed_tackles_pct", "tkl_miss_pct",
            "def_missed_tackle_pct", "def_missed_tackles_pct"
        ]
    )
    hist_missed_column = first_existing_column(
        hist_pfr_def, [
            "missed_tackles", "missed_tackle", "tkl_miss",
            "tackles_missed", "def_missed_tackles", "def_tkl_miss"
        ]
    )
    hist_tackles_column = first_existing_column(
        hist_pfr_def, [
            "tackles", "combined_tackles", "tkl_comb",
            "def_tackles", "def_tackles_combined"
        ]
    )

    if hist_pfr_id_column is None:
        raise KeyError(
            "Historical PFR advanced defense has no PFR player ID."
        )

    hist_pfr_clean = pd.DataFrame({
        "season": hist_pfr_def["season"],
        "pfr_id": hist_pfr_def[hist_pfr_id_column],
        "Coverage_Targets": safe_numeric(hist_pfr_def, hist_targets_column),
        "QBR_When_Targeted": safe_numeric(hist_pfr_def, hist_qbr_column),
        "PFR_Completions_Allowed": safe_numeric(hist_pfr_def, hist_completions_column),
        "YAC_Allowed": safe_numeric(hist_pfr_def, hist_yac_column),
        "Missed_Tackle_Rate": calculate_missed_tackle_rate(
            hist_pfr_def,
            hist_missed_pct_column,
            hist_missed_column,
            hist_tackles_column
        ),
        "PFR_Missed_Tackles": safe_numeric(hist_pfr_def, hist_missed_column),
        "PFR_Tackles": safe_numeric(hist_pfr_def, hist_tackles_column),
    })

    hist_pfr_clean = hist_pfr_clean.drop_duplicates(
        ["season", "pfr_id"]
    )

    hist_s = hist_s.merge(
        hist_pfr_clean,
        on=["season", "pfr_id"],
        how="left"
    )

except Exception as error:
    print(f"Historical PFR advanced defense failed: {error}")
    hist_s["Coverage_Targets"] = np.nan
    hist_s["QBR_When_Targeted"] = np.nan
    hist_s["PFR_Completions_Allowed"] = np.nan
    hist_s["YAC_Allowed"] = np.nan
    hist_s["Missed_Tackle_Rate"] = np.nan
    hist_s["PFR_Missed_Tackles"] = np.nan
    hist_s["PFR_Tackles"] = np.nan

# ==================================================
# FUTURE HISTORICAL COVERAGE / RUN-DEFENSE METRICS
# ==================================================

hist_s["Coverage_Snaps"] = np.nan
hist_s["EPA_Per_Target"] = np.nan
hist_s["Yards_Per_Coverage_Snap"] = np.nan
hist_s["Coverage_Success_Rate"] = np.nan
hist_s["Targets_Per_Coverage_Snap"] = np.nan
hist_s["Completion_Pct_Over_Expected"] = np.nan
hist_s["Forced_Incompletion_Rate"] = np.nan
hist_s["Run_Stop_Rate_Over_Expected"] = np.nan
hist_s["Explosive_Play_Rate_Allowed"] = np.nan

# ==================================================
# HISTORICAL QUALIFIERS
# ==================================================

hist_target_eligible = (
    hist_s["Coverage_Targets"].fillna(0) >= TARGET_QUALIFIER
)

hist_coverage_snap_eligible = (
    hist_s["Coverage_Snaps"].fillna(0) >= COVERAGE_SNAP_QUALIFIER
)

hist_snap_eligible = (
    hist_s["Defensive_Snaps"] >= DEFENSIVE_SNAP_QUALIFIER
)


# ==================================================
# FIVE-YEAR HISTORICAL PERCENTILES
# ==================================================

TARGET_METRICS = [
    ("QBR_When_Targeted", "QBR_When_Targeted_Pctl", False),
    ("EPA_Per_Target", "EPA_Per_Target_Pctl", False),
    ("Coverage_Success_Rate", "Coverage_Success_Rate_Pctl", True),
    ("Completion_Pct_Over_Expected", "Completion_Pct_Over_Expected_Pctl", False),
    ("Forced_Incompletion_Rate", "Forced_Incompletion_Rate_Pctl", True),
    ("YAC_Allowed", "YAC_Allowed_Pctl", False),
]

for value_column, percentile_column, higher_is_better in TARGET_METRICS:
    add_historical_percentile(
        s_stats,
        hist_s,
        value_column,
        percentile_column,
        hist_target_eligible,
        higher_is_better
    )

COVERAGE_SNAP_METRICS = [
    ("Yards_Per_Coverage_Snap", "Yards_Per_Coverage_Snap_Pctl", False),
    ("Targets_Per_Coverage_Snap", "Targets_Per_Coverage_Snap_Pctl", False),
    ("Explosive_Play_Rate_Allowed", "Explosive_Play_Rate_Allowed_Pctl", False),
]

for value_column, percentile_column, higher_is_better in COVERAGE_SNAP_METRICS:
    add_historical_percentile(
        s_stats,
        hist_s,
        value_column,
        percentile_column,
        hist_coverage_snap_eligible,
        higher_is_better
    )

SNAP_METRICS = [
    ("Missed_Tackle_Rate", "Missed_Tackle_Rate_Pctl", False),
    ("Run_Stop_Rate_Over_Expected", "Run_Stop_Rate_Over_Expected_Pctl", True),
]

for value_column, percentile_column, higher_is_better in SNAP_METRICS:
    add_historical_percentile(
        s_stats,
        hist_s,
        value_column,
        percentile_column,
        hist_snap_eligible,
        higher_is_better
    )

# ==================================================
# SAMPLE-SIZE REGRESSION
# ==================================================

TARGET_BASED_PERCENTILES = [
    "QBR_When_Targeted_Pctl",
    "EPA_Per_Target_Pctl",
    "Coverage_Success_Rate_Pctl",
    "Completion_Pct_Over_Expected_Pctl",
    "Forced_Incompletion_Rate_Pctl",
    "YAC_Allowed_Pctl",
]

for column in TARGET_BASED_PERCENTILES:
    s_stats[column + "_Raw"] = s_stats[column]
    s_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        s_stats["Target_Stat_Confidence"]
        *
        (
            s_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )

COVERAGE_SNAP_BASED_PERCENTILES = [
    "Yards_Per_Coverage_Snap_Pctl",
    "Targets_Per_Coverage_Snap_Pctl",
    "Explosive_Play_Rate_Allowed_Pctl",
]

for column in COVERAGE_SNAP_BASED_PERCENTILES:
    s_stats[column + "_Raw"] = s_stats[column]
    s_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        s_stats["Coverage_Snap_Stat_Confidence"]
        *
        (
            s_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )

SNAP_BASED_PERCENTILES = [
    "Missed_Tackle_Rate_Pctl",
    "Run_Stop_Rate_Over_Expected_Pctl",
]

for column in SNAP_BASED_PERCENTILES:
    s_stats[column + "_Raw"] = s_stats[column]
    s_stats[column] = (
        NEUTRAL_PERCENTILE
        +
        s_stats["Snap_Stat_Confidence"]
        *
        (
            s_stats[column]
            -
            NEUTRAL_PERCENTILE
        )
    )

# ==================================================
# ADVANCED SCORE / DATA COVERAGE
# ==================================================

s_stats["Advanced_Weighted_Pctl"] = (
    s_stats.apply(
        calculate_weighted_score,
        axis=1,
        weights=S_ADVANCED_WEIGHTS
    )
)

TOTAL_ADVANCED_WEIGHT = sum(
    S_ADVANCED_WEIGHTS.values()
)


def calculate_available_advanced_weight(row):
    available_weight = 0.0

    for metric, weight in S_ADVANCED_WEIGHTS.items():
        if pd.notna(row.get(metric)):
            available_weight += weight

    return available_weight


s_stats["Advanced_Available_Weight"] = (
    s_stats.apply(
        calculate_available_advanced_weight,
        axis=1
    )
)

s_stats["Advanced_Data_Coverage"] = (
    s_stats["Advanced_Available_Weight"]
    /
    TOTAL_ADVANCED_WEIGHT
)

s_stats["Advanced_Rating"] = (
    60
    +
    s_stats[
        "Advanced_Weighted_Pctl"
    ]
    *
    0.40
)


# ==================================================
# LOAD LEGACY DATABASE
# ==================================================
#
# Legacy defensive position labels are allowed to vary.
# A current role-classified safety can match a legacy S / FS / SS / DB-style row. We do not require the exact old position string to match.
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

legacy["Raw_Pos_Key"] = (
    legacy["Pos"]
    .astype(str)
    .str.upper()
    .str.strip()
)

legacy = legacy[
    legacy["Raw_Pos_Key"].isin(
        LEGACY_SAFETY_LABELS
    )
].copy()

legacy["Pos_Group_Key"] = "SAFETY"

legacy["Team_Key"] = (
    legacy["Team"]
    .astype(str)
    .str.upper()
    .str.strip()
)

s_stats["Name_Key"] = (
    s_stats["Player"]
    .apply(normalize_player_name)
)

s_stats["Pos_Group_Key"] = "SAFETY"

s_stats["Team_Key"] = (
    s_stats["Team_2025"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ==================================================
# PRIMARY LEGACY MATCH
# NAME + TEAM + CB GROUP
# ==================================================

legacy_primary = (
    legacy
    .drop_duplicates(
        [
            "Name_Key",
            "Pos_Group_Key",
            "Team_Key"
        ],
        keep=False
    )
    [
        [
            "Name_Key",
            "Pos_Group_Key",
            "Team_Key",
            "Age",
            "Madden",
            "PFF",
            "PFR"
        ]
    ]
)

s_stats = s_stats.merge(
    legacy_primary,
    on=[
        "Name_Key",
        "Pos_Group_Key",
        "Team_Key"
    ],
    how="left"
)

s_stats["Legacy_Match_Status"] = np.where(
    s_stats["Madden"].notna(),
    "NAME+TEAM (SAFETY GROUP)",
    "UNMATCHED"
)


# ==================================================
# FALLBACK LEGACY MATCH
# NAME + CB GROUP ONLY
# ==================================================

name_group_counts = (
    legacy
    .groupby(
        [
            "Name_Key",
            "Pos_Group_Key"
        ]
    )
    .size()
    .reset_index(
        name="Match_Count"
    )
)

legacy_fallback = legacy.merge(
    name_group_counts,
    on=[
        "Name_Key",
        "Pos_Group_Key"
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
        "Pos_Group_Key",
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

s_stats = s_stats.merge(
    legacy_fallback,
    on=[
        "Name_Key",
        "Pos_Group_Key"
    ],
    how="left"
)

needs_fallback = (
    s_stats["Madden"].isna()
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

    s_stats.loc[
        needs_fallback,
        column
    ] = s_stats.loc[
        needs_fallback,
        fallback_column
    ]

fallback_success = (
    needs_fallback
    &
    s_stats[
        "Fallback_Madden"
    ].notna()
)

s_stats.loc[
    fallback_success,
    "Legacy_Match_Status"
] = "NAME (SAFETY GROUP)"


# ==================================================
# AGE FALLBACK FROM NFLVERSE BIRTH DATE
# ==================================================

AGE_REFERENCE_DATE = pd.Timestamp(
    "2026-09-09"
)

birth_dates = pd.to_datetime(
    s_stats["birth_date"],
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

s_stats["Age_Source"] = np.select(
    [
        s_stats["Age"].notna(),
        calculated_age.notna()
    ],
    [
        "LEGACY",
        "NFLVERSE_BIRTH_DATE"
    ],
    default="MISSING"
)

s_stats["Age"] = (
    s_stats["Age"]
    .fillna(
        calculated_age
    )
)

s_stats = s_stats.drop(
    columns=[
        "Name_Key",
        "Pos_Group_Key",
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
        available_weight += (
            LEGACY_WEIGHTS["Madden"]
        )

    if pd.notna(row["PFF"]):
        weighted_total += (
            row["PFF"]
            *
            1.1
            *
            LEGACY_WEIGHTS["PFF"]
        )
        available_weight += (
            LEGACY_WEIGHTS["PFF"]
        )

    if pd.notna(row["PFR"]):
        weighted_total += (
            row["PFR"]
            *
            5.2
            *
            LEGACY_WEIGHTS["PFR"]
        )
        available_weight += (
            LEGACY_WEIGHTS["PFR"]
        )

    if available_weight == 0:
        return np.nan

    return (
        weighted_total
        /
        available_weight
    )


s_stats["Legacy_Rating"] = s_stats.apply(
    calculate_legacy_rating,
    axis=1
)

s_stats["Legacy_Fallback_Applied"] = (
    s_stats["Legacy_Rating"].isna()
)

s_stats["Legacy_Rating_Used"] = (
    s_stats["Legacy_Rating"]
    .fillna(MISSING_LEGACY_RATING)
)


# ==================================================
# FINAL ADVANCED / LEGACY CONFIDENCE
# ==================================================

s_stats["Opportunity_Confidence"] = np.minimum(
    s_stats["Defensive_Snaps"]
    /
    FULL_CONFIDENCE_DEFENSIVE_SNAPS,
    1.0
)

s_stats["Actual_Advanced_Weight"] = (
    BASE_ADVANCED_OVR_WEIGHT
    *
    s_stats[
        "Opportunity_Confidence"
    ]
    *
    s_stats[
        "Advanced_Data_Coverage"
    ]
)

s_stats["Actual_Legacy_Weight"] = (
    1
    -
    s_stats[
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
            advanced * advanced_weight
            +
            legacy * legacy_weight
        )

    if pd.notna(legacy):
        return legacy

    if pd.notna(advanced):
        return advanced

    return np.nan


s_stats["Pre_Bonus_Overall"] = s_stats.apply(
    calculate_pre_bonus_overall,
    axis=1
)

# No safety-specific historical record bonus in V1.
s_stats["Historical_Record_Bonus"] = 0.0

s_stats["Final_Overall"] = (
    s_stats["Pre_Bonus_Overall"]
    +
    s_stats["Historical_Record_Bonus"]
)


# ==================================================
# AGE-ADJUSTED VALUE
# ==================================================

s_stats["Age_Adjusted_Overall"] = (
    s_stats["Final_Overall"]
    *
    (
        (
            100
            +
            (
                S_PRIME_AGE
                -
                s_stats["Age"]
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
    "Player", "Player_ID", "Team_2025", "Position", "Age", "Age_Source",
    "Final_Overall", "Age_Adjusted_Overall",
    "Legacy_Rating", "Legacy_Rating_Used", "Legacy_Fallback_Applied",
    "Advanced_Rating", "Advanced_Weighted_Pctl", "Pre_Bonus_Overall",
    "Historical_Record_Bonus", "Legacy_Match_Status",

    "Snap_Position", "Roster_Position", "Depth_Chart_Position", "Safety_Role_Source",
    "Defensive_Snaps", "Coverage_Targets", "Coverage_Snaps",

    "Target_Pct_Eligible", "Coverage_Snap_Pct_Eligible", "Snap_Pct_Eligible",
    "Target_Stat_Confidence", "Coverage_Snap_Stat_Confidence",
    "Snap_Stat_Confidence", "Opportunity_Confidence",
    "Advanced_Data_Coverage", "Actual_Advanced_Weight", "Actual_Legacy_Weight",

    "QBR_When_Targeted", "QBR_When_Targeted_Pctl", "QBR_When_Targeted_Pctl_Raw",
    "EPA_Per_Target", "EPA_Per_Target_Pctl", "EPA_Per_Target_Pctl_Raw",
    "Yards_Per_Coverage_Snap", "Yards_Per_Coverage_Snap_Pctl", "Yards_Per_Coverage_Snap_Pctl_Raw",
    "Coverage_Success_Rate", "Coverage_Success_Rate_Pctl", "Coverage_Success_Rate_Pctl_Raw",
    "Targets_Per_Coverage_Snap", "Targets_Per_Coverage_Snap_Pctl", "Targets_Per_Coverage_Snap_Pctl_Raw",
    "Completion_Pct_Over_Expected", "Completion_Pct_Over_Expected_Pctl", "Completion_Pct_Over_Expected_Pctl_Raw",
    "Forced_Incompletion_Rate", "Forced_Incompletion_Rate_Pctl", "Forced_Incompletion_Rate_Pctl_Raw",
    "YAC_Allowed", "YAC_Allowed_Pctl", "YAC_Allowed_Pctl_Raw",
    "Missed_Tackle_Rate", "Missed_Tackle_Rate_Pctl", "Missed_Tackle_Rate_Pctl_Raw",
    "Run_Stop_Rate_Over_Expected", "Run_Stop_Rate_Over_Expected_Pctl", "Run_Stop_Rate_Over_Expected_Pctl_Raw",
    "Explosive_Play_Rate_Allowed", "Explosive_Play_Rate_Allowed_Pctl", "Explosive_Play_Rate_Allowed_Pctl_Raw",

    "PFR_Completions_Allowed", "PFR_Missed_Tackles", "PFR_Tackles",
    "Madden", "PFF", "PFR", "pfr_id", "pff_id", "espn_id",
]

for column in output_columns:
    if column not in s_stats.columns:
        s_stats[column] = np.nan

s_output = s_stats[output_columns].copy()

s_output = s_output.round({
    "Age": 1,
    "Final_Overall": 2, "Age_Adjusted_Overall": 2,
    "Legacy_Rating": 2, "Legacy_Rating_Used": 2,
    "Advanced_Rating": 2, "Advanced_Weighted_Pctl": 2,
    "Pre_Bonus_Overall": 2, "Historical_Record_Bonus": 2,
    "Defensive_Snaps": 1, "Coverage_Targets": 1, "Coverage_Snaps": 1,
    "Target_Stat_Confidence": 3, "Coverage_Snap_Stat_Confidence": 3,
    "Snap_Stat_Confidence": 3, "Opportunity_Confidence": 3,
    "Advanced_Data_Coverage": 3, "Actual_Advanced_Weight": 3,
    "Actual_Legacy_Weight": 3,
    "QBR_When_Targeted": 2, "QBR_When_Targeted_Pctl": 1, "QBR_When_Targeted_Pctl_Raw": 1,
    "EPA_Per_Target": 3, "EPA_Per_Target_Pctl": 1, "EPA_Per_Target_Pctl_Raw": 1,
    "Yards_Per_Coverage_Snap": 3, "Yards_Per_Coverage_Snap_Pctl": 1, "Yards_Per_Coverage_Snap_Pctl_Raw": 1,
    "Coverage_Success_Rate": 2, "Coverage_Success_Rate_Pctl": 1, "Coverage_Success_Rate_Pctl_Raw": 1,
    "Targets_Per_Coverage_Snap": 3, "Targets_Per_Coverage_Snap_Pctl": 1, "Targets_Per_Coverage_Snap_Pctl_Raw": 1,
    "Completion_Pct_Over_Expected": 2, "Completion_Pct_Over_Expected_Pctl": 1, "Completion_Pct_Over_Expected_Pctl_Raw": 1,
    "Forced_Incompletion_Rate": 2, "Forced_Incompletion_Rate_Pctl": 1, "Forced_Incompletion_Rate_Pctl_Raw": 1,
    "YAC_Allowed": 2, "YAC_Allowed_Pctl": 1, "YAC_Allowed_Pctl_Raw": 1,
    "Missed_Tackle_Rate": 2, "Missed_Tackle_Rate_Pctl": 1, "Missed_Tackle_Rate_Pctl_Raw": 1,
    "Run_Stop_Rate_Over_Expected": 2, "Run_Stop_Rate_Over_Expected_Pctl": 1, "Run_Stop_Rate_Over_Expected_Pctl_Raw": 1,
    "Explosive_Play_Rate_Allowed": 3, "Explosive_Play_Rate_Allowed_Pctl": 1, "Explosive_Play_Rate_Allowed_Pctl_Raw": 1,
    "PFR_Completions_Allowed": 1, "PFR_Missed_Tackles": 1, "PFR_Tackles": 1,
    "Madden": 1, "PFF": 1, "PFR": 1,
})

s_output = s_output.sort_values(
    "Final_Overall", ascending=False, na_position="last"
)

top_25 = s_output.head(25).copy()

missing_age_players = s_stats[s_stats["Age"].isna()][
    ["Player", "Player_ID", "Team_2025"]
]

if not missing_age_players.empty:
    print("\nPlayers still missing age:")
    print(missing_age_players.to_string(index=False))


# ==================================================
# SAVE PARQUET FOR FUTURE MASTER WORKBOOK
# ==================================================

s_output.to_parquet(
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

        s_output.to_excel(
            writer,
            sheet_name="All_Safeties",
            index=False
        )

        top_25.to_excel(
            writer,
            sheet_name="Top_25_Check",
            index=False
        )

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

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
                    column_cells[0]
                    .column_letter
                )

                worksheet.column_dimensions[
                    column_letter
                ].width = min(
                    max_length + 2,
                    30
                )

    print(
        f"Created {OUTPUT_FILE} "
        f"with {len(s_output)} CB players."
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
