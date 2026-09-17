import nflreadpy as nfl
import pandas as pd

# ==================================================
# SETTINGS
# ==================================================

SEASON = 2025
MIN_QB_DROPBACKS = 200
OUTPUT_FILE = "Josh_Allen_2025_Test.xlsx"

# Temporary values/weights — replace as database develops
MADDEN_RATING = 95.0
PFF_RATING = 92.0
PFR_RATING = 90.0

WEIGHTS = {
    "Madden": 0.20,
    "PFF": 0.30,
    "PFR": 0.20,
    "EPA": 0.30
}

# ==================================================
# LOAD NFLVERSE DATA
# ==================================================

pbp = nfl.load_pbp([SEASON]).to_pandas()

# ==================================================
# QB EPA
# ==================================================

qb_plays = pbp[
    (pbp["qb_dropback"] == 1) &
    (pbp["passer_player_id"].notna()) &
    (pbp["epa"].notna())
]

qb_stats = (
    qb_plays
    .groupby(["passer_player_id", "passer_player_name"])
    .agg(
        Dropbacks=("epa", "count"),
        EPA_per_Dropback=("epa", "mean")
    )
    .reset_index()
)

# Only compare QBs with meaningful playing time
qb_stats = qb_stats[qb_stats["Dropbacks"] >= MIN_QB_DROPBACKS].copy()

# Normalize EPA against qualifying QBs
qb_stats["EPA_Percentile"] = (
    qb_stats["EPA_per_Dropback"].rank(pct=True) * 100
)

# Temporary 60–100 conversion
qb_stats["EPA_Rating"] = 60 + (qb_stats["EPA_Percentile"] * 0.40)

# ==================================================
# JOSH ALLEN TEST PLAYER
# ==================================================

allen = qb_stats[
    qb_stats["passer_player_id"] == "00-0034857"
].iloc[0]

overall = (
    MADDEN_RATING * WEIGHTS["Madden"] +
    PFF_RATING * WEIGHTS["PFF"] +
    PFR_RATING * WEIGHTS["PFR"] +
    allen["EPA_Rating"] * WEIGHTS["EPA"]
)

# ==================================================
# FINAL OUTPUT
# ==================================================

output = pd.DataFrame([{
    "Player_ID": allen["passer_player_id"],
    "Player": "Josh Allen",
    "Season": SEASON,
    "Team": "BUF",
    "Position": "QB",
    "Dropbacks": int(allen["Dropbacks"]),
    "EPA_per_Dropback": round(allen["EPA_per_Dropback"], 3),
    "EPA_Percentile": round(allen["EPA_Percentile"], 1),
    "EPA_Rating": round(allen["EPA_Rating"], 1),
    "Madden": MADDEN_RATING,
    "PFF": PFF_RATING,
    "PFR": PFR_RATING,
    "Overall": round(overall, 1)
}])

output.to_excel(OUTPUT_FILE, index=False)

print(f"Created {OUTPUT_FILE}")