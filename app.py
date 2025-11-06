import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib  # required for pandas Styler gradient

# ------------------------------------------------------------
# 🎯 PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="FPL Assistant - 2025/26", layout="wide")
st.title("⚽ FPL Assistant - 2025/26 Season")
st.caption("Auto-updated each refresh using live data from FPL-Elo-Insights")

# ------------------------------------------------------------
# 📦 LOAD DATA
# ------------------------------------------------------------
base_url = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data/2025-2026"
by_tournament = f"{base_url}/By%20Tournament/Premier%20League"
urls = {
    "teams": f"{base_url}/teams.csv",
    "players": f"{by_tournament}/GW11/players.csv",
    "playerstats": f"{base_url}/playerstats.csv",
    "gameweek_summaries": f"{base_url}/gameweek_summaries.csv",
}

@st.cache_data(ttl=3600)
def load_data():
    teams = pd.read_csv(urls["teams"])
    playerstats = pd.read_csv(urls["playerstats"])
    gameweek_summaries = pd.read_csv(urls["gameweek_summaries"])
    players = pd.read_csv(urls["players"])

    gw_data = []
    for i in range(1, 39):
        url = f"{by_tournament}/GW{i}/player_gameweek_stats.csv"
        try:
            df = pd.read_csv(url)
            df["gameweek"] = i
            gw_data.append(df)
        except Exception:
            pass
    player_gw_stats = pd.concat(gw_data, ignore_index=True)

    player_gw_stats = player_gw_stats.rename(columns={"id": "player_id"})
    playerstats = playerstats.rename(columns={"id": "player_id"})
    teams = teams.rename(columns={"id": "team_id"})

    merged = (
        player_gw_stats.merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
        .merge(players, on="player_id", how="left")
        .merge(teams, left_on="team_code", right_on="team_id", how="left", suffixes=("", "_team"))
    )
    return merged, teams, players

df_clean, teams, players = load_data()

# ------------------------------------------------------------
# 🧠 FEATURE ENGINEERING
# ------------------------------------------------------------
df_clean = df_clean.sort_values(["player_id", "gameweek"])
df_clean["next_gw_points"] = df_clean.groupby("player_id")["event_points_gw"].shift(-1)

for col in ["event_points_gw", "goals_scored_gw", "assists_gw", "expected_goals_gw", "expected_assists_gw"]:
    if col in df_clean.columns:
        df_clean[f"{col}_roll3"] = (
            df_clean.groupby("player_id")[col]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

team_form = (
    df_clean.groupby(["team_id", "gameweek"])["event_points_gw"]
    .mean()
    .reset_index(name="team_avg_points")
)
df_clean = df_clean.merge(team_form, on=["team_id", "gameweek"], how="left")

if "strength_defence_home" in teams.columns and "strength_defence_away" in teams.columns:
    df_clean["team_strength_avg"] = df_clean[["strength_defence_home", "strength_defence_away"]].mean(axis=1)
else:
    df_clean["team_strength_avg"] = np.nan

df_clean = df_clean.dropna(subset=["next_gw_points"])

# ------------------------------------------------------------
# ⚙️ MODEL TRAINING (XGBoost)
# ------------------------------------------------------------
feature_cols = [c for c in df_clean.columns if c.endswith("_gw") or c.endswith("_roll3") or "team" in c]
X = df_clean[feature_cols].select_dtypes(include=[np.number]).fillna(0)
y = df_clean["next_gw_points"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)
df_clean["predicted_points"] = model.predict(X)

# ------------------------------------------------------------
# 🏆 RANKINGS
# ------------------------------------------------------------
possible_team_cols = [c for c in df_clean.columns if "team" in c.lower() and "id" not in c.lower()]
team_col = possible_team_cols[0] if possible_team_cols else "team_id"

top_teams = (
    df_clean.groupby(team_col)["team_avg_points"]
    .mean()
    .reset_index()
    .sort_values("team_avg_points", ascending=False)
    .head(10)
)

top_players = (
    df_clean.groupby(["player_id", "web_name"])["predicted_points"]
    .mean()
    .reset_index()
    .sort_values("predicted_points", ascending=False)
    .head(10)
)

# ------------------------------------------------------------
# 🖥️ DASHBOARD DISPLAY
# ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("🏟️ Top 10 Teams (Avg Points per GW)")
    st.dataframe(
        top_teams.style.background_gradient("Greens").format({"team_avg_points": "{:.2f}"})
    )

with col2:
    st.subheader("🔥 Top 10 Players (Predicted Points)")
    st.dataframe(
        top_players.style.background_gradient("Blues").format({"predicted_points": "{:.2f}"})
    )

# ------------------------------------------------------------
# 🧍 PLAYER INPUT & RECOMMENDATION
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Player Performance Insight")

player_name = st.text_input("Enter Player Name (case insensitive):")

if player_name:
    player_match = df_clean[df_clean["web_name"].str.contains(player_name, case=False, na=False)]

    if not player_match.empty:
        avg_pred = float(player_match["predicted_points"].mean())
        st.success(f"**{player_name.title()}** predicted next GW points: **{avg_pred:.2f}**")

        # Recommendation logic
        if avg_pred >= 7:
            rec = "Start"
        elif avg_pred >= 5:
            rec = "Buy"
        else:
            rec = "Sell"
        st.info(f"Recommended Action: **{rec}**")

        # Replacements
        pos_cols = [c for c in df_clean.columns if "element_type" in c.lower() or "position" in c.lower()]
        pos_col = pos_cols[0] if pos_cols else None

        if pos_col and pos_col in df_clean.columns:
            same_pos = df_clean[df_clean[pos_col] == player_match[pos_col].iloc[0]]
        else:
            same_pos = df_clean.copy()

        replacements = (
            same_pos.groupby("web_name")["predicted_points"]
            .mean()
            .reset_index()
            .sort_values("predicted_points", ascending=False)
            .head(5)
        )

        st.markdown(f"### 🧩 Suggested Replacements for {player_name.title()}")
        st.dataframe(
            replacements.style.background_gradient("Purples").format({"predicted_points": "{:.2f}"})
        )

    else:
        st.warning("⚠️ Player not found. Please check spelling.")

# ------------------------------------------------------------
# ⭐ TOP 5 PLAYERS OVERALL
# ------------------------------------------------------------
st.markdown("---")
st.subheader("⭐ Top 5 Predicted Performers")
top5 = top_players.head(5)
st.dataframe(top5.style.background_gradient("Oranges").format({"predicted_points": "{:.2f}"}))

st.caption("✅ Model retrains and updates each refresh using the latest GitHub data.")
