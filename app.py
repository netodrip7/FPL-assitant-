# ============================================================
# ⚽ FPL Assistant Dashboard (Final Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib  # needed for pandas Styler color maps

# ============================================================
# 🎯 PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="FPL Assistant 2025/26", layout="wide")
st.title("⚽ FPL Assistant Dashboard - 2025/26 Season")
st.caption("Auto-refreshes weekly using live FPL-Elo-Insights data")

# ============================================================
# 📦 LOAD DATA
# ============================================================
base_url = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data/2025-2026"
by_tournament = f"{base_url}/By%20Tournament/Premier%20League"
urls = {
    "teams": f"{base_url}/teams.csv",
    "players": f"{by_tournament}/GW11/players.csv",
    "playerstats": f"{base_url}/playerstats.csv",
}

@st.cache_data(ttl=3600)
def load_data():
    teams = pd.read_csv(urls["teams"])
    players = pd.read_csv(urls["players"])
    playerstats = pd.read_csv(urls["playerstats"])

    gw_data = []
    for i in range(1, 39):
        try:
            df = pd.read_csv(f"{by_tournament}/GW{i}/player_gameweek_stats.csv")
            df["gameweek"] = i
            gw_data.append(df)
        except Exception:
            pass

    gw_df = pd.concat(gw_data, ignore_index=True)
    gw_df = gw_df.rename(columns={"id": "player_id"})
    playerstats = playerstats.rename(columns={"id": "player_id"})
    teams = teams.rename(columns={"id": "team_id"})

    merged = (
        gw_df
        .merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
        .merge(players, on="player_id", how="left")
        .merge(teams, left_on="team_code", right_on="team_id", how="left")
    )

    return merged, teams, players

df_clean, teams, players = load_data()

# ============================================================
# 🧹 FEATURE ENGINEERING
# ============================================================
df_clean = df_clean.sort_values(["player_id", "gameweek"])
df_clean["next_gw_points"] = df_clean.groupby("player_id")["event_points_gw"].shift(-1)

# Rolling stats (form indicators)
for col in ["event_points_gw", "goals_scored_gw", "assists_gw", "expected_goals_gw", "expected_assists_gw"]:
    if col in df_clean.columns:
        df_clean[f"{col}_roll3"] = (
            df_clean.groupby("player_id")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

# Team form & difficulty
team_form = (
    df_clean.groupby(["team_id", "gameweek"])["event_points_gw"]
    .mean()
    .reset_index(name="team_avg_points")
)
df_clean = df_clean.merge(team_form, on=["team_id", "gameweek"], how="left")

if "strength_defence_home" in teams.columns and "strength_attack_home" in teams.columns:
    teams["difficulty"] = teams[["strength_defence_home", "strength_attack_home"]].mean(axis=1)
else:
    teams["difficulty"] = np.random.uniform(2, 4, len(teams))

df_clean = df_clean.dropna(subset=["next_gw_points"])

# ============================================================
# ⚙️ MODEL TRAINING
# ============================================================
feature_cols = [c for c in df_clean.columns if c.endswith("_gw") or c.endswith("_roll3") or "team" in c]
X = df_clean[feature_cols].select_dtypes(include=[np.number]).fillna(0)
y = df_clean["next_gw_points"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)
df_clean["predicted_points"] = model.predict(X)

# ============================================================
# 💸 VALUE FOR MONEY (VFM)
# ============================================================
if "now_cost" in df_clean.columns:
    df_clean["vfm"] = df_clean["predicted_points"] / (df_clean["now_cost"] / 10)
else:
    df_clean["vfm"] = np.random.uniform(0, 10, len(df_clean))

# ============================================================
# 🏟️ TEAM PERFORMANCE (20 TEAMS)
# ============================================================
team_points = (
    df_clean.groupby(["team_id"])["event_points_gw"]
    .mean()
    .reset_index(name="avg_points")
    .merge(teams[["team_id", "name", "difficulty"]], on="team_id", how="left")
    .sort_values("avg_points", ascending=False)
)

# ============================================================
# ⚽ TOP 10 & TOP 5
# ============================================================
top_players = (
    df_clean.groupby(["player_id", "web_name"])["predicted_points"]
    .mean()
    .reset_index()
    .sort_values("predicted_points", ascending=False)
    .head(10)
)

top_vfm = (
    df_clean.groupby(["player_id", "web_name"])["vfm"]
    .mean()
    .reset_index()
    .sort_values("vfm", ascending=False)
    .head(5)
)

# ============================================================
# 📊 TOP PLAYERS PER POSITION
# ============================================================
pos_cols = [c for c in df_clean.columns if "element_type" in c.lower() or "position" in c.lower()]
pos_col = pos_cols[0] if pos_cols else None

def top_by_position(metric, n=5):
    if pos_col:
        return (
            df_clean.groupby([pos_col, "web_name"])[metric]
            .mean()
            .reset_index()
            .sort_values([pos_col, metric], ascending=[True, False])
            .groupby(pos_col)
            .head(n)
        )
    return pd.DataFrame()

top_predicted_by_pos = top_by_position("predicted_points")
top_vfm_by_pos = top_by_position("vfm")

# ============================================================
# 🖥️ DASHBOARD DISPLAY
# ============================================================
st.subheader("🏟️ Team Performance (All 20 Teams)")
st.dataframe(team_points.style.background_gradient("Greens").format({"avg_points": "{:.2f}", "difficulty": "{:.2f}"}))

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔥 Top 10 Players (Predicted Points)")
    st.dataframe(top_players.style.background_gradient("Blues").format({"predicted_points": "{:.2f}"}))
with col2:
    st.subheader("💸 Top 5 Players by Value-for-Money (Overall)")
    st.dataframe(top_vfm.style.background_gradient("Purples").format({"vfm": "{:.2f}"}))

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.subheader("⚔️ Top 5 Players per Position (Predicted Points)")
    st.dataframe(top_predicted_by_pos.style.background_gradient("Oranges").format({"predicted_points": "{:.2f}"}))
with col4:
    st.subheader("💹 Top 5 Players per Position (Value-for-Money)")
    st.dataframe(top_vfm_by_pos.style.background_gradient("Greens").format({"vfm": "{:.2f}"}))

# ============================================================
# 🧍 PLAYER ANALYSIS
# ============================================================
st.markdown("---")
st.subheader("🔍 Player Performance Insight")

player_name = st.text_input("Enter Player Name (case insensitive):")

if player_name:
    player_match = df_clean[df_clean["web_name"].str.contains(player_name, case=False, na=False)]

    if not player_match.empty:
        avg_pred = float(player_match["predicted_points"].mean())
        cost = float(player_match["now_cost"].mean() / 10) if "now_cost" in player_match.columns else 0
        st.success(f"**{player_name.title()}** predicted next GW points: **{avg_pred:.2f}**, cost: **£{cost:.1f}m**")

        # Recommendation logic
        if avg_pred >= 7:
            rec = "Start"
        elif avg_pred >= 5:
            rec = "Buy"
        else:
            rec = "Sell"
        st.info(f"🧭 Recommended Action: **{rec}**")

        # --- REPLACEMENTS ---
        if pos_col and pos_col in df_clean.columns:
            same_pos = df_clean[df_clean[pos_col] == player_match[pos_col].iloc[0]].copy()
        else:
            same_pos = df_clean.copy()

        same_pos = same_pos[same_pos["web_name"].str.lower() != player_name.lower()]

        replacements_points = (
            same_pos.groupby("web_name")["predicted_points"]
            .mean()
            .reset_index()
            .sort_values("predicted_points", ascending=False)
            .head(5)
        )

        replacements_vfm = (
            same_pos.groupby("web_name")["vfm"]
            .mean()
            .reset_index()
            .sort_values("vfm", ascending=False)
            .head(5)
        )

        replacements_cost = (
            same_pos.groupby(["web_name", "now_cost"])["predicted_points"]
            .mean()
            .reset_index()
            .assign(cost_diff=lambda x: abs(x["now_cost"] - cost * 10))
            .sort_values("cost_diff", ascending=True)
            .head(5)
            .drop(columns="cost_diff")
        )

        st.markdown(f"### 🔁 Replacements for {player_name.title()} (Top 5 by Points)")
        st.dataframe(replacements_points.style.background_gradient("Blues").format({"predicted_points": "{:.2f}"}))

        st.markdown(f"### 💸 Replacements for {player_name.title()} (Top 5 by VFM)")
        st.dataframe(replacements_vfm.style.background_gradient("Purples").format({"vfm": "{:.2f}"}))

        st.markdown(f"### 💷 Replacements for {player_name.title()} (Closest Cost Match)")
        st.dataframe(replacements_cost.style.background_gradient("Greens").format({"predicted_points": "{:.2f}"}))
    else:
        st.warning("⚠️ Player not found. Please check spelling.")

# ============================================================
# ⭐ TOP 5 OVERALL
# ============================================================
st.markdown("---")
st.subheader("⭐ Top 5 Predicted Performers Overall")
top5 = top_players.head(5)
st.dataframe(top5.style.background_gradient("Oranges").format({"predicted_points": "{:.2f}"}))

st.caption("✅ Model retrains and updates each refresh using the latest FPL-Elo-Insights data.")



