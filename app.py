import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# ===============================================
# ⚡ LOAD PRE-PROCESSED DATA (FAST)
# ===============================================




st.write("🚀 App starting...")

url = "https://raw.githubusercontent.com/netodrip7/stats-merchant/main/final_data.parquet"

try:
    df = pd.read_parquet(url)
    st.write("✅ Data loaded successfully")
    st.write(df.shape)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ===============================================
# 🎨 PAGE CONFIG
# ===============================================

st.set_page_config(page_title="Stats Merchant", layout="wide")

st.markdown("""
<style>

body {
    background-color: #0b0f1a;
    color: #e6edf3;
}

.title {
    text-align: center;
    color: #2f81f7;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: 2px;
}

.tagline {
    text-align: center;
    color: #8b949e;
    font-size: 18px;
}

.smalltext {
    text-align: center;
    color: #6e7681;
    font-size: 14px;
}

.center-text {
    text-align: center;
    max-width: 700px;
    margin: auto;
    color: #c9d1d9;
    line-height: 1.6;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
}

/* Headers */
thead tr th {
    color: #2f81f7 !important;
    font-weight: 600 !important;
}

/* Rows */
tbody tr {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}

</style>
""", unsafe_allow_html=True)

# ===============================================
# 🧠 HEADER
# ===============================================

st.markdown('<div class="title">STATS MERCHANT</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">ball knowledge backed by stats</div>', unsafe_allow_html=True)
st.markdown('<div class="smalltext">here for the first time? please wait for 30–50 seconds. it’ll be so much quicker next time.</div>', unsafe_allow_html=True)

st.markdown('<div class="center-text">FPL managers, tap in.<br>This ain’t just another stats site.<br>This is your differential factory.<br>Get clean data, fixture swings, xG juice, and captaincy calls that actually hit.<br>No more picking your team on vibes only — we move with data now.<br><br><b>GET THAT RANK UP.</b></div>', unsafe_allow_html=True)

# ===============================================
# 🧹 PREP DATA
# ===============================================
df_latest = df.copy()

# ===============================================
# 🔍 SEARCH HELPER
# ===============================================

def normalize(text):
    text = unicodedata.normalize('NFKD', str(text))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]', '', text.lower())

df_latest["search_name"] = (
    df_latest["first_name_gw"].fillna('') + " " + df_latest["second_name_gw"].fillna('')
).apply(normalize)

df_latest["web_name_norm"] = df_latest["web_name_gw"].apply(normalize)

# ===============================================
# 🔮 PLAYER PREDICTION
# ===============================================

st.markdown("### 🔮 Enter a player’s name to see predicted points")
player_input = st.text_input("", key="pred")

if player_input:
    q = normalize(player_input)
    res = df_latest[
        df_latest["search_name"].str.contains(q) |
        df_latest["web_name_norm"].str.contains(q)
    ]

    if res.empty:
        st.warning("No player found")
    else:
        st.dataframe(
            res[["first_name_gw","second_name_gw","team_name_final","predicted_next_points"]]
            .sort_values("predicted_next_points", ascending=False)
        )

# ===============================================
# 🧠 RECOMMENDATION
# ===============================================

st.markdown("### 🧠 Enter a player’s name to get recommendation")
rec_input = st.text_input("", key="rec")

if rec_input:
    q = normalize(rec_input)
    player = df_latest[
        df_latest["search_name"].str.contains(q) |
        df_latest["web_name_norm"].str.contains(q)
    ]

    if not player.empty:
        p = player.iloc[0]
        st.success(f"""
        **{p['first_name_gw']} {p['second_name_gw']}**
        
        Predicted Points: {p['predicted_next_points']:.2f}  
        Value for Money: {p['value_for_money']:.2f}  
        Recommendation: **{p['recommendation']}**
        """)

# ===============================================
# 🔁 REPLACEMENTS
# ===============================================

st.markdown("### 🔁 Enter a player’s name for replacements")
rep_input = st.text_input("", key="rep")

if rep_input:
    q = normalize(rep_input)
    player = df_latest[
        df_latest["search_name"].str.contains(q) |
        df_latest["web_name_norm"].str.contains(q)
    ]

    if not player.empty:
        p = player.iloc[0]
        pos = p["position"]

        candidates = df_latest[df_latest["position"] == pos].copy()
        candidates = candidates[candidates["player_id"] != p["player_id"]]

        candidates["score"] = (
            candidates["predicted_next_points"] * 0.5 +
            candidates["value_for_money"] * 0.3 +
            candidates["form_gw"] * 0.2
        )

        top = candidates.sort_values("score", ascending=False).head(3)

        st.dataframe(top[[
            "first_name_gw","second_name_gw","predicted_next_points","value_for_money"
        ]])

# ===============================================
# 📊 ALWAYS VISIBLE TABLES
# ===============================================

st.markdown("## ⚡ Top Players by Position")
top5 = (
    df_latest.sort_values(['position','predicted_next_points'], ascending=[True,False])
    .groupby('position').head(5)
)
st.dataframe(top5[["first_name_gw","second_name_gw","position","predicted_next_points"]])

st.markdown("## 💸 Best Value Players")
vfm = (
    df_latest.sort_values(['position','value_for_money'], ascending=[True,False])
    .groupby('position').head(5)
)
st.dataframe(vfm[["first_name_gw","second_name_gw","value_for_money"]])

st.markdown("## Team Difficulty")

st.markdown("""
**What this shows:**  
Average difficulty of upcoming opponents.  
Lower value = easier fixtures  
Higher value = tougher fixtures
""")

team_diff = (
    df_latest.groupby("team_name_final")["opp_difficulty_proxy"]
    .mean()
    .reset_index()
    .rename(columns={
        "team_name_final": "Team",
        "opp_difficulty_proxy": "Difficulty Rating"
    })
    .sort_values("Difficulty Rating")
)

st.dataframe(team_diff, use_container_width=True)

st.markdown("## Team Predicted Points")

team_pts = (
    df_latest.groupby("team_name_final")["predicted_next_points"]
    .mean()
    .reset_index()
    .rename(columns={
        "team_name_final": "Team",
        "predicted_next_points": "Avg Predicted Points"
    })
    .sort_values("Avg Predicted Points", ascending=False)
)

st.dataframe(team_pts, use_container_width=True)
import requests
import pandas as pd
import streamlit as st

st.subheader("Upcoming Fixtures")

try:
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)

    if response.status_code == 200:
        fixtures = pd.DataFrame(response.json())

        fixtures = fixtures[fixtures['finished'] == False]
        next_gw = fixtures['event'].min()
        gw_fixtures = fixtures[fixtures['event'] == next_gw]

        team_map = {
            1: "Arsenal", 2: "Aston Villa", 3: "Bournemouth",
            4: "Brentford", 5: "Brighton", 6: "Chelsea",
            7: "Crystal Palace", 8: "Everton", 9: "Fulham",
            10: "Ipswich Town", 11: "Leicester City",
            12: "Liverpool", 13: "Manchester City",
            14: "Manchester United", 15: "Newcastle United",
            16: "Nottingham Forest", 17: "Southampton",
            18: "Tottenham Hotspur", 19: "West Ham United",
            20: "Wolverhampton Wanderers"
        }

        gw_fixtures["Home Team"] = gw_fixtures["team_h"].map(team_map)
        gw_fixtures["Away Team"] = gw_fixtures["team_a"].map(team_map)

        fixtures_display = gw_fixtures[["Home Team", "Away Team"]]

        st.write(f"Gameweek {int(next_gw)}")
        st.dataframe(fixtures_display, use_container_width=True)

    else:
        st.warning("Fixtures unavailable")

except:
    st.warning("Fixtures unavailable")
