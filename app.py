import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# ===============================================
# ⚡ LOAD PRE-PROCESSED DATA (FAST)
# ===============================================

url = "https://raw.githubusercontent.com/YOUR_USERNAME/stats-merchant/main/final_data.parquet"
df = pd.read_parquet(url)

# ===============================================
# 🎨 PAGE CONFIG
# ===============================================

st.set_page_config(page_title="Stats Merchant", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0a0f1c;
        color: white;
    }
    .title {
        text-align: center;
        color: #1f6feb;
        font-size: 60px;
        font-weight: 900;
    }
    .tagline {
        text-align: center;
        color: #4ea1ff;
        font-size: 20px;
    }
    .smalltext {
        text-align: center;
        color: #8fbaff;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================================
# 🧠 HEADER
# ===============================================

st.markdown('<div class="title">STATS MERCHANT</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">ball knowledge backed by stats</div>', unsafe_allow_html=True)
st.markdown('<div class="smalltext">here for the first time? please wait for 30–50 seconds. it’ll be so much quicker next time.</div>', unsafe_allow_html=True)

st.markdown("""
FPL managers, tap in.  
This ain’t just another stats site.  
This is your differential factory.  
Get clean data, fixture swings, xG juice, and captaincy calls that actually hit.  
No more picking your team on vibes only — we move with data now.  
GET THAT RANK UP.
""")

# ===============================================
# 🧹 PREP DATA
# ===============================================

df_latest = df.sort_values(['player_id', 'gameweek'], ascending=[True, False]).drop_duplicates('player_id')

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

st.markdown("## 🧱 Team Difficulty")
team_diff = (
    df_latest.groupby("team_name_final")["opp_difficulty_proxy"]
    .mean().sort_values()
)
st.dataframe(team_diff)

st.markdown("## ⚽ Team Predicted Points")
team_pts = (
    df_latest.groupby("team_name_final")["predicted_next_points"]
    .sum().sort_values(ascending=False)
)
st.dataframe(team_pts)

