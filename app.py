import streamlit as st
import pandas as pd
import re
import unicodedata

# ===============================================
# ⚡ LOAD DATA FAST
# ===============================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/final_data.parquet"
    return pd.read_parquet(url)

df_clean = load_data()

# ===============================================
# 🎨 UI DESIGN
# ===============================================
st.set_page_config(layout="wide")

st.markdown("""
<style>
body {background-color: #0a0f2c; color: white;}
.big-title {text-align: center; font-size: 60px; font-weight: 800; color: #4da6ff;}
.tagline {text-align: center; font-size: 22px; color: #7ec8ff;}
.small-note {text-align: center; font-size: 16px; color: #a3cfff;}
.desc {text-align: center; font-size: 20px; color: #cce6ff; line-height: 1.3;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">STATS MERCHANT</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">ball knowledge backed by stats</div>', unsafe_allow_html=True)
st.markdown('<div class="small-note">first time? wait 30-50 sec — next time is instant.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="desc">
FPL managers, tap in.<br>
This ain’t just another stats site.<br>
This is your differential factory.<br>
We move with data now.
</div>
""", unsafe_allow_html=True)

# ===============================================
# 🔍 NORMALIZATION FUNCTION (UNCHANGED LOGIC)
# ===============================================
def normalize_text(text):
    text = unicodedata.normalize('NFKD', str(text))
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text.lower()

# ===============================================
# 🔮 PLAYER PREDICTION
# ===============================================
st.markdown("### 🔮 Enter player name for predicted points")
name = st.text_input("")

if name:
    name = normalize_text(name)
    df_clean['search'] = df_clean['web_name_gw'].apply(normalize_text)

    results = df_clean[df_clean['search'].str.contains(name, na=False)]
    st.dataframe(results.sort_values('predicted_next_points', ascending=False).head(10))

# ===============================================
# 🧠 RECOMMENDATION
# ===============================================
st.markdown("### 🧠 Start / Bench / Sell")

rec = st.text_input(" ", key="rec")

if rec:
    rec = normalize_text(rec)
    df_clean['search'] = df_clean['web_name_gw'].apply(normalize_text)

    player = df_clean[df_clean['search'].str.contains(rec, na=False)]

    if not player.empty:
        p = player.iloc[0]
        st.write(f"Recommendation: {p['recommendation']}")

# ===============================================
# 🔁 REPLACEMENTS
# ===============================================
st.markdown("### 🔁 Replacement Suggestions")

rep = st.text_input("  ", key="rep")

if rep:
    rep = normalize_text(rep)
    df_clean['search'] = df_clean['web_name_gw'].apply(normalize_text)

    player = df_clean[df_clean['search'].str.contains(rep, na=False)]

    if not player.empty:
        p = player.iloc[0]
        pos = p['position']

        candidates = df_clean[df_clean['position'] == pos]
        candidates = candidates[candidates['player_id'] != p['player_id']]

        candidates['score'] = (
            candidates['predicted_next_points'] * 0.5 +
            candidates['value_for_money'] * 0.3
        )

        st.dataframe(candidates.sort_values('score', ascending=False).head(5))

# ===============================================
# 📊 DASHBOARD TABLES
# ===============================================
df_latest = df_clean.sort_values(['player_id','gameweek']).drop_duplicates('player_id', keep='last')

st.markdown("## ⚡ Top 5 by Predicted Points")
st.dataframe(
    df_latest.sort_values('predicted_next_points', ascending=False)
    .groupby('position').head(5)
)

st.markdown("## 💸 Value for Money")
st.dataframe(
    df_latest.sort_values('value_for_money', ascending=False)
    .groupby('position').head(5)
)

st.markdown("## 🧱 Team Difficulty")
st.dataframe(
    df_latest.groupby('team_name_final')['opp_difficulty_proxy']
    .mean().sort_values()
)

st.markdown("## ⚽ Team Predicted Points")
st.dataframe(
    df_latest.groupby('team_name_final')['predicted_next_points']
    .sum().sort_values(ascending=False)
)


