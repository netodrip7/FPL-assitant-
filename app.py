import streamlit as st
from pipeline import run_pipeline
from utils import search_player, get_replacements, captain_pick

st.set_page_config(layout="wide")

# --------- STYLE ----------
st.markdown("""
<style>
body {background:#0a192f;color:white;}
.title {text-align:center;font-size:64px;font-weight:bold;color:#5fa8ff;}
.tag {text-align:center;font-size:20px;color:#9fb3c8;}
.small {text-align:center;color:#6b85a6;}
.desc {text-align:center;color:#cbd5e1;font-size:18px;}
</style>
""", unsafe_allow_html=True)

# --------- HEADER ----------
st.markdown('<div class="title">STATS MERCHANT</div>', unsafe_allow_html=True)
st.markdown('<div class="tag">ball knowledge backed by stats</div>', unsafe_allow_html=True)
st.markdown('<div class="small">here for the first time? please wait for 30-50 seconds. it’ll be so much quicker the next time i promise.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="desc">
FPL managers, tap in. This is your differential factory.<br>

Get clean data, fixture swings, xG juice, and captaincy calls that actually hit.<br>

No more picking your team on vibes only — we move with data now.<br><br>

GET THAT RANK UP.
</div>
""", unsafe_allow_html=True)

# --------- LOAD ----------
@st.cache_data(ttl=300)
def load():
    return run_pipeline()

df, df_latest = load()

# --------- INPUT ----------
name = st.text_input("Enter player name")

if name:
    st.write("### Predicted Points")
    st.dataframe(search_player(df_latest, name))

    st.write("### Replacements")
    st.dataframe(get_replacements(df_latest, name))

# --------- CAPTAIN ----------
st.write("### Best Captain")
st.dataframe(captain_pick(df_latest))

# --------- TOP PLAYERS ----------
st.write("### Top Players")
st.dataframe(df_latest.sort_values('predicted_next_points', ascending=False).head(10))



