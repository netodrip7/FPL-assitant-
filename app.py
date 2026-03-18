import streamlit as st
from script import run_pipeline

st.set_page_config(layout="wide")

# ===== STYLE =====
st.markdown("""
<style>
body {background:#050c1b;color:white;}
.title {text-align:center;font-size:64px;font-weight:800;color:#5fa8ff;}
.tag {text-align:center;font-size:22px;color:#7fb3ff;}
.note {text-align:center;color:#8aa4c5;}
.desc {text-align:center;font-size:18px;color:#c6d4f0;}
.section {font-size:26px;margin-top:40px;color:#e3ecff;}
.helper {color:#7f9bb3;font-size:14px;}
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown('<div class="title">STATS MERCHANT</div>', unsafe_allow_html=True)
st.markdown('<div class="tag">ball knowledge backed by stats</div>', unsafe_allow_html=True)
st.markdown('<div class="note">here for the first time? please wait for 30-50 seconds. it’ll be so much quicker the next time you’re here i promise.</div>', unsafe_allow_html=True)
st.markdown('<div class="desc">FPL managers, tap in. This ain’t just another stats site, this is your differential factory. Get clean data, fixture swings, xG juice, and captaincy calls that actually hit. No more picking your team on vibes only, we move with data now. GET THAT RANK UP.</div>', unsafe_allow_html=True)

# ===== LOAD =====
@st.cache_data(ttl=600)
def load():
    return run_pipeline()

df_clean, df_latest, top5_pred, top5_vfm, team_diff, team_pts, fixtures, next_gw = load()

# ===== INPUTS =====
st.markdown('<div class="section">Player Tools</div>', unsafe_allow_html=True)

st.markdown('<div class="helper">Enter a player’s name to see their predicted points.</div>', unsafe_allow_html=True)
p1 = st.text_input("")

if p1:
    st.dataframe(df_latest[df_latest['full_name'].str.contains(p1, case=False)][['full_name','predicted_next_points']])

st.markdown('<div class="helper">Enter a player’s name to get suggested replacements.</div>', unsafe_allow_html=True)
p2 = st.text_input(" ", key="r")

if p2:
    player = df_latest[df_latest['full_name'].str.contains(p2, case=False)].head(1)
    if not player.empty:
        pos = player.iloc[0]['position']
        st.dataframe(df_latest[df_latest['position']==pos].sort_values('predicted_next_points', ascending=False).head(3))

st.markdown('<div class="helper">Enter a player’s name to find out what to do with them.</div>', unsafe_allow_html=True)
p3 = st.text_input("  ", key="rec")

if p3:
    player = df_latest[df_latest['full_name'].str.contains(p3, case=False)]
    st.dataframe(player[['full_name','recommendation','predicted_next_points']])

# ===== ANALYTICS =====
st.markdown('<div class="section">Top 5 Players by Predicted Points</div>', unsafe_allow_html=True)
st.dataframe(top5_pred)

st.markdown('<div class="section">Top 5 Players by Value for Money</div>', unsafe_allow_html=True)
st.dataframe(top5_vfm)

st.markdown('<div class="section">Teams by Difficulty</div>', unsafe_allow_html=True)
st.dataframe(team_diff)

st.markdown('<div class="section">Teams by Total Predicted Points</div>', unsafe_allow_html=True)
st.dataframe(team_pts)



