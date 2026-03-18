import streamlit as st
from data_loader import load_all_data
from processing import process_data
from models import train_model
from utils import get_player_prediction

st.set_page_config(page_title="Stat Merchant", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
body {
    background-color: #0b1e3c;
    color: white;
    font-family: 'Trebuchet MS', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("Stat Merchant")
st.write("ball knowledge, certified by stats.")

# ---------- LOAD PIPELINE ----------
@st.cache_data(ttl=300)
def load_pipeline():
    teams, playerstats, gameweek_summaries, players, player_gw_stats = load_all_data()
    df = process_data(teams, playerstats, players, player_gw_stats)

    feature_cols = [col for col in df.columns if "gw" in col or "season" in col]

    model = train_model(df, feature_cols)

    X_all = df[feature_cols].fillna(0)
    df['predicted_next_points'] = model.predict(X_all)

    return df

df_clean = load_pipeline()

# ---------- INPUT (TOP) ----------
st.header("Player Lookup")

player_name = st.text_input("Enter player name")

if player_name:
    results = get_player_prediction(df_clean, player_name)
    st.dataframe(results)

# ---------- ANALYTICS ----------
st.header("Analytics")

top_players = df_clean[['web_name_gw','predicted_next_points']]\
    .drop_duplicates()\
    .sort_values('predicted_next_points', ascending=False)\
    .head(10)

st.subheader("Top 10 Players")
st.dataframe(top_players)




