
#!/usr/bin/env python3
# coding: utf-8
"""FPL Assistant – GW Predictions & Recommendations
Streamlit-ready app converted from the user's notebook code. 
Features:
 - Loads data from GitHub (same sources as original notebook)
 - Cleans and feature-engineers data (same logic)
 - Trains XGBoost regressor + LogisticRegression classifier
 - Caches models and data for 7 days so they retrain weekly
 - Interactive lookups: predicted points, start/bench/sell, replacements
 - Top-5 players per position and team analyses (tables + simple charts)
"""


import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="FPL Assistant – GW Predictions & Recommendations", layout="wide")
st.title("FPL Assistant – GW Predictions & Recommendations")


# ----------------------
# 1) Data loading (cached for 7 days)
# ----------------------
@st.cache_data(ttl=7*24*60*60)
def load_data():
    base_url = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data/2025-2026"
    by_tournament = f"{base_url}/By%20Tournament/Premier%20League"
    urls = {
        "teams": f"{base_url}/teams.csv",
        "players": f"{by_tournament}/GW11/players.csv",
        "playerstats": f"{base_url}/playerstats.csv",
        "gameweek_summaries": f"{base_url}/gameweek_summaries.csv"
    }

    teams = pd.read_csv(urls["teams"]) 
    playerstats = pd.read_csv(urls["playerstats"]) 
    gameweek_summaries = pd.read_csv(urls["gameweek_summaries"]) 
    players = pd.read_csv(urls["players"]) 

    # load per-gameweek files where available
    gw_data = []
    for i in range(1, 39):
        url = f"{by_tournament}/GW{i}/player_gameweek_stats.csv"
        try:
            df = pd.read_csv(url)
            df["gameweek"] = i
            gw_data.append(df)
        except Exception:
            # skip missing future GWs
            pass

    if gw_data:
        player_gw_stats = pd.concat(gw_data, ignore_index=True)
    else:
        player_gw_stats = pd.DataFrame()

    return teams, players, playerstats, gameweek_summaries, player_gw_stats


teams, players, playerstats, gameweek_summaries, player_gw_stats = load_data()
st.success("✅ Data loaded from GitHub")

# ----------------------
# 2) Data cleaning & merge (cached for 7 days)
# ----------------------
@st.cache_data(ttl=7*24*60*60)
def prepare_data(teams, players, playerstats, player_gw_stats):
    # copy to avoid modifying cached originals
    teams = teams.copy()
    players = players.copy()
    playerstats = playerstats.copy()
    player_gw_stats = player_gw_stats.copy()

    # Standardize column names for merging
    player_gw_stats = player_gw_stats.rename(columns={"id": "player_id"})
    playerstats = playerstats.rename(columns={"id": "player_id"})
    teams = teams.rename(columns={"id": "team_id"})

    # Merge
    if not player_gw_stats.empty:
        merged_df = (
            player_gw_stats
            .merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
            .merge(players, on="player_id", how="left")
            .merge(teams, left_on="team_code", right_on="team_id", how="left", suffixes=("", "_team"))
        )
    else:
        merged_df = pd.DataFrame()

    df_clean = merged_df.copy()

    if df_clean.empty:
        return df_clean

    # Drop columns mostly missing
    threshold = 0.6
    too_many_missing = df_clean.columns[df_clean.isnull().mean() > threshold]
    df_clean = df_clean.drop(columns=too_many_missing)

    # Separate numeric and non-numeric
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    # Fill numeric with median
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    # Fill categorical columns
    df_clean[non_numeric_cols] = (
        df_clean[non_numeric_cols]
        .fillna("Unknown")
        .infer_objects(copy=False)
    )

    # Target variable: next GW points
    if 'event_points_gw' in df_clean.columns:
        df_clean['next_gw_points'] = df_clean.groupby('player_id')['event_points_gw'].shift(-1)

    # Feature engineering
    df_clean = df_clean.sort_values(['player_id', 'gameweek'])

    for col in ['event_points_gw', 'goals_scored_gw', 'assists_gw', 'expected_goals_gw', 'expected_assists_gw']:
        if col in df_clean.columns:
            df_clean[f'{col}_roll3'] = (
                df_clean.groupby('player_id')[col]
                .transform(lambda x: x.rolling(3, min_periods=1).mean())
            )

    # Team form
    if 'event_points_gw' in df_clean.columns:
        team_form = (
            df_clean.groupby(['team_id', 'gameweek'])['event_points_gw']
            .mean()
            .reset_index(name='team_avg_points')
        )
        df_clean = df_clean.merge(team_form, on=['team_id', 'gameweek'], how='left')

    # opponent difficulty proxy
    df_clean = df_clean.merge(
        teams[['team_id','strength_defence_home', 'strength_defence_away']].drop_duplicates('team_id'),
        on='team_id', how='left'
    )

    def_cols = [c for c in ['strength_defence_home', 'strength_defence_away'] if c in df_clean.columns]
    if def_cols:
        df_clean['opp_difficulty_proxy'] = df_clean[def_cols].mean(axis=1)
        df_clean['team_strength_avg'] = df_clean[def_cols].mean(axis=1)
    else:
        if 'strength' in df_clean.columns:
            df_clean['opp_difficulty_proxy'] = df_clean['strength']
            df_clean['team_strength_avg'] = df_clean['strength']
        else:
            df_clean['opp_difficulty_proxy'] = 0
            df_clean['team_strength_avg'] = 0

    # Ensure name columns exist
    if 'first_name_gw' not in df_clean.columns and 'first_name' in df_clean.columns:
        df_clean['first_name_gw'] = df_clean['first_name']
    if 'second_name_gw' not in df_clean.columns and 'second_name' in df_clean.columns:
        df_clean['second_name_gw'] = df_clean['second_name']
    if 'web_name_gw' not in df_clean.columns and 'web_name' in df_clean.columns:
        df_clean['web_name_gw'] = df_clean['web_name']

    # full name
    first = df_clean.get('first_name_gw', pd.Series('', index=df_clean.index)).fillna('')
    second = df_clean.get('second_name_gw', pd.Series('', index=df_clean.index)).fillna('')
    df_clean['full_name'] = (first + ' ' + second).str.strip()

    return df_clean

df_clean = prepare_data(teams, players, playerstats, player_gw_stats)

if df_clean.empty:
    st.warning("No gameweek player data detected (player_gameweek_stats might be missing on GitHub). The app needs those CSVs to run the full pipeline.")
    st.stop()

st.success("✅ Data prepared and cleaned")

# ----------------------
# 3) Features & Model training (cached for 7 days so retrain weekly)
# ----------------------
@st.cache_resource(ttl=7*24*60*60)
def train_models(df_clean):
    df_model = df_clean.copy()

    # Feature columns from original notebook (filter by availability)
    feature_cols = [
        'form_gw', 'points_per_game_gw', 'value_form_gw', 'selected_by_percent_gw',
        'minutes_gw', 'total_points_gw',
        'goals_scored_gw', 'assists_gw', 'clean_sheets_gw',
        'bps_gw', 'ict_index_gw',
        'expected_goals_gw', 'expected_assists_gw', 'expected_goal_involvements_gw',
        'expected_goals_conceded_gw',
        'influence_gw', 'creativity_gw', 'threat_gw',
        'event_points_gw_roll3', 'goals_scored_gw_roll3',
        'assists_gw_roll3', 'expected_goals_gw_roll3', 'expected_assists_gw_roll3',
        'team_avg_points', 'team_strength_avg', 'opp_difficulty_proxy',
        'form_season', 'points_per_game_season', 'total_points_season',
        'expected_goals_season', 'expected_assists_season', 'expected_goal_involvements_season',
        'value_form_season', 'value_season_season', 'influence_season', 'creativity_season', 'threat_season', 'ict_index_season'
    ]

    feature_cols = [col for col in feature_cols if col in df_model.columns]

    # Clean target
    if 'next_gw_points' not in df_model.columns:
        df_model['next_gw_points'] = np.nan

    df_model = df_model[df_model['next_gw_points'].notnull()]

    if df_model.shape[0] < 50:
        # Too little data to train properly
        raise ValueError("Not enough labelled rows to train models (need rows with next_gw_points).")

    X = df_model[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_model['next_gw_points'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Logistic Regression for Start/Bench/Sell
    # create required columns if missing
    required_cols = [
        'now_cost_gw', 'position', 'predicted_next_points', 'form_gw', 'team_strength_avg', 'opp_difficulty_proxy'
    ]
    for c in required_cols:
        if c not in df_clean.columns:
            df_clean[c] = 0

    # We'll compute predictions for all rows using xgb
    X_all = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df_clean['predicted_next_points'] = xgb.predict(X_all)

    df_for_logit = df_clean.copy()
    df_for_logit['value_for_money'] = df_for_logit['predicted_next_points'] / df_for_logit.get('now_cost_gw', 1)

    def categorize_player(row):
        if row['predicted_next_points'] >= 10 or row['value_for_money'] >= 1.5:
            return 'Start'
        elif row['predicted_next_points'] >= 6 or row['value_for_money'] >= 1.0:
            return 'Bench'
        else:
            return 'Sell'

    df_for_logit['label'] = df_for_logit.apply(categorize_player, axis=1)

    features_logit = ['predicted_next_points', 'now_cost_gw', 'value_for_money', 'form_gw', 'team_strength_avg', 'opp_difficulty_proxy', 'position']
    features_logit = [c for c in features_logit if c in df_for_logit.columns]

    X_log = df_for_logit[features_logit]
    y_log = df_for_logit['label']

    # Preprocessor
    num_cols = [c for c in ['predicted_next_points', 'now_cost_gw', 'value_for_money', 'form_gw', 'team_strength_avg', 'opp_difficulty_proxy'] if c in X_log.columns]
    cat_cols = [c for c in ['position'] if c in X_log.columns]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='drop')

    logit_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42))
    ])

    # Fit logistic regression
    logit_pipe.fit(X_log.fillna(0), y_log)

    # store meta
    meta = {
        'xgb_mae': mae,
        'xgb_r2': r2,
        'n_rows_train': X_train.shape[0]
    }

    return xgb, logit_pipe, df_clean, feature_cols, meta


with st.spinner('Training models (this happens weekly and is cached for 7 days)...'):
    try:
        xgb_model, logit_model, df_clean_with_preds, feature_cols, meta = train_models(df_clean)
        st.success(f"✅ Models trained — MAE: {meta['xgb_mae']:.3f}, R²: {meta['xgb_r2']:.3f}")
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

# ----------------------
# 4) Helper functions for lookup & recommendations
# ----------------------

def normalize_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text.lower().strip()
    return str(text).lower().strip()

def get_player_prediction(df, player_input):
    player_input_norm = normalize_text(player_input)
    df_search = df.copy()
    df_search['web_name_norm'] = df_search['web_name_gw'].astype(str).apply(normalize_text)
    df_search['first_name_norm'] = df_search['first_name_gw'].astype(str).apply(normalize_text)
    df_search['second_name_norm'] = df_search['second_name_gw'].astype(str).apply(normalize_text)
    df_search['full_name_norm'] = (df_search['first_name_norm'].fillna('') + ' ' + df_search['second_name_norm'].fillna('')).str.strip()
    df_search['player_id_str'] = df_search['player_id'].astype(str)

    mask = (
        df_search['web_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['first_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['second_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['full_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['player_id_str'].str.contains(player_input_norm, na=False)
    )

    results = df_search.loc[mask, ['player_id', 'first_name_gw', 'second_name_gw', 'web_name_gw', 'team_id', 'predicted_next_points']]
    return results.drop_duplicates('player_id').sort_values('predicted_next_points', ascending=False)

def get_recommendation_for_player(df, player_row, logit_model):
    # Prepare features same as training
    features = ['predicted_next_points', 'now_cost_gw', 'value_for_money', 'form_gw', 'team_strength_avg', 'opp_difficulty_proxy', 'position']
    features = [c for c in features if c in df.columns]
    X_player = player_row[features]
    # ensure shape
    X_player = X_player.fillna(0)
    pred = logit_model.predict(X_player)[0]
    return pred

def suggest_replacements(df, player_row, top_n=3):
    position = player_row['position']
    candidates = df[df['position'] == position].copy()
    candidates = candidates.drop_duplicates(subset=['player_id'])
    candidates = candidates[candidates['player_id'] != player_row['player_id']]
    candidates['value_for_money'] = candidates['predicted_next_points'] / candidates.get('now_cost_gw', 1)
    candidates['score'] = (
        candidates['predicted_next_points'] * 0.5 +
        candidates['value_for_money'] * 0.3 +
        candidates['form_gw'].fillna(0) * 0.2
    )
    top = candidates.sort_values(by='score', ascending=False).head(top_n)
    return top[['player_id', 'full_name', 'position', 'team_name_final', 'predicted_next_points', 'value_for_money', 'form_gw']]

# ----------------------
# 5) Interactive UI elements
# ----------------------
st.header("🔎 Player Lookup & Predictions")
col1, col2 = st.columns([2,1])
with col1:
    lookup = st.text_input("Enter player web name, first/second name, full name, or player ID:")
with col2:
    lookup_button = st.button("Search")

if lookup and lookup_button:
    results = get_player_prediction(df_clean_with_preds, lookup)
    if results.empty:
        st.warning("No matching player found. Try a different name or player ID.")
    else:
        st.dataframe(results.reset_index(drop=True))
        # For first result, show recommendation
        first_id = results['player_id'].iloc[0]
        player_row = df_clean_with_preds[df_clean_with_preds['player_id'] == first_id].drop_duplicates('player_id').set_index('player_id')
        pred_points = player_row['predicted_next_points'].iloc[0]
        st.metric("🔮 Predicted Next GW Points", f"{pred_points:.2f}")
        # prepare for logistic prediction
        player_for_logit = player_row[[c for c in ['predicted_next_points', 'now_cost_gw', 'value_for_money', 'form_gw', 'team_strength_avg', 'opp_difficulty_proxy', 'position'] if c in player_row.columns]]
        rec = get_recommendation_for_player(df_clean_with_preds, player_for_logit, logit_model)
        st.info(f"Recommendation: {rec}")
        # Replacements
        replacements = suggest_replacements(df_clean_with_preds, player_row.iloc[0], top_n=3)
        st.subheader("Top 3 Replacement Suggestions")
        st.table(replacements)

# ----------------------
# 6) Top-5 analyses and team tables
# ----------------------
st.header("📋 Top-5 Players by Predicted Points & Value-for-Money")
# latest per-player
df_latest = (
    df_clean_with_preds.sort_values(['player_id', 'gameweek'], ascending=[True, False])
    .drop_duplicates(subset='player_id', keep='first')
)
# compute value_for_money (avoid divide by zero)
df_latest['value_for_money'] = df_latest['predicted_next_points'] / df_latest.get('now_cost_gw', 1)

positions = df_latest['position'].dropna().unique()
for pos in sorted(positions):
    st.subheader(f"Top 5 {pos}s by Predicted Points")
    top5 = (
        df_latest[df_latest['position'] == pos]
        .dropna(subset=['predicted_next_points'])
        .sort_values('predicted_next_points', ascending=False)
        .head(5)[['full_name', 'team_name_final', 'predicted_next_points']]
    )
    st.table(top5.reset_index(drop=True))

    st.subheader(f"Top 5 {pos}s by Value-for-Money")
    top5_vfm = (
        df_latest[df_latest['position'] == pos]
        .dropna(subset=['value_for_money'])
        .sort_values('value_for_money', ascending=False)
        .head(5)[['full_name', 'team_name_final', 'value_for_money']]
    )
    st.table(top5_vfm.reset_index(drop=True))

# Teams by opponent difficulty and total predicted points
st.header("🏟️ Teams by Opponent Difficulty & Total Predicted Points")
team_difficulty = (
    df_latest.groupby('team_name_final')['opp_difficulty_proxy']
    .mean()
    .reset_index(name='avg_opponent_difficulty')
    .sort_values('avg_opponent_difficulty', ascending=True)
)
team_predicted_points = (
    df_latest.groupby('team_name_final')['predicted_next_points']
    .sum()
    .reset_index(name='total_predicted_points')
    .sort_values('total_predicted_points', ascending=False)
)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Teams — Easier Opponents (lower = easier)")
    st.table(team_difficulty)
    # simple bar chart
    st.bar_chart(team_difficulty.set_index('team_name_final')['avg_opponent_difficulty'])
with col_b:
    st.subheader("Teams — Total Predicted Points")
    st.table(team_predicted_points)
    st.bar_chart(team_predicted_points.set_index('team_name_final')['total_predicted_points'])

# ----------------------
# 7) Misc & Export
# ----------------------
st.header("⚙️ Export & Notes")
st.markdown("- Models retrain weekly (cache TTL = 7 days).\n- If GitHub per-GW CSVs are not present for future gameweeks, the app will use whatever is available.")


# Allow downloading top lists as CSV
if st.button('Download latest top players (CSV)'):
    csv = df_latest.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='df_latest_top_players.csv', mime='text/csv')

st.caption("App created from user's notebook logic; only data & logic present in the original code are used.")
