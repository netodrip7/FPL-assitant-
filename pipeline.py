import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_pipeline():

    base_url = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data/2025-2026"
    by_tournament = f"{base_url}/By%20Tournament/Premier%20League"

    teams = pd.read_csv(f"{base_url}/teams.csv")
    playerstats = pd.read_csv(f"{base_url}/playerstats.csv")
    players = pd.read_csv(f"{by_tournament}/GW11/players.csv")

    gw_data = []
    for i in range(1, 39):
        try:
            df = pd.read_csv(f"{by_tournament}/GW{i}/player_gameweek_stats.csv")
            df["gameweek"] = i
            gw_data.append(df)
        except:
            pass

    player_gw_stats = pd.concat(gw_data, ignore_index=True)

    player_gw_stats = player_gw_stats.rename(columns={"id": "player_id"})
    playerstats = playerstats.rename(columns={"id": "player_id"})
    teams = teams.rename(columns={"id": "team_id"})

    df = (
    player_gw_stats
    .merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
    .merge(players, on="player_id", how="left")
    .merge(teams, left_on="team_code", right_on="team_id", how="left")
)

    # CLEAN
    df = df.drop(columns=df.columns[df.isnull().mean() > 0.6])
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median())
    df = df.fillna("Unknown")

    df['next_gw_points'] = df.groupby('player_id')['event_points_gw'].shift(-1)
    df = df.sort_values(['player_id', 'gameweek'])

    # ROLLING
    for col in ['event_points_gw','goals_scored_gw','assists_gw']:
        if col in df.columns:
            df[f'{col}_roll3'] = df.groupby('player_id')[col].transform(lambda x: x.rolling(3,1).mean())

    # TEAM FORM
    team_form = df.groupby(['team_id','gameweek'])['event_points_gw'].mean().reset_index(name='team_avg_points')
    df = df.merge(team_form, on=['team_id','gameweek'], how='left')

    df['opp_difficulty_proxy'] = df[['strength_defence_home','strength_defence_away']].mean(axis=1)
    df['team_strength_avg'] = df['opp_difficulty_proxy']

    feature_cols = [c for c in df.columns if df[c].dtype != 'object']

    df_model = df[df['next_gw_points'].notnull()]
    X = df_model[feature_cols].fillna(0)
    y = df_model['next_gw_points']

    model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6)
    model.fit(X, y)

    df['predicted_next_points'] = model.predict(df[feature_cols].fillna(0))

    # LATEST ONLY
    df_latest = df.sort_values(['player_id','gameweek'], ascending=[True,False]).drop_duplicates('player_id')

    df_latest['value_for_money'] = df_latest['predicted_next_points'] / (df_latest['now_cost_gw']+1)

    return df, df_latest
