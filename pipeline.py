# THIS IS YOUR ORIGINAL CODE — STRUCTURED ONLY

def run_pipeline():

    import pandas as pd
    import numpy as np
    import requests
    from xgboost import XGBRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

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

    merged_df = (
        player_gw_stats
        .merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
        .merge(players, on="player_id", how="left")
        .merge(teams, left_on="team_code", right_on="team_id", how="left")
    )

    df_clean = merged_df.copy()

    # CLEANING
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > 0.6])
    num = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num] = df_clean[num].fillna(df_clean[num].median())
    df_clean = df_clean.fillna("Unknown")

    # TARGET
    df_clean['next_gw_points'] = df_clean.groupby('player_id')['event_points_gw'].shift(-1)

    df_clean = df_clean.sort_values(['player_id','gameweek'])

    # ROLLING
    for col in ['event_points_gw','goals_scored_gw','assists_gw','expected_goals_gw','expected_assists_gw']:
        if col in df_clean.columns:
            df_clean[f'{col}_roll3'] = df_clean.groupby('player_id')[col].transform(lambda x: x.rolling(3,1).mean())

    # TEAM FORM
    team_form = df_clean.groupby(['team_id','gameweek'])['event_points_gw'].mean().reset_index(name='team_avg_points')
    df_clean = df_clean.merge(team_form, on=['team_id','gameweek'], how='left')

    df_clean['opp_difficulty_proxy'] = df_clean[['strength_defence_home','strength_defence_away']].mean(axis=1)
    df_clean['team_strength_avg'] = df_clean['opp_difficulty_proxy']

    # FEATURES (EXACT)
    feature_cols = [
        'form_gw','points_per_game_gw','value_form_gw','selected_by_percent_gw',
        'minutes_gw','total_points_gw',
        'goals_scored_gw','assists_gw','clean_sheets_gw',
        'bps_gw','ict_index_gw',
        'expected_goals_gw','expected_assists_gw','expected_goal_involvements_gw',
        'expected_goals_conceded_gw',
        'influence_gw','creativity_gw','threat_gw',
        'event_points_gw_roll3','goals_scored_gw_roll3',
        'assists_gw_roll3','expected_goals_gw_roll3','expected_assists_gw_roll3',
        'team_avg_points','team_strength_avg',
        'opp_difficulty_proxy',
        'form_season','points_per_game_season','total_points_season',
        'expected_goals_season','expected_assists_season',
        'expected_goal_involvements_season','value_form_season','value_season_season',
        'influence_season','creativity_season','threat_season','ict_index_season'
    ]

    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    df_model = df_clean[df_clean['next_gw_points'].notnull()]
    X = df_model[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    y = df_model['next_gw_points']

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
    xgb.fit(X,y)

    df_clean['predicted_next_points'] = xgb.predict(df_clean[feature_cols].fillna(0))

    # FULL NAME (CRITICAL FIX)
    df_clean['full_name'] = df_clean['first_name_gw'].fillna('') + ' ' + df_clean['second_name_gw'].fillna('')

    # LATEST ONLY
    df_latest = df_clean.sort_values(['player_id','gameweek'], ascending=[True,False]).drop_duplicates('player_id')

    return df_clean, df_latest
