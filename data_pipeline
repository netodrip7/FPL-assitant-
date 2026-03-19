import pandas as pd
base_url = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data/2025-2026"
by_tournament = f"{base_url}/By%20Tournament/Premier%20League"
urls = {
    "teams": f"{base_url}/teams.csv",
    "players": f"{by_tournament}/GW11/players.csv",  # snapshot (or replace GW11 with latest)
    "playerstats": f"{base_url}/playerstats.csv",
    "gameweek_summaries": f"{base_url}/gameweek_summaries.csv"
}

teams = pd.read_csv(urls["teams"])
playerstats = pd.read_csv(urls["playerstats"])
gameweek_summaries = pd.read_csv(urls["gameweek_summaries"])
players = pd.read_csv(urls["players"])

gw_data = []
for i in range(1, 39):  # 1–38 GWs
    url = f"{by_tournament}/GW{i}/player_gameweek_stats.csv"
    try:
        df = pd.read_csv(url)
        df["gameweek"] = i
        gw_data.append(df)
    except Exception:
        pass  # skip future GWs that aren't released yet

player_gw_stats = pd.concat(gw_data, ignore_index=True)

# Standardize column names for merging
player_gw_stats = player_gw_stats.rename(columns={"id": "player_id"})
playerstats = playerstats.rename(columns={"id": "player_id"})
players = players.rename(columns={"player_id": "player_id"})  # already fine
teams = teams.rename(columns={"id": "team_id"})

merged_df = (
    player_gw_stats
    .merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
    .merge(players, on="player_id", how="left")  # adds team_code + position
    .merge(teams, left_on="team_code", right_on="team_id", how="left", suffixes=("", "_team"))
)

print(merged_df.shape)
print(merged_df.columns[:25])
#Each row = one player in one gameweek.
#Each column = a metric from one of those 4 sources.
#You have about 79,000 rows × 194 columns, meaning:
#roughly 2,000+ players × 35–40 gameweeks worth of data
#every single numeric and categorical variable combined into one master table.

import numpy as np
df_clean = merged_df.copy()
# 1️⃣ Drop columns that are mostly missing
threshold = 0.6
too_many_missing = df_clean.columns[df_clean.isnull().mean() > threshold]
df_clean = df_clean.drop(columns=too_many_missing)
# 2️⃣ Separate numeric and non-numeric columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
# 3️⃣ Fill numeric columns with median
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
# 4️⃣ Fill categorical columns with 'Unknown' + infer_objects() fix
df_clean[non_numeric_cols] = (
    df_clean[non_numeric_cols]
    .fillna("Unknown")
    .infer_objects(copy=False)   # ✅ <— This is the exact line that fixes the FutureWarning
)
# 5️⃣ Verify cleaning
print("✅ Data cleaned:")
print("Rows:", df_clean.shape[0], "| Columns:", df_clean.shape[1])
print("Remaining NaNs:", df_clean.isnull().sum().sum())

#target variable
df_clean['next_gw_points'] = df_clean.groupby('player_id')['event_points_gw'].shift(-1)

# --- Feature Engineering (fixed for your dataset) ---

df_clean = df_clean.sort_values(['player_id', 'gameweek'])

# Rolling averages (form over last 3 gameweeks)
for col in ['event_points_gw', 'goals_scored_gw', 'assists_gw', 'expected_goals_gw', 'expected_assists_gw']:
    if col in df_clean.columns:
        df_clean[f'{col}_roll3'] = (
            df_clean.groupby('player_id')[col]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

# Team form (average points per team per GW)
team_form = (
    df_clean.groupby(['team_id', 'gameweek'])['event_points_gw']
    .mean()
    .reset_index(name='team_avg_points')
)
df_clean = df_clean.merge(team_form, on=['team_id', 'gameweek'], how='left')

# Opponent difficulty (proxy using team defensive and overall strength)
df_clean = df_clean.merge(
    teams[['team_id','strength_defence_home', 'strength_defence_away']],
    on='team_id', how='left'
)

# ✅ Use available team strength columns safely
def_cols = [c for c in ['strength_defence_home', 'strength_defence_away'] if c in df_clean.columns]
if def_cols:
    df_clean['opp_difficulty_proxy'] = df_clean[def_cols].mean(axis=1)
    df_clean['team_strength_avg'] = df_clean[def_cols].mean(axis=1)
else:
    # fallback if those cols missing — use 'strength' as proxy
    if 'strength' in df_clean.columns:
        df_clean['opp_difficulty_proxy'] = df_clean['strength']
        df_clean['team_strength_avg'] = df_clean['strength']

print("✅ Feature engineering complete.")
print("New columns added:", [c for c in df_clean.columns if 'roll3' in c or 'avg' in c or 'diff' in c])

# --- Feature selection for prediction ---

# Target variable: next gameweek points
y = df_clean['next_gw_points']

# Feature columns
feature_cols = [
    # Player performance and form
    'form_gw', 'points_per_game_gw', 'value_form_gw', 'selected_by_percent_gw',
    'minutes_gw', 'total_points_gw',
    'goals_scored_gw', 'assists_gw', 'clean_sheets_gw',
    'bps_gw', 'ict_index_gw',
    'expected_goals_gw', 'expected_assists_gw', 'expected_goal_involvements_gw',
    'expected_goals_conceded_gw',
    'influence_gw', 'creativity_gw', 'threat_gw',

    # Rolling averages (short-term form)
    'event_points_gw_roll3', 'goals_scored_gw_roll3',
    'assists_gw_roll3', 'expected_goals_gw_roll3', 'expected_assists_gw_roll3',

    # Team-level features
    'team_avg_points', 'team_strength_avg',

    # Opponent difficulty proxy
    'opp_difficulty_proxy',

    # Season-level performance
    'form_season', 'points_per_game_season', 'total_points_season',
    'expected_goals_season', 'expected_assists_season',
    'expected_goal_involvements_season', 'value_form_season', 'value_season_season',
    'influence_season', 'creativity_season', 'threat_season', 'ict_index_season'
]

# Filter only available columns (avoids KeyErrors)
feature_cols = [col for col in feature_cols if col in df_clean.columns]

# Define X
X = df_clean[feature_cols]

print("✅ Features ready for modeling:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Features used:", feature_cols)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Drop rows where target is missing or invalid
df_model = df_clean.copy()

# Remove NaN, inf, or -inf from target column
df_model = df_model[
    df_model['next_gw_points'].notnull() & 
    (~df_model['next_gw_points'].isin([np.inf, -np.inf]))
]

# Replace infinite values in features with NaN and then fill
X = df_model[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df_model['next_gw_points'].astype(float)

print("✅ Cleaned model data:")
print("Rows:", X.shape[0], "| Columns:", X.shape[1])
print("Remaining NaNs in y:", y.isna().sum())


# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

# Initialize model
model = XGBRegressor(
    n_estimators=300,         # Number of boosting rounds
    learning_rate=0.05,       # Step size shrinkage
    max_depth=6,              # Max depth of trees
    subsample=0.8,            # Row sampling
    colsample_bytree=0.8,     # Feature sampling
    random_state=42,
    objective='reg:squarederror'
)

# Train model
model.fit(X_train, y_train)

print("✅ Model training complete!")

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Absolute Error (MAE): {mae:.3f}")
print(f"📈 R² Score: {r2:.3f}")

import matplotlib.pyplot as plt

# --- PREDICT FOR ALL PLAYERS (including those without y) ---

# Clean the feature data for all rows
X_all = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Generate predictions using the trained model
df_clean['predicted_next_points'] = model.predict(X_all)

# Show top 10 predicted players for next GW
predictions = (
    df_clean[['player_id', 'web_name_gw', 'team_id', 'predicted_next_points']]
    .drop_duplicates('player_id')
    .sort_values('predicted_next_points', ascending=False)
)

print("🔮 Top 10 predicted players for next GW:")
print(predictions.head(10))

import unicodedata

# --- Helper: Normalize text (remove accents, lowercase) ---
def normalize_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text.lower().strip()
    return str(text).lower().strip()


# --- INTERACTIVE PREDICTION LOOKUP (accent-insensitive + full name support) ---
def get_player_prediction(df, player_input):
    """
    Search for player by web_name, first name, second name, full name, or player_id
    and return their predicted next GW points.
    Accent-insensitive and flexible for partial matches.
    """
    player_input_norm = normalize_text(player_input)

    # Create normalized versions of relevant columns for easy matching
    df_search = df.copy()
    df_search['web_name_norm'] = df_search['web_name_gw'].apply(normalize_text)
    df_search['first_name_norm'] = df_search['first_name_gw'].apply(normalize_text)
    df_search['second_name_norm'] = df_search['second_name_gw'].apply(normalize_text)
    df_search['full_name_norm'] = (  # combine first + second name for full name search
        df_search['first_name_norm'].fillna('') + ' ' + df_search['second_name_norm'].fillna('')
    ).str.strip()
    df_search['player_id_str'] = df_search['player_id'].astype(str)

    # Flexible search mask
    mask = (
        df_search['web_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['first_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['second_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['full_name_norm'].str.contains(player_input_norm, na=False)
        | df_search['player_id_str'].str.contains(player_input_norm, na=False)
    )

    results = df_search.loc[
        mask,
        ['player_id', 'first_name_gw', 'second_name_gw', 'web_name_gw', 'team_id', 'predicted_next_points']
    ]

    if results.empty:
        print("⚠️ No matching player found. Try a different name or player ID.")
    else:
        print(f"✅ Predicted Next GW Points for players matching '{player_input}':")
        print(results.drop_duplicates('player_id').sort_values('predicted_next_points', ascending=False))

# ===============================================
# 🧠 Logistic Regression: FPL Player Recommendation
# ===============================================

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1️⃣ Ensure necessary columns exist
required_cols = [
    'predicted_next_points', 'now_cost_gw', 'position',
    'form_gw', 'team_strength_avg', 'opp_difficulty_proxy'
]
missing = [c for c in required_cols if c not in df_clean.columns]
if missing:
    print(f"⚠️ Missing columns: {missing}. Filling with defaults.")
    for c in missing:
        df_clean[c] = 0

# 2️⃣ Value-for-money feature
df_clean['value_for_money'] = df_clean['predicted_next_points'] / df_clean['now_cost_gw'].replace(0, 1)

# 3️⃣ Create target labels (FPL logic-based)
def categorize_player(row):
    if row['predicted_next_points'] >= 10 or row['value_for_money'] >= 1.5:
        return 'Start'
    elif row['predicted_next_points'] >= 6 or row['value_for_money'] >= 1.0:
        return 'Bench'
    else:
        return 'Sell'

df_clean['label'] = df_clean.apply(categorize_player, axis=1)

# 4️⃣ Prepare features and target
features = [
    'predicted_next_points', 'now_cost_gw', 'value_for_money',
    'form_gw', 'team_strength_avg', 'opp_difficulty_proxy', 'position'
]
X = df_clean[features]
y = df_clean['label']

# 5️⃣ Preprocessing: scale numeric & one-hot encode position
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), [
        'predicted_next_points', 'now_cost_gw', 'value_for_money',
        'form_gw', 'team_strength_avg', 'opp_difficulty_proxy'
    ]),
    ('cat', OneHotEncoder(), ['position'])
])

# 6️⃣ Build pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42))
])

# 7️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8️⃣ Train model
model.fit(X_train, y_train)
print("✅ Logistic Regression model trained successfully!")

# 9️⃣ Evaluate model
acc = model.score(X_test, y_test)
print(f"📊 Model accuracy on test set: {acc:.2f}")

# 🔟 Predict recommendations for all players
df_clean['recommendation'] = model.predict(X)

import re

# 🧩 Interactive Recommendation Function
def get_fpl_recommendation():
    user_input = input("Enter player name or ID: ").strip().lower()

    # Try to match player by ID or any part of their name
    if user_input.isdigit():
        player = df_clean[df_clean['player_id'] == int(user_input)]
    else:
        # Normalize names like João → joao for easier matching
        normalized_input = re.sub(r'[^a-z0-9 ]', '', user_input)
        df_clean['normalized_name'] = (
            df_clean['full_name'].str.lower().replace({r'[^a-z0-9 ]': ''}, regex=True)
        )
        df_clean['normalized_webname'] = (
            df_clean['web_name_gw'].astype(str).str.lower().replace({r'[^a-z0-9 ]': ''}, regex=True)
        )
        player = df_clean[
            df_clean['normalized_name'].str.contains(normalized_input, na=False)
            | df_clean['normalized_webname'].str.contains(normalized_input, na=False)
        ]

    if player.empty:
        print("❌ Player not found. Try again with different spelling or ID.")
        return

    # Predict using the logistic regression model
    X_player = player[[
        'predicted_next_points', 'now_cost_gw', 'value_for_money',
        'form_gw', 'team_strength_avg', 'opp_difficulty_proxy', 'position'
    ]]
    recommendation = model.predict(X_player)[0]

    # Display clean output
    print("\n🎯 Player Recommendation:")
    print("──────────────────────────")
    print(f"🧍‍♂️ Name: {player['full_name'].iloc[0]}")

    # Handle team name gracefully (avoiding .iloc on strings)
    if 'team_name_final' in player.columns and not player['team_name_final'].isna().all():
        team_name = player['team_name_final'].iloc[0]
    elif 'team_name' in player.columns and not player['team_name'].isna().all():
        team_name = player['team_name'].iloc[0]
    else:
        team_name = "Unknown"

    print(f"🏟️ Team: {team_name}")
    print(f"🎯 Predicted Next GW Points: {player['predicted_next_points'].iloc[0]:.2f}")
    print(f"💰 Cost: £{player['now_cost_gw'].iloc[0]:.1f}m")
    print(f"⚖️ Value for Money: {player['value_for_money'].iloc[0]:.2f}")
    print(f"📊 Position: {player['position'].iloc[0]}")
    print(f"🧩 Recommendation: {recommendation.upper()}")
    print("──────────────────────────")


# ✅ Ensure full_name exists before recommendation lookup
if 'full_name' not in df_clean.columns:
    first = df_clean.get('first_name_gw', pd.Series('', index=df_clean.index)).fillna('')
    second = df_clean.get('second_name_gw', pd.Series('', index=df_clean.index)).fillna('')
    df_clean['full_name'] = (first + ' ' + second).str.strip()



# ===============================================
# 🔁 Interactive Replacement Suggestion Function
# ===============================================

def suggest_replacements_interactive(df=df_clean, top_n=3):
    """
    Ask user for player name and suggest top replacements
    based on predicted points, value for money, and form.
    Ensures no duplicate players appear.
    """
    import re

    user_input = input("Enter player name to find replacements: ").strip().lower()
    normalized_input = re.sub(r'[^a-z0-9 ]', '', user_input)

    # Normalize names for matching
    df['full_name_norm'] = (
        (df['first_name_gw'].astype(str) + ' ' + df['second_name_gw'].astype(str))
        .str.lower().replace({r'[^a-z0-9 ]': ''}, regex=True)
    )
    df['web_name_norm'] = (
        df['web_name_gw'].astype(str).str.lower().replace({r'[^a-z0-9 ]': ''}, regex=True)
    )

    # Find matching player(s)
    target = df[
        df['full_name_norm'].str.contains(normalized_input, na=False)
        | df['web_name_norm'].str.contains(normalized_input, na=False)
    ]

    if target.empty:
        print("❌ Player not found. Try again.")
        return

    # Pick first match if multiple
    target = target.iloc[0]
    position = target['position']
    team_name = target.get('team_name_final', target.get('team_name', 'Unknown'))
    print(f"\n💡 Finding replacements for {target['first_name_gw']} {target['second_name_gw']} ({position}) from {team_name}...\n")

    # Filter same-position players
    candidates = df[df['position'] == position].copy()

    # Remove duplicates (keep one row per player)
    candidates = candidates.drop_duplicates(subset=['player_id'])

    # Remove the same player
    candidates = candidates[candidates['player_id'] != target['player_id']]

    # Compute overall score
    candidates['score'] = (
        candidates['predicted_next_points'] * 0.5 +
        candidates['value_for_money'] * 0.3 +
        candidates['form_gw'] * 0.2
    )

    # Get top N
    top_replacements = candidates.sort_values(by='score', ascending=False).head(top_n)

    print("✨ Top 3 Recommended Replacements:")
    print("─────────────────────────────────")

    for _, row in top_replacements.iterrows():
        print(f"🧍‍♂️ {row['first_name_gw']} {row['second_name_gw']} ({row['position']})")
        print(f"💫 Predicted Pts: {row['predicted_next_points']:.2f}")
        print(f"💰 Value for Money: {row['value_for_money']:.2f}")
        print(f"🔥 Form: {row['form_gw']:.2f}")
        print("─────────────────────────────────")


# ===========================================
# 🧠 AUTO-FIX TEAM IDS & NAMES (robust mapping)
# ===========================================

import numpy as np
import pandas as pd

# ✅ Reference map (official FPL-like ID ↔ Name)
team_map_df = pd.DataFrame({
    'team_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'team_name': [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
        'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
        'Newcastle United', 'Nottingham Forest', 'Southampton',
        'Tottenham Hotspur', 'West Ham United', 'Wolverhampton Wanderers'
    ]
})

# 🧩 Create normalized versions for matching
team_map_df['team_name_norm'] = team_map_df['team_name'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)

# 🧠 Step 1: Try to normalize and map if `team_id` isn’t 1–20 yet
if 'team_id' in df_clean.columns:
    # Convert team_id to numeric safely
    df_clean['team_id'] = pd.to_numeric(df_clean['team_id'], errors='coerce')

    # If IDs look invalid (not in 1–20 range), fallback to team name mapping
    invalid_mask = ~df_clean['team_id'].isin(range(1, 21))
    if invalid_mask.sum() > 0 and 'team_name_final' in df_clean.columns:
        df_clean['team_name_final_norm'] = df_clean['team_name_final'].astype(str).str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
        df_clean = df_clean.merge(
            team_map_df[['team_id', 'team_name_norm']],
            left_on='team_name_final_norm',
            right_on='team_name_norm',
            how='left',
            suffixes=('', '_mapped')
        )
        # Fill missing numeric IDs with mapped ones
        df_clean['team_id'] = df_clean['team_id'].fillna(df_clean['team_id_mapped'])
        df_clean.drop(columns=['team_name_norm', 'team_name_final_norm', 'team_id_mapped'], inplace=True, errors='ignore')

# 🧠 Step 2: Map team names for both home and opponent teams
df_clean['team_name_final'] = df_clean['team_id'].map(dict(zip(team_map_df['team_id'], team_map_df['team_name'])))
if 'opponent_team' in df_clean.columns:
    df_clean['opp_name_final'] = df_clean['opponent_team'].map(dict(zip(team_map_df['team_id'], team_map_df['team_name'])))
else:
    df_clean['opp_name_final'] = np.nan

# 🧠 Step 3: Fill missing values cleanly
df_clean['team_name_final'] = df_clean['team_name_final'].fillna("Unknown")
df_clean['opp_name_final'] = df_clean['opp_name_final'].fillna("Unknown")

# ===============================================
# 🏟️ Detect Fixture Data + Show Fixtures for Latest Gameweek (Safe version)
# ===============================================

# 🧩 Check team code ↔ name mapping

# Select only relevant columns from your teams dataset
team_mapping = teams[['team_id', 'name', 'short_name']].sort_values('team_id').reset_index(drop=True)

# Display it nicely
print("📘 Team Code ↔ Name Mapping:")
print(team_mapping)

def detect_fixtures_df():
    candidates = []
    for name, obj in globals().items():
        if isinstance(obj, pd.DataFrame):
            # Convert all column names to strings safely
            cols = set(map(str.lower, map(str, obj.columns)))
            if {'home_team', 'away_team', 'gameweek'}.issubset(cols):
                candidates.append(name)
    return candidates[0] if candidates else None

# 1️⃣ Detect fixture DataFrame
fixtures_var_name = detect_fixtures_df()

if fixtures_var_name:
    fixtures_df = globals()[fixtures_var_name]
    print(f"✅ Detected fixtures dataframe: {fixtures_var_name}")
else:
    print("⚠️ Could not detect any fixture dataset — please load one containing columns ['home_team', 'away_team', 'gameweek']")
    fixtures_df = None

# 2️⃣ Get latest gameweek
if 'gameweek' in df_clean.columns:
    latest_gw = df_clean['gameweek'].max()
    print(f"📅 Latest Gameweek in Data: GW{latest_gw}")
else:
    print("⚠️ 'gameweek' column not found in df_clean.")
    latest_gw = None

# 3️⃣ Prepare team map
if 'teams' in globals() and not teams.empty:
    team_map = teams[['team_id', 'name', 'short_name']].rename(columns={'team_id': 'team_code'})
else:
    print("⚠️ 'teams' DataFrame not found.")
    team_map = pd.DataFrame()

# 4️⃣ Show fixtures
if fixtures_df is not None and latest_gw and not team_map.empty:
    gw_fixtures = (
        fixtures_df[fixtures_df['gameweek'] == latest_gw][['home_team', 'away_team']]
        .merge(team_map, left_on='home_team', right_on='team_code', how='left')
        .rename(columns={'name': 'Home Team Name', 'short_name': 'Home Short Name'})
        .merge(team_map, left_on='away_team', right_on='team_code', how='left')
        .rename(columns={'name': 'Away Team Name', 'short_name': 'Away Short Name'})
        [['home_team', 'Home Team Name', 'Home Short Name', 'away_team', 'Away Team Name', 'Away Short Name']]
    )

  print(f"\n📋 Fixtures for Gameweek {latest_gw}:")
  print(gw_fixtures)
else:
    print("⚠️ Could not generate fixtures — fixture or team mapping missing.")
for name, obj in globals().items():
    if isinstance(obj, pd.DataFrame):
        print(name, list(obj.columns)[:10])

# ===============================================
# 🏟️ SHOW FIXTURES FOR LATEST GAMEWEEK (FINAL CLEAN VERSION)
# ===============================================

import pandas as pd
import requests

# ===============================================
# 1️⃣ LOAD FIXTURES FROM FPL API
# ===============================================
print("📥 Fetching fixtures from FPL API...")

url = "https://fantasy.premierleague.com/api/fixtures/"
response = requests.get(url)

if response.status_code == 200:
    fixtures_df = pd.DataFrame(response.json())

    # Rename to match our logic
    fixtures_df = fixtures_df.rename(columns={
        'team_h': 'home_team',
        'team_a': 'away_team',
        'event': 'gameweek'
    })

    print("✅ Fixtures loaded successfully!")
else:
    print("❌ Failed to fetch fixtures")
    fixtures_df = None

# ===============================================
# 2️⃣ TEAM MAPPING
# ===============================================
if 'teams' in globals() and not teams.empty:
    team_map = teams[['team_id', 'name', 'short_name']]\
        .rename(columns={'team_id': 'team_code'})

    print("\n📘 Team Mapping Loaded")
else:
    print("❌ 'teams' dataframe not found")
    team_map = pd.DataFrame()


# ===============================================
# 3️⃣ GET LATEST GAMEWEEK
# ===============================================

if fixtures_df is not None:

    # Filter only upcoming fixtures
    upcoming_fixtures = fixtures_df[fixtures_df['finished'] == False]

    if not upcoming_fixtures.empty:
        latest_gw = upcoming_fixtures['gameweek'].min()
    else:
        print("⚠️ No upcoming fixtures found")
        latest_gw = fixtures_df['gameweek'].max()

else:
    latest_gw = None

print(f"📅 Next Gameweek: GW{latest_gw}")
# ===============================================
# 4️⃣ GENERATE FIXTURES TABLE
# ===============================================
if fixtures_df is not None and latest_gw is not None and not team_map.empty:

    gw_fixtures = (
        fixtures_df[fixtures_df['gameweek'] == latest_gw][['home_team', 'away_team']]
        .merge(team_map, left_on='home_team', right_on='team_code', how='left')
        .rename(columns={
            'name': 'Home Team',
            'short_name': 'Home Short'
        })
        .merge(team_map, left_on='away_team', right_on='team_code', how='left')
        .rename(columns={
            'name': 'Away Team',
            'short_name': 'Away Short'
        })
        [['Home Team', 'Home Short', 'Away Team', 'Away Short']]
        .sort_values('Home Team')
        .reset_index(drop=True)
    )

    print(f"\n📋 Fixtures for Gameweek {latest_gw}:")
    print(gw_fixtures)

else:
    print("⚠️ Could not generate fixtures — missing data.")


# ===============================================
# 5️⃣ DEBUG (OPTIONAL)
# ===============================================
print("\n🔍 Available DataFrames:")
for name, obj in globals().items():
    if isinstance(obj, pd.DataFrame):
        print(name, list(obj.columns)[:10])
# ======================================================
# 🧹 Clean duplicates before analysis
# ======================================================

# Keep only the most recent gameweek entry per player
df_latest = (
    df_clean.sort_values(['player_id', 'gameweek'], ascending=[True, False])
    .drop_duplicates(subset='player_id', keep='first')
)

# Compute value for money again (in case we dropped some columns)
df_latest['value_for_money'] = df_latest['predicted_next_points'] / (df_latest['now_cost_gw'] / 10)

# ======================================================
# 1️⃣ Top 5 Players by Predicted Points per Position
# ======================================================
top5_predicted = (
    df_latest[['player_id', 'full_name', 'position', 'team_name_final', 'predicted_next_points']]
    .dropna(subset=['predicted_next_points'])
    .sort_values(['position', 'predicted_next_points'], ascending=[True, False])
    .groupby('position', group_keys=False)
    .head(5)
)

print("⚡ Top 5 Players by Predicted Points per Position:")
print(top5_predicted)

# ======================================================
# 2️⃣ Top 5 Players by Value for Money
# ======================================================
top5_vfm = (
    df_latest[['player_id', 'full_name', 'position', 'team_name_final', 'value_for_money']]
    .dropna(subset=['value_for_money'])
    .sort_values(['position', 'value_for_money'], ascending=[True, False])
    .groupby('position', group_keys=False)
    .head(5)
)

print("\n💸 Top 5 Players by Value-for-Money per Position:")
print(top5_vfm)
# ======================================================
# 3️⃣ All Teams by Difficulty (Average Opponent Difficulty)
# ======================================================
team_difficulty = (
    df_latest.groupby('team_name_final')['opp_difficulty_proxy']
    .mean()
    .reset_index(name='avg_opponent_difficulty')
    .sort_values('avg_opponent_difficulty', ascending=True)
)

print("\n🧱 All Teams by Opponent Difficulty (lower = easier):")
print(team_difficulty)

# ======================================================
# 4️⃣ All Teams by Total Predicted Points
# ======================================================
team_predicted_points = (
    df_latest.groupby('team_name_final')['predicted_next_points']
    .sum()
    .reset_index(name='total_predicted_points')
    .sort_values('total_predicted_points', ascending=False)
)

print("\n⚽ All Teams by Total Predicted Points:")
print(team_predicted_points)

# Find all unique gameweeks present
unique_gws = sorted(df_clean['gameweek'].dropna().unique())

print(f"📅 Gameweeks in dataset: {unique_gws}")

# Find the latest (most recent) gameweek
latest_gw = df_clean['gameweek'].max()
print(f"✅ Latest gameweek in your dataset: GW{int(latest_gw)}")

# If you kept df_latest from the last code:
print(f"🧠 Predictions are based on latest available data — most likely for GW{int(latest_gw + 1)} (the next gameweek).")

df_clean.to_parquet("final_data.parquet", index=False)
print("✅ Saved final_data.parquet")
