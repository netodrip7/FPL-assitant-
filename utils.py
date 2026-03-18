def search_player(df, name):
    return df[df['web_name_gw'].str.contains(name, case=False, na=False)]

def get_replacements(df, player_name):
    player = df[df['web_name_gw'].str.contains(player_name, case=False)].head(1)
    if player.empty:
        return None

    pos = player.iloc[0]['position']

    candidates = df[df['position'] == pos]
    candidates = candidates[candidates['player_id'] != player.iloc[0]['player_id']]

    candidates['score'] = (
        candidates['predicted_next_points'] * 0.5 +
        candidates['value_for_money'] * 0.3 +
        candidates['form_gw'] * 0.2
    )

    return candidates.sort_values('score', ascending=False).head(3)

def captain_pick(df):
    return df.sort_values('predicted_next_points', ascending=False).head(1)
