def get_player_prediction(df, player_input):
    result = df[df['web_name_gw'].str.contains(player_input, case=False, na=False)]
    return result[['web_name_gw', 'predicted_next_points']].sort_values(
        'predicted_next_points', ascending=False
    )
