import numpy as np
from xgboost import XGBRegressor

def train_model(df_clean, feature_cols):

    df_model = df_clean[
        df_clean['next_gw_points'].notnull() &
        (~df_clean['next_gw_points'].isin([np.inf, -np.inf]))
    ]

    # ONLY numeric features
    X = df_model[feature_cols].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df_model['next_gw_points'].astype(float)

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)

    return model
