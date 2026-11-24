from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_models():

    models = {

        "Linear Regression": LinearRegression(),

        "Ridge": Ridge(),

        "Lasso": Lasso(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        ),

        "LightGBM": LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),

        "CatBoost": CatBoostRegressor(
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
    }

    return models