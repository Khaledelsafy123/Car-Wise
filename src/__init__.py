import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)


def evaluate_regression_model(model, X, y):
    y_pred = model.predict(X)

    metrics = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred),
        "Median AE": median_absolute_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred) * 100,
        "Explained Variance": explained_variance_score(y, y_pred)
    }

    return metrics, y_pred