import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from builder_model import build_rf_model


def evaluate_model(model=None, X_test=None, X_train=None, y_test=None,  y_train=None):

    if (    
        model is None or 
        X_test is None or 
        X_train is None or 
        y_test is None or 
        y_train is None
    ):
        try:
            model = joblib.load('svm_model.pkl')
            _, X_train, X_test, y_train, y_test, _, _ = build_rf_model()
        except FileNotFoundError:
            model, X_train, X_test, y_train, y_test, _, _ = build_rf_model()


    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, scoring='neg_root_mean_squared_error'
    )

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("-----------------------------")
    print("Model performance on test set:")
    print(f"Root Mean Squared Error: {rmse:.2f} grams")
    print(f"Mean Absolute Error: {mae:.2f} grams")
    print(f"R² Score: {r2:.4f}")
    print("-----------------------------")

    return y_pred, rmse, mae, r2, cv_scores

