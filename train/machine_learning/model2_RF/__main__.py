import sys
sys.path.append('..')
from EDA import data_cleansing

from builder_model import build_rf_model
from crossval import evaluate_model
from visualize_model import visualize

def main():
    
    # Build 
    model, X_train, X_test, y_train, y_test, X, y = build_rf_model()
    
    # Evaluate
    y_pred, rmse, mae, r2, cv_scores = evaluate_model(model, X_test, X_train, y_test, y_train)
    
    # Visualize 
    visualize(model, X, y, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()