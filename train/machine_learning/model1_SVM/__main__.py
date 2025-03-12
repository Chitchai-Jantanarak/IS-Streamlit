import numpy as np
import pandas as pd

# Import modules
from builder_model import build_svm_model
from evaluation_model import evaluate_model
from PCA_model import visualize_pca_and_decision_boundary


def main():

    # Build SVM model
    model, scaler, X_train, X_test, y_train, y_test, X, y = build_svm_model()
    
    # Evaluate model
    evaluate_model(model=model, scaler=scaler, X_test=X_test, y_test=y_test)
    
    # Visualizations
    visualize_pca_and_decision_boundary(
        X=X, y=y, model=model, scaler=scaler
    )

if __name__ == "__main__":
    main()