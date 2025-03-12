import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def evaluate_model(model=None, scaler=None, X_test=None, y_test=None):

    if model is None or X_test is None or y_test is None:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
    
    # Scale test data
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test  # Take an examine (IF NOT SCALER)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Adelie', 'Chinstrap', 'Gentoo'], 
                yticklabels=['Adelie', 'Chinstrap', 'Gentoo'])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    return accuracy, conf_matrix, class_report