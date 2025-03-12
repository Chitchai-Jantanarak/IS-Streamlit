import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.append('..')
from EDA import data_cleansing


def build_svm_model():
    # Load and prepare data
    df = data_cleansing()
    
    data_prep = df[['Body Mass (g)', 'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 
                   'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 
                   'Sex', 'Clutch Completion', 'Species']]
    data_prep = pd.get_dummies(data_prep, columns=['Species'])  # One-hot encoding
    

    X = data_prep.drop(columns=['Species_Adelie Penguin (Pygoscelis adeliae)', 
                               'Species_Chinstrap penguin (Pygoscelis antarctica)', 
                               'Species_Gentoo penguin (Pygoscelis papua)'])
    y = df['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Save
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return svm_model, scaler, X_train, X_test, y_train, y_test, X, y

def main():
    build_svm_model()
    print("SVM model built and saved successfully!")

if __name__ == "__main__":
    main()