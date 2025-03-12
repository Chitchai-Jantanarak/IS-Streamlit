import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import sys
sys.path.append('..')
from EDA import data_cleansing

def build_rf_model():
    # Load and prepare data
    df = data_cleansing()

    categorical_feat = ['Species', 'Sex', 'Island', 'Clutch Completion']
    numerical_feat   = ['Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 'Culmen Length (mm)', 
                        'Culmen Depth (mm)', 'Flipper Length (mm)']
    
    X = df[categorical_feat + numerical_feat]
    y = df['Body Mass (g)']

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_feat),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feat)
    ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=275, 
            max_features='log2', 
            max_depth=12, 
            min_samples_split=7,
            min_samples_leaf=2, 
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf_pipeline.fit(X_train, y_train)

    # Save feature for sending to another files
    feat_datas = {
        'numerical_feat': numerical_feat,
        'categorical_feat': categorical_feat
    }

    # Save
    joblib.dump(rf_pipeline, 'rf_model.pkl')
    joblib.dump(feat_datas, 'feature_data.pkl')

    return rf_pipeline, X_train, X_test, y_train, y_test, X, y