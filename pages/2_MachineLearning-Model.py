import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def svm_model():
    # Loader
    @st.cache_resource
    def load_model_and_scaler():
        model_path = "data/machine_learning/model1_SVM/svm_model.pkl"
        scaler_path = "data/machine_learning/model1_SVM/scaler.pkl"
        
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model or scaler: {e}")
            return None, None

    # Load the training data
    @st.cache_data
    def load_training_data():
        try:
            data_path = "data/machine_learning/train_data.csv"
            train_data = pd.read_csv(data_path)
            return train_data
        except Exception as e:
            st.error(f"Error loading training data: {e}")
            return None

    model, scaler = load_model_and_scaler()
    train_data = load_training_data()

    st.title("Penguin Species Classifier (SVM)")
    st.write("Enter the penguin features to predict its species")

    st.header("Input Features")

    col1, col2 = st.columns(2)

    with col1:
        body_mass = st.number_input("Body Mass (g)", min_value=2000.0, max_value=6500.0, value=4000.0, step=100.0)
        delta_15n = st.number_input("Delta 15 N (o/oo)", min_value=7.0, max_value=10.0, value=8.5, step=0.1)
        delta_13c = st.number_input("Delta 13 C (o/oo)", min_value=-28.0, max_value=-22.0, value=-25.0, step=0.1)
        culmen_length = st.number_input("Culmen Length (mm)", min_value=30.0, max_value=60.0, value=45.0, step=0.5)

    with col2:
        culmen_depth = st.number_input("Culmen Depth (mm)", min_value=10.0, max_value=22.0, value=15.0, step=0.5)
        flipper_length = st.number_input("Flipper Length (mm)", min_value=170.0, max_value=240.0, value=200.0, step=1.0)
        sex = st.selectbox("Sex", options=["MALE", "FEMALE"], index=0)
        clutch_completion = st.selectbox("Clutch Completion", options=["Yes", "No"], index=0)

    # Mappign categorical
    sex_numeric = 1 if sex == "MALE" else 0
    clutch_completion_numeric = 1 if clutch_completion == "Yes" else 0

    # Prepare input data for prediction
    def prepare_input_data():
        input_data = pd.DataFrame({
            'Body Mass (g)': [body_mass],
            'Delta 15 N (o/oo)': [delta_15n],
            'Delta 13 C (o/oo)': [delta_13c],
            'Culmen Length (mm)': [culmen_length],
            'Culmen Depth (mm)': [culmen_depth],
            'Flipper Length (mm)': [flipper_length],
            'Sex': [sex_numeric],
            'Clutch Completion': [clutch_completion_numeric]
        })
        return input_data

    # create visualizations
    def create_visualization(input_data, predicted_species):
        if train_data is None:
            st.error("Cannot create visualization: Training data not loaded")
            return
        
        st.header("Data Visualization")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization type:",
            ["PCA Projection", "Feature Relationships"]
        )
        
        if viz_type == "PCA Projection":
            feature_cols = input_data.columns.tolist()
            
            if 'Species' not in train_data.columns:
                st.error("Species column not found in training data")
                return
                
            X_train = train_data[feature_cols]
            y_train = train_data['Species']
            
            combined = pd.concat([X_train, input_data])

            if scaler is not None:
                combined_scaled = scaler.transform(combined)
            else:
                combined_scaled = (combined - combined.mean()) / combined.std()
                
            X_train_scaled = combined_scaled[:-1]
            input_scaled = combined_scaled[-1].reshape(1, -1)
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train_scaled)
            input_pca = pca.transform(input_scaled)
            
            # Create PCA plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a color map
            species_list = y_train.unique()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            species_colors = {species: color for species, color in zip(species_list, colors)}
            
            # Plot training data
            for species in species_list:
                mask = y_train == species
                ax.scatter(
                    X_train_pca[mask, 0], 
                    X_train_pca[mask, 1],
                    label=species,
                    alpha=0.7
                )
            
            # Plot the input data point with a star marker
            ax.scatter(
                input_pca[0, 0],
                input_pca[0, 1],
                marker='*',
                color='black',
                s=200,
                label='Input'
            )
            
            ax.set_title('PCA Projection of Penguin Data')
            ax.set_xlabel('Principal Component I')
            ax.set_ylabel('Principal Component II')
            ax.legend()
            
            st.pyplot(fig)
            
            explained_variance = pca.explained_variance_ratio_
            st.write(f"Explained variance: PC1 {explained_variance[0]:.2f}, PC2 {explained_variance[1]:.2f}")
            
        elif viz_type == "Feature Relationships":
            st.subheader("Select features to plot")
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature", input_data.columns.tolist(), index=3)  # Default to Culmen Length
            with col2:
                y_feature = st.selectbox("Y-axis feature", input_data.columns.tolist(), index=4)  # Default to Culmen Depth
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot training data
            species_list = train_data['Species'].unique()
            for species in species_list:
                mask = train_data['Species'] == species
                ax.scatter(
                    train_data.loc[mask, x_feature], 
                    train_data.loc[mask, y_feature],
                    label=species,
                    alpha=0.7
                )
            
            # Plot the input data point with a star marker
            ax.scatter(
                input_data[x_feature].values[0],
                input_data[y_feature].values[0],
                marker='*',
                color='black',
                s=200,
                label='Input'
            )
            
            # Add predicted species information to title
            if predicted_species:
                ax.set_title(f'Feature Relationship (Predicted: {predicted_species})')
            else:
                ax.set_title('Feature Relationship')
                
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.legend()
            
            st.pyplot(fig)


    # Make prediction when button is clicked
    if st.button("Predict Species"):
        if model is not None and scaler is not None:
            input_data = prepare_input_data()
            # Scale the input data using the loaded scaler
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(scaled_data)
            
            # Get unique species from training data if available
            if train_data is not None and 'Species' in train_data.columns:
                unique_species = train_data['Species'].unique()
                species_mapping = {i: species for i, species in enumerate(unique_species)}
            else:
                # Fallback mapping
                species_mapping = {
                    "Adelie Penguin (Pygoscelis adeliae)": "Adelie Penguin (Pygoscelis adeliae)",
                    "Chinstrap penguin (Pygoscelis antarctica)": "Chinstrap penguin (Pygoscelis antarctica)",
                    "Gentoo penguin (Pygoscelis papua)": "Gentoo penguin (Pygoscelis papua)"
                }
                # Also try with string-based class labels
                for val in ["Adelie", "Chinstrap", "Gentoo"]:
                    species_mapping[val] = val
            
            # Get predicted species - handle different prediction output types
            predicted_species = None
            
            # If prediction is an array of indices
            if isinstance(prediction, np.ndarray) and prediction.dtype.kind in 'iuf':
                predicted_idx = prediction[0]
                if predicted_idx in species_mapping:
                    predicted_species = species_mapping[predicted_idx]
            
            # If prediction is a string (class name directly)
            elif isinstance(prediction[0], str):
                pred_string = prediction[0]
                # Try direct mapping or find closest match
                if pred_string in species_mapping:
                    predicted_species = species_mapping[pred_string]
                else:
                    # Try to find matching species by substring
                    for key, value in species_mapping.items():
                        if isinstance(key, str) and pred_string in key:
                            predicted_species = value
                            break
                        elif isinstance(value, str) and pred_string in value:
                            predicted_species = value
                            break
            
            # If no mapping worked, use the prediction as-is
            if predicted_species is None:
                predicted_species = str(prediction[0])
            
            # Display the prediction
            st.success(f"Predicted Species: {predicted_species}")
            
            # Create visualization
            create_visualization(input_data, predicted_species)
        else:
            st.error("Model or scaler could not be loaded. Please check the file paths.")
    else:
        # Show visualization without prediction highlight
        if train_data is not None:
            input_data = prepare_input_data()
            create_visualization(input_data, None)

def rf_model():
    
    @st.cache_resource
    def load_model_and_features():
        model_path = "data/machine_learning/model2_RF/rf_model.pkl"
        features_path = "data/machine_learning/model2_RF/feature_data.pkl"
        
        try:
            model = joblib.load(model_path)
            features = joblib.load(features_path)
            return model, features
        except Exception as e:
            st.error(f"Error loading model or features: {e}")
            return None, None

    @st.cache_data
    def load_training_data():
        try:
            df = pd.read_csv("data/machine_learning/penguins_lter_cleaned.csv")
            return df
        except Exception as e:
            st.error(f"Error loading training data: {e}")
            return None

    model, features = load_model_and_features()
    train_data = load_training_data()

    st.title("Penguin Body Mass Predictor (Random Forest)")
    st.write("Enter the penguin features to predict its body mass")

    st.header("Input Features")

    categorical_feat = features['categorical_feat'] if features else ['Species', 'Sex', 'Island', 'Clutch Completion']
    numerical_feat = features['numerical_feat'] if features else ['Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 
                                                                 'Culmen Length (mm)', 'Culmen Depth (mm)', 
                                                                 'Flipper Length (mm)']

    col1, col2 = st.columns(2)

    # Categorical feature
    with col1:
        st.subheader("Categorical Features")
        species = st.selectbox("Species", options=["Adelie", "Chinstrap", "Gentoo"], index=0)
        sex = st.selectbox("Sex", options=["MALE", "FEMALE"], index=0)
        island = st.selectbox("Island", options=["Biscoe", "Dream", "Torgersen"], index=0)
        clutch_completion = st.selectbox("Clutch Completion", options=["Yes", "No"], index=0)

    # Numerical feature
    with col2:
        st.subheader("Numerical Features")
        delta_15n = st.number_input("Delta 15 N (o/oo)", min_value=7.0, max_value=10.0, value=8.5, step=0.1)
        delta_13c = st.number_input("Delta 13 C (o/oo)", min_value=-28.0, max_value=-22.0, value=-25.0, step=0.1)
        culmen_length = st.number_input("Culmen Length (mm)", min_value=30.0, max_value=60.0, value=45.0, step=0.5)
        culmen_depth = st.number_input("Culmen Depth (mm)", min_value=10.0, max_value=22.0, value=15.0, step=0.5)
        flipper_length = st.number_input("Flipper Length (mm)", min_value=170.0, max_value=240.0, value=200.0, step=1.0)

    # Prepare input
    def prepare_input_data():
        input_data = pd.DataFrame({
            'Species': [species],
            'Sex': [sex],
            'Island': [island],
            'Clutch Completion': [clutch_completion],
            'Delta 15 N (o/oo)': [delta_15n],
            'Delta 13 C (o/oo)': [delta_13c],
            'Culmen Length (mm)': [culmen_length],
            'Culmen Depth (mm)': [culmen_depth],
            'Flipper Length (mm)': [flipper_length]
        })
        
        # Ensure categorical columns
        for col in categorical_feat:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype('object')
                
        # Ensure numerical columns
        for col in numerical_feat:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype('float64')
                
        return input_data

    # Create visualizations
    def create_visualization(input_data, predicted_mass):
        if train_data is None:
            st.error("Cannot create visualization: Training data not loaded")
            return
        
        st.header("Data Visualization")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Regression Plot", "Feature Relationships", "Mass Distribution"]
        )
        
        if viz_type == "Regression Plot":
            if model is not None:
                try:
                    train_data_copy = train_data.copy()
                    
                    for col in categorical_feat:
                        if col in train_data_copy.columns:
                            train_data_copy[col] = train_data_copy[col].astype('object')
                    
                    for col in numerical_feat:
                        if col in train_data_copy.columns:
                            train_data_copy[col] = train_data_copy[col].astype('float64')
                    
                    X = train_data_copy[categorical_feat + numerical_feat]
                    y_true = train_data_copy['Body Mass (g)']
                    
                    sample_size = min(500, len(X))
                    sample_indices = np.random.choice(len(X), sample_size, replace=False)
                    X_sample = X.iloc[sample_indices]
                    y_true_sample = y_true.iloc[sample_indices]
                    
                    preprocessor = model.named_steps['preprocessor']
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    X_transformed = preprocessor.transform(X_sample)
                    
                    y_pred_sample = model.predict(X_sample)
                    
                    ax.scatter(y_true_sample, y_pred_sample, alpha=0.5, label='Training Data')
                    
                    min_val = min(min(y_true_sample), min(y_pred_sample))
                    max_val = max(max(y_true_sample), max(y_pred_sample))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                    
                    # Plot predicted point if available
                    if predicted_mass:
                        # Find similar penguins in training data
                        species_mask = train_data_copy['Species'] == species
                        
                        if species_mask.any():
                            # Get average actual mass
                            similar_mass_avg = train_data_copy.loc[species_mask, 'Body Mass (g)'].mean()
                            
                            # Plot the prediction
                            ax.scatter(similar_mass_avg, predicted_mass, color='red', s=200, marker='*', 
                                        label=f'Your Prediction: {predicted_mass:.1f} g')
                            
                            # Add annotation
                            ax.annotate(f"Prediction: {predicted_mass:.1f} g",
                                        (similar_mass_avg, predicted_mass),
                                        xytext=(10, 10), textcoords='offset points',
                                        arrowprops=dict(arrowstyle='->', color='black'))
                    
                    ax.set_xlabel('Actual Body Mass (g)')
                    ax.set_ylabel('Predicted Body Mass (g)')
                    ax.set_title('Actual vs Predicted Body Mass')
                    ax.legend()
                    
                    # Calculate and display metrics
                    mse = mean_squared_error(y_true_sample, y_pred_sample)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true_sample, y_pred_sample)
                    r2 = r2_score(y_true_sample, y_pred_sample)
                    
                    st.pyplot(fig)
                    
                    # Show metrics in columns
                    col1, col2, col3 = st.columns(3)
                    col2.metric("RMSE", f"{rmse:.2f}")
                    col1.metric("MAE", f"{mae:.2f}")
                    col3.metric("R² Score", f"{r2:.3f}")
                        
                
                except Exception as e:
                    st.error(f"Error creating regression plot: {e}")
                    
                    # Create a simple mock plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, f"Cannot create regression plot: {str(e)}", 
                            ha='center', va='center', fontsize=14)
                    ax.set_axis_off()
                    st.pyplot(fig)
            else:
                st.error("Model is not loaded.")
        
        elif viz_type == "Feature Relationships":
            st.subheader("Select features to plot")
            
            # numerical features
            available_num_feat = [f for f in numerical_feat if f in train_data.columns]
            if not available_num_feat:
                st.warning("No numerical features found in training data")
                return
                
            # categorical features for coloring
            available_cat_feat = [f for f in categorical_feat if f in train_data.columns]
            if not available_cat_feat:
                available_cat_feat = [None]  # If no categorical
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature", available_num_feat, 
                                        index=min(2, len(available_num_feat)-1))
            with col2:
                y_feature = st.selectbox("Y-axis feature", available_num_feat, 
                                        index=min(3, len(available_num_feat)-1))
            
            if available_cat_feat[0] is not None:
                color_by = st.selectbox("Color by", available_cat_feat, index=0)
            else:
                color_by = None
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                # Check if selected features exist in input_data
                x_in_input = x_feature in input_data.columns
                y_in_input = y_feature in input_data.columns
                
                # Plot training data
                if color_by:
                    scatter = sns.scatterplot(
                        data=train_data,
                        x=x_feature,
                        y=y_feature,
                        hue=color_by,
                        alpha=0.7,
                        ax=ax
                    )
                else:
                    scatter = sns.scatterplot(
                        data=train_data,
                        x=x_feature,
                        y=y_feature,
                        alpha=0.7,
                        ax=ax
                    )
                
                # Plot the input data point
                if x_in_input and y_in_input:
                    ax.scatter(
                        input_data[x_feature].values[0],
                        input_data[y_feature].values[0],
                        marker='*',
                        color='black',
                        s=200,
                        label='Input'
                    )
                
                # Add predicted mass information
                if predicted_mass:
                    ax.set_title(f'Feature Relationship (Predicted Mass: {predicted_mass:.1f} g)')
                else:
                    ax.set_title('Feature Relationship')
                    
                ax.set_xlabel(x_feature)
                ax.set_ylabel(y_feature)
                
                if x_in_input and y_in_input:
                    handles, labels = ax.get_legend_handles_labels()
                    handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=15))
                    ax.legend(handles=handles, labels=labels)
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating feature relationship plot: {e}")
            
        elif viz_type == "Mass Distribution":
            if 'Body Mass (g)' not in train_data.columns:
                st.warning("Training data doesn't have 'Body Mass (g)' column")
                return
                
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(train_data['Body Mass (g)'], bins=20, kde=True, ax=ax)
                
                if predicted_mass:
                    ax.axvline(x=predicted_mass, color='red', linestyle='--', linewidth=2, 
                               label=f'Predicted: {predicted_mass:.1f} g')
                    
                    ax.text(predicted_mass, ax.get_ylim()[1]*0.9, f'Prediction: {predicted_mass:.1f} g', 
                            rotation=90, color='red', ha='right')
                    ax.legend()
                
                if 'Species' in train_data.columns and species:
                    species_mask = train_data['Species'] == species
                    if species_mask.any():
                        species_data = train_data.loc[species_mask, 'Body Mass (g)']
                        sns.kdeplot(species_data, ax=ax, color='green', linestyle='-', 
                                    label=f'{species} Mass Distribution')
                        ax.legend()
                
                ax.set_title('Distribution of Penguin Body Mass')
                ax.set_xlabel('Body Mass (g)')
                ax.set_ylabel('Frequency')
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating mass distribution plot: {e}")

    if st.button("Predict Body Mass"):
        if model is not None:
            input_data = prepare_input_data()
            
            try:
                prediction = model.predict(input_data)
                
                # Display prediction
                predicted_mass = float(prediction[0])
                st.success(f"Predicted Body Mass: {predicted_mass:.1f} grams")
                
                # visualization
                create_visualization(input_data, predicted_mass)
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("This could be due to mismatched features between the model and input data.")
                
                # Show detailed error info in an expander
                with st.expander("Detailed Error Information (for debugging)"):
                    st.write(f"Error type: {type(e).__name__}")
                    st.write(f"Error message: {str(e)}")
                    st.write("Input data types:")
                    st.write(input_data.dtypes)
                    if hasattr(model, 'named_steps'):
                        st.write("Model pipeline steps:", list(model.named_steps.keys()))
        else:
            st.error("Model could not be loaded. Please check the file paths.")
    else:
        # without prediction highlight
        if train_data is not None:
            input_data = prepare_input_data()
            create_visualization(input_data, None)

def main():

    st.set_page_config(page_title="Machine learning Model", page_icon=":penguin:")

    st.title("Penguin Dataset :penguin:")
    
    st.markdown(""" --- """)

    st.markdown("""
    ### Models
    """)
    choice = st.radio("Select the option below to show Model", ["SVM Model", "RF Model"])

    st.markdown(""" --- """)

    if choice == "SVM Model":
        svm_model()
    elif choice == "RF Model":
        rf_model()

if __name__ == "__main__":
    main()
