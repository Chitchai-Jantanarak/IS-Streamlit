import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


# Caller function
def visualize_pca_and_decision_boundary(X=None, y=None, model=None, scaler=None):

    if X is None or y is None or model is None:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_scaled = scaler.transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    species_names = ['Adelie', 'Chinstrap', 'Gentoo']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    

    for i, species in enumerate(np.unique(y_encoded)):
        idx = y_encoded == species
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=colors[i],
                   s=80,
                   marker=markers[i],
                   edgecolor='k',
                   linewidth=1,
                   alpha=0.8,
                   label=species_names[i])
    
    plt.title('Penguin Species in PCA Space', fontsize=14)
    plt.xlabel('Principal Component I')
    plt.ylabel('Principal Component II')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    var_ratio = pca.explained_variance_ratio_
    var_text = f'PC1: {var_ratio[0]:.2%} var\nPC2: {var_ratio[1]:.2%} var'
    plt.annotate(var_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300)
    plt.close()

    # Boundary plot
    plot_decision_boundary(X_scaled, y_encoded, model, pca)
    
    # Create PCA loadings
    plot_pca_loadings(pca, X)

def plot_decision_boundary(X_scaled, y_encoded, model, pca):

    X_pca = pca.transform(X_scaled)
    
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_pca, y_encoded)
    
    # Create a mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # meshgrid prediction
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    species_names = ['Adelie', 'Chinstrap', 'Gentoo']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Get unique numeric classes (0, 1, 2)
    for i in range(3):  # 3 Species
        idx = y_encoded == i
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                  c=colors[i],
                  s=80,
                  marker=markers[i],
                  edgecolor='k',
                  linewidth=1,
                  alpha=0.8,
                  label=species_names[i])
    
    ax.set_title("SVM Decision Boundary (PCA-reduced Data)", fontsize=16)
    ax.set_xlabel(f'Principal Component 1', fontsize=14)
    ax.set_ylabel(f'Principal Component 2', fontsize=14)

    fig.colorbar(contour, ax=ax, label='Class')
    
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    kernel_name = svm.kernel.upper() if hasattr(svm, 'kernel') else 'Unknown'
    c_value = svm.C if hasattr(svm, 'C') else 'Unknown'
    
    textstr = f'Model: SVM ({kernel_name} kernel)\nC parameter: {c_value}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    if hasattr(pca, 'explained_variance_ratio_'):
        var_ratio = pca.explained_variance_ratio_
        var_text = f'PC1: {var_ratio[0]:.2%} variance\nPC2: {var_ratio[1]:.2%} variance'
        ax.text(0.05, 0.85, var_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('svm_decision_boundary.png', dpi=300)
    plt.close()

def plot_pca_loadings(pca, X):

    # Plot PCA feature as a heatmap

    # Get feature names from X
    feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
    
    # PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Contributions to Principal Components', fontsize=14)
    plt.tight_layout()
    plt.savefig('pca_feature_contributions.png', dpi=300)
    plt.close()
