import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class FeatureReductor:
    def __init__(self, method='pca', n_components=2, random_state=42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.reductor = None
        self.is_fitted = False
        
        if method == 'pca':
            self.reductor = PCA(n_components=n_components, random_state=random_state)
        elif method == 'lda':
            self.reductor = LinearDiscriminantAnalysis(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'lda':
            self.reductor.fit(X_scaled, y)
        else:
            self.reductor.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Reductor must be fitted before transformation")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.reductor.transform(X_scaled)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_explained_variance(self):
        """Get explained variance ratio for PCA.
        
        Returns:
            Array of explained variance ratios or None for LDA.
        """
        if self.method == 'pca' and self.is_fitted:
            return self.reductor.explained_variance_ratio_
        return None
    
    def get_components(self):
        if not self.is_fitted:
            raise RuntimeError("Reductor must be fitted first")
        
        if self.method == 'pca':
            return self.reductor.components_
        elif self.method == 'lda':
            return self.reductor.scalings_


def visualize_tsne(X, y, n_components=2, perplexity=30, random_state=42, 
                   figsize=(10, 8), title="t-SNE Visualization", 
                   class_names=None, save_path=None):
    """Visualize high-dimensional data in 2D using t-SNE.
    
    Args:
        X: Feature array or DataFrame.
        y: Labels for coloring points.
        n_components: Number of dimensions (default 2 for 2D visualization).
        perplexity: t-SNE perplexity parameter (5-50 typical range).
        random_state: Random seed for reproducibility.
        figsize: Figure size as (width, height).
        title: Plot title.
        class_names: Dictionary mapping label values to names.
        save_path: Path to save figure (if None, displays instead).
    
    Returns:
        Dictionary containing t-SNE embeddings and figure object.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if isinstance(y, pd.Series):
        y = y.values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000
    )
    X_embedded = tsne.fit_transform(X_scaled)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = y == label
        label_name = class_names[label] if class_names else f"Class {label}"
        
        ax.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=[colors[idx]],
            label=label_name,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )
    
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return {
        'embeddings': X_embedded,
        'labels': y,
        'figure': fig,
        'axis': ax
    }


def visualize_pca_variance(reductor, figsize=(10, 6), save_path=None):
    """Visualize PCA explained variance.
    
    Args:
        reductor: Fitted FeatureReductor with PCA method.
        figsize: Figure size as (width, height).
        save_path: Path to save figure (if None, displays instead).
    
    Returns:
        Dictionary containing variance information and figure.
    """
    if reductor.method != 'pca':
        raise ValueError("This function only works with PCA reductor")
    
    if not reductor.is_fitted:
        raise RuntimeError("Reductor must be fitted first")
    
    explained_variance = reductor.get_explained_variance()
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Variance Explained by Each Component")
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return {
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'figure': fig
    }
