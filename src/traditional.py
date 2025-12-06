import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from .logger_config import setup_logger

logger = setup_logger("traditional")


class Eigenfaces:
    def __init__(self, n_components=150):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=True)
        self.X_train_pca = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        logger.info(f"Training Eigenfaces (n_components={self.n_components})...")
        n_samples, h, w = X_train.shape
        X_train_flat = X_train.reshape(n_samples, -1)
        self.X_train_pca = self.pca.fit_transform(X_train_flat)
        self.y_train = y_train
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"Explained variance ratio: {explained_var:.4f}")
        
    def predict(self, X_test):
        n_samples, h, w = X_test.shape
        X_test_flat = X_test.reshape(n_samples, -1)
        X_test_pca = self.pca.transform(X_test_flat)
        
        y_pred = []
        for test_sample in X_test_pca:
            distances = np.linalg.norm(self.X_train_pca - test_sample, axis=1)
            nearest_idx = np.argmin(distances)
            y_pred.append(self.y_train[nearest_idx])
        return np.array(y_pred)
    
    def get_eigenfaces(self, n_faces=10):
        components = self.pca.components_[:n_faces]
        h = w = int(np.sqrt(components.shape[1]))
        eigenfaces = components.reshape(n_faces, h, w)
        return eigenfaces


class Fisherfaces:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = PCA(n_components=0.98, whiten=True)
        self.lda = None
        self.X_train_lda = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        logger.info("Training Fisherfaces...")
        n_samples, h, w = X_train.shape
        X_train_flat = X_train.reshape(n_samples, -1)
        
        logger.info("Step 1: PCA dimensionality reduction...")
        X_train_pca = self.pca.fit_transform(X_train_flat)
        logger.info(f"PCA output dim: {X_train_pca.shape[1]}")
        
        logger.info("Step 2: LDA dimensionality reduction...")
        n_classes = len(np.unique(y_train))
        if self.n_components is None:
            self.n_components = min(n_classes - 1, X_train_pca.shape[1])
        
        self.lda = LDA(n_components=self.n_components)
        self.X_train_lda = self.lda.fit_transform(X_train_pca, y_train)
        self.y_train = y_train
        logger.info(f"LDA output dim: {self.X_train_lda.shape[1]}")
        
    def predict(self, X_test):
        n_samples, h, w = X_test.shape
        X_test_flat = X_test.reshape(n_samples, -1)
        X_test_pca = self.pca.transform(X_test_flat)
        X_test_lda = self.lda.transform(X_test_pca)
        
        y_pred = []
        for test_sample in X_test_lda:
            distances = np.linalg.norm(self.X_train_lda - test_sample, axis=1)
            nearest_idx = np.argmin(distances)
            y_pred.append(self.y_train[nearest_idx])
        return np.array(y_pred)


class HOGFaceRecognition:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        from skimage.feature import hog
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.X_train_hog = None
        self.y_train = None
    
    def _extract_hog_features(self, image):
        from skimage.feature import hog
        features = hog(image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block, block_norm='L2-Hys', visualize=False, feature_vector=True)
        return features
    
    def fit(self, X_train, y_train):
        logger.info("Extracting HOG features...")
        self.X_train_hog = np.array([self._extract_hog_features(img) for img in X_train])
        self.y_train = y_train
        logger.info(f"HOG feature dim: {self.X_train_hog.shape[1]}")
    
    def predict(self, X_test):
        X_test_hog = np.array([self._extract_hog_features(img) for img in X_test])
        y_pred = []
        for test_sample in X_test_hog:
            distances = np.linalg.norm(self.X_train_hog - test_sample, axis=1)
            nearest_idx = np.argmin(distances)
            y_pred.append(self.y_train[nearest_idx])
        return np.array(y_pred)


def evaluate_traditional_method(model, X_train, y_train, X_test, y_test, method_name="Traditional"):
    logger.info(f"Evaluating {method_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"{method_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report


def visualize_eigenfaces(eigenfaces, save_path='results/eigenfaces.png'):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_faces = len(eigenfaces)
    n_cols = 5
    n_rows = (n_faces + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.4*n_rows))
    fig.suptitle('Eigenfaces', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    for i in range(n_faces):
        axes[i].imshow(eigenfaces[i], cmap='gray')
        axes[i].set_title(f'PC {i+1}')
        axes[i].axis('off')
    
    for i in range(n_faces, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"Eigenfaces saved: {save_path}")
    plt.close()

