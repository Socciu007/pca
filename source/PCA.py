import numpy as np
import pandas as pd

# PCA IMPLEMENTATION
class PCAModel:
  def __init__(self, variance_threshold=0.9):
    self.variance_threshold = variance_threshold # Ngưỡng biến đổi variance
    self.mean_vector = None # Vector trung bình
    self.cov_matrix = None # Ma trận hiệp phương sai
    self.eigenvalues = None # Giá trị riêng
    self.eigenvectors = None # Vector riêng
    self.explained_variance_ratio = None # Tỷ lệ biến đổi variance
    self.cumulative_variance = None # Tỷ lệ biến đổi variance tích lũy
    self.n_components = None # Số lượng thành phần chính
    self.components = None # Ma trận thành phần chính

  # FIT PCA
  def fit(self, X):
    X = np.array(X)

    # Mean vector
    self.mean_vector = np.mean(X, axis=0)

    # Covariance matrix
    self.cov_matrix = np.cov(X, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)

    # Sort descending
    sorted_idx = np.argsort(eigenvalues)[::-1]
    self.eigenvalues = eigenvalues[sorted_idx]
    self.eigenvectors = eigenvectors[:, sorted_idx]

    # Explained variance
    total_variance = np.sum(self.eigenvalues)
    self.explained_variance_ratio = (self.eigenvalues / total_variance)
    self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

    # Select number of components
    self.n_components = (np.argmax(self.cumulative_variance >= self.variance_threshold) + 1)
    print(f"Selected components: {self.n_components}")

    # Principal components
    self.components = self.eigenvectors[:,:self.n_components]

  # TRANSFORM DATA
  def transform(self, X):
    X = np.array(X)
    X_centered = X - self.mean_vector
    X_pca = np.dot(X_centered, self.components)
    return X_pca

  # FIT + TRANSFORM
  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)

  # GET LOADING MATRIX
  def get_loading_matrix(self, feature_names):
    return pd.DataFrame(
      self.components,
      index=feature_names,
      columns=[f"PC{i+1}" for i in range(self.n_components)]
    )

  # GET EIGENVALUES
  def get_eigenvalues(self):
    return self.eigenvalues

  # GET EXPLAINED VARIANCE RATIO
  def get_explained_variance_ratio(self):
    return self.explained_variance_ratio

  # GET CUMULATIVE VARIANCE
  def get_cumulative_variance(self):
    return self.cumulative_variance

  # GET TOTAL VARIANCE RETAINED
  def get_total_variance_retained(self):
    return self.cumulative_variance[self.n_components - 1]

  # GET DIMENSION REDUCTION RATIO
  def get_dimension_reduction_ratio(self):
    original_dim = len(self.eigenvalues)
    reduced_dim = self.n_components
    reduction_ratio = ((original_dim - reduced_dim)/ original_dim)
    return reduction_ratio

  # GET RECONSTRUCTION ERROR
  def get_reconstruction_error(self, X):
    X = np.array(X)
    X_pca = self.transform(X)# PCA transform
    X_reconstructed = np.dot(X_pca, self.components.T) + self.mean_vector # Reconstruct data
    mse = np.mean((X - X_reconstructed) ** 2) # Mean Squared Error
    return mse

  def get_metrics(self, X):
    metrics = {
      "original_dimensions": len(self.eigenvalues),
      "selected_components": self.n_components,
      "variance_retained": self.get_total_variance_retained(),
      "dimension_reduction_ratio": self.get_dimension_reduction_ratio(),
      "reconstruction_error": self.get_reconstruction_error(X),
      "condition_number": (np.max(self.eigenvalues) / np.min(self.eigenvalues)),
      "kaiser_components": np.sum(self.eigenvalues > 1)
    }
    return metrics

  # PRINT SUMMARY
  def summary(self):
    print("\n===== PCA SUMMARY =====")
    print("\nEigenvalues:")
    print(self.eigenvalues)
    print("\nExplained Variance Ratio:")
    print(self.explained_variance_ratio)
    print("\nCumulative Variance:")
    print(self.cumulative_variance)
    print(f"\nSelected Components: {self.n_components}")