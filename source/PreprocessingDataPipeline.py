import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

# OBESITY MAPPING
obesity_mapping = {
  'Insufficient_Weight': 0,
  'Normal_Weight': 1,
  'Overweight_Level_I': 2,
  'Overweight_Level_II': 3,
  'Obesity_Type_I': 4,
  'Obesity_Type_II': 5,
  'Obesity_Type_III': 6
}
# DATA PREPROCESSING PIPELINE
class DataPipeline:
  def __init__(
    self,
    target_col=None, # Biến mục tiêu
    low_variance_threshold=0.01, # Ngưỡng thấp để loại bỏ biến
    outlier_alpha=0.975, # Ngưỡng để phát hiện outliers (mức ý nghĩa dùng trong Mahalanobis distance)
    remove_outlier=False # Loại bỏ outliers
  ):
    self.target_col = target_col
    self.low_variance_threshold = low_variance_threshold
    self.outlier_alpha = outlier_alpha
    self.remove_outlier = remove_outlier

    self.numeric_cols = None
    self.categorical_cols = None

    self.means = {}
    self.stds = {}

    self.label_maps = {}

    self.selected_columns = None

  # HANDLE MISSING VALUES
  def handle_missing(self, df):
    data = df.copy()
    self.numeric_cols = data.select_dtypes(
      include=["int64", "float64"]
    ).columns.tolist()
    self.categorical_cols = data.select_dtypes(
      include=["object"]
    ).columns.tolist()
    # Numeric -> median
    for col in self.numeric_cols:
      median_value = data[col].median()
      data[col] = data[col].fillna(median_value)
    # Categorical -> mode
    for col in self.categorical_cols:
      mode_value = data[col].mode()[0]
      data[col] = data[col].fillna(mode_value)
    return data

  # ENCODE CATEGORICAL VARIABLES
  def encode_categorical(self, df):
    data = df.copy()
    for col in self.categorical_cols:
      le = LabelEncoder()
      unique_vals = [str(x).lower() for x in data[col].unique()]
      if all(val in ['yes', 'no'] for val in unique_vals):
        data[col] = data[col].map({'no': 0, 'yes': 1, 'No': 0, 'Yes': 1})
      elif all(val in ['male', 'female'] for val in unique_vals):
        data[col] = data[col].map({'male': 0, 'female': 1, 'Male': 0, 'Female': 1})
      elif col == 'NObeyesdad':
        data[col] = data[col].map(obesity_mapping)
      else:
        data[col] = le.fit_transform(data[col])
        self.label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return data

  # REMOVE LOW VARIANCE FEATURE
  def remove_low_variance(self, df):
    data = df.copy()
    variances = data.var()
    keep_cols = variances[
      variances >= self.low_variance_threshold
    ].index.tolist()
    self.selected_columns = keep_cols
    return data[keep_cols]

  # STANDARDIZE DATA
  def standardize(self, df):
    data = df.copy()
    for col in data.columns:
      mean_value = np.mean(data[col])
      std_value = np.std(data[col], ddof=1)
      self.means[col] = mean_value
      self.stds[col] = std_value
      data[col] = (data[col] - mean_value) / std_value
    return data

  # DETECT OUTLIERS USING MAHALANOBIS
  def detect_outliers(self, X):
    mean_vec = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    distances = []
    for row in X:
      d = mahalanobis(row, mean_vec, inv_cov_matrix)
      distances.append(d)
    distances = np.array(distances)
    mahal_sq = distances ** 2
    p = X.shape[1]
    threshold = chi2.ppf(self.outlier_alpha, df=p)
    outlier_mask = mahal_sq > threshold
    return outlier_mask

  # FULL PREPROCESSING PIPELINE
  def fit_transform(self, df):
    data = df.copy()

    # Split target
    y = None
    if self.target_col is not None:
      y = data[self.target_col]
      data = data.drop(columns=[self.target_col])

    # Missing values
    data = self.handle_missing(data)

    # Encode categorical
    data = self.encode_categorical(data)

    # Remove low variance
    # data = self.remove_low_variance(data)

    # Standardization
    data = self.standardize(data)

    # Outlier detection
    outlier_mask = self.detect_outliers(data.values)

    print(f"Outliers detected: {outlier_mask.sum()}")
    if self.remove_outlier:
      data = data.loc[~outlier_mask]
      if y is not None:
        y = y.loc[~outlier_mask]

    return data, y

  # TRANSFORM NEW DATA
  def transform(self, df):
    data = df.copy()
    # Encode
    for col, mapping in self.label_maps.items():
      if col in data.columns:
        data[col] = data[col].map(mapping)
    # Keep selected columns
    data = data[self.selected_columns]
    # Standardize
    for col in data.columns:
      data[col] = ([col] - self.means[col]) / self.stds[col]
    return data