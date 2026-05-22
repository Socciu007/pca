from ucimlrepo import fetch_ucirepo
from PreprocessingDataPipeline import DataPipeline
from pca import PCAModel

# fetch dataset
ds = fetch_ucirepo(id=544)
print("META DATA")
print(ds.metadata)
print("VARIABLE INFORMATION")
print(ds.variables)
data = ds.data.original
X = ds.data.features
y = ds.data.targets
obesity_mapping = {
  'Insufficient_Weight': 0,
  'Normal_Weight': 1,
  'Overweight_Level_I': 2,
  'Overweight_Level_II': 3,
  'Obesity_Type_I': 4,
  'Obesity_Type_II': 5,
  'Obesity_Type_III': 6
}

y_numeric = y['NObeyesdad'].map(obesity_mapping)
number_feature = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

#### DATA PREPROCESSING PIPELINE ####
pipeline = DataPipeline(
    target_col="NObeyesdad",
    low_variance_threshold=0.01,
    remove_outlier=True
)
X_processed, y = pipeline.fit_transform(data)
# X_processed, y = pipeline.fit_transform(pd.concat([data[number_feature], y], axis=1))
print(X_processed.head())
print(y.head())

#### PCA ####
pca = PCAModel(variance_threshold=0.9)
X_pca = pca.fit_transform(X_processed.values)
pca.summary() # PCA summary
loading_matrix = pca.get_loading_matrix(X_processed.columns) # Loading matrix
print("\n===== LOADING MATRIX =====")
print(loading_matrix.T.round(3))
metrics = pca.get_metrics(X_processed.values) # get metrics
print("--- PCA Metrics ---")
for key, value in metrics.items():
    print(f"{key}: {value}")