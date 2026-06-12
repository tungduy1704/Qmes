from sklearn.utils import resample
import numpy as np
from Qmes.data.classification import load_classification_datasets

# Chạy 2 lần, in shape và 5 rows đầu của CDC Diabetes
datasets = load_classification_datasets()
X, y = datasets["CDC Diabetes"]
print(X.shape)
print(X[:3])