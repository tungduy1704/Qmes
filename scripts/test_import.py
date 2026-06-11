import logging
logging.basicConfig(level=logging.WARNING)

from Qmes.data.classification import load_classification_datasets
from Qmes.data.regression import load_regression_datasets

clf = load_classification_datasets()
affected_clf = [name for name, (X, y) in clf.items() if len(X) == 300]
print(f"Classification cần re-run ({len(affected_clf)}):")
print(affected_clf)

reg = load_regression_datasets()
affected_reg_capped = [name for name, (X, y) in reg.items() if len(X) == 300]
# + đối chiếu warning "dropped ... NaN target" trong log để lấy nhóm Fix 5
print(f"Regression bị cap ({len(affected_reg_capped)}):")
print(affected_reg_capped)