# Add the necessary imports for the starter code.
import os
import pandas as pd
import numpy as np
import pickle
from ml.data import process_data
from sklearn.model_selection import StratifiedKFold
from ml.model import train_model, inference, compute_model_metrics
from slice_output import performance_on_slices

# Add code to load in the data.
data = pd.read_csv("data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a
# train-test split.

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

label = "salary"
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

all_precisions, all_recalls, all_fbetas = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(data, data[label])):
    print(f"\n--- Fold {fold + 1}/{k_folds} ---")

    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb)

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fbeta:.4f}")

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_fbetas.append(fbeta)

print("\n=== Cross-Validation Summary ===")
print(f"Average Precision: {np.mean(all_precisions):.4f}")
print(f"Average Recall:    {np.mean(all_recalls):.4f}")
print(f"Average F1 Score:  {np.mean(all_fbetas):.4f}")

X_final, y_final, encoder, lb = process_data(
    data, categorical_features=cat_features, label=label, training=True
)
final_model = train_model(X_final, y_final)

model_dir = os.path.join('model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(final_model, f)

with open(os.path.join(model_dir, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

with open(os.path.join(model_dir, "lb.pkl"), "wb") as f:
    pickle.dump(lb, f)

performance_on_slices(test, model, encoder, lb, cat_features)
