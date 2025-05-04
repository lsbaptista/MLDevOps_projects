from ml.data import process_data
from ml.model import inference, compute_model_metrics


def performance_on_slices(data, model, encoder, lb, cat_features,
                          label="salary", output_path="slice_output.txt"):
    """
    Evaluate model performance on data slices for each category in categorical features.
    Writes results to a text file.
    """
    with open(output_path, "w") as f:
        for feature in cat_features:
            for val in data[feature].unique():
                slice_df = data[data[feature] == val]
                X_slice, y_slice, _, _ = process_data(
                    slice_df, categorical_features=cat_features, label=label,
                    training=False, encoder=encoder, lb=lb
                )
                preds = inference(model, X_slice)
                precision, recall, fbeta = compute_model_metrics(
                    y_slice, preds)
                f.write(f"{feature}={val} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}\n")
