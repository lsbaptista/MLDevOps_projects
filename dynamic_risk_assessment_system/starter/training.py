import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
from logger import get_logger

logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")


with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(script_dir, config['output_folder_path'])
model_path = os.path.join(script_dir, config['output_model_path'])


def train_model():

    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               n_jobs=None, penalty='l2', random_state=0,
                               solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
    df = pd.read_csv(os.path.join(
        script_dir, dataset_csv_path, 'finaldata.csv'))

    X = df.loc[:, ['lastmonth_activity', 'lastyear_activity',
                   'number_of_employees']].values.reshape(-1, 3)
    y = df['exited'].values.reshape(-1, 1).ravel()

    model = logit.fit(X, y)

    model_file_path = os.path.join(model_path, "trainedmodel.pkl")
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Error in training model: {e}")
