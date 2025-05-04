import pandas as pd
import pickle
import os
from sklearn import metrics
import json
from logger import get_logger


logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")


with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

model_path = os.path.join(script_dir, config['output_model_path'])
test_data_path = os.path.join(script_dir, config['test_data_path'])
prod_deployment_path = os.path.join(script_dir, config['prod_deployment_path'])


def score_model(model_path=None, data_path=None):

    if not model_path:
        model_path = os.path.join(model_path, "trainedmodel.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if not data_path:
        data_path = os.path.join(
            script_dir, config['test_data_path'], 'testdata.csv')
    df = pd.read_csv(data_path)
    X = df.loc[:, ['lastmonth_activity', 'lastyear_activity',
                   'number_of_employees']].values.reshape(-1, 3)
    y = df['exited'].values.reshape(-1, 1).ravel()
    predicted = model.predict(X)
    flscore = metrics.f1_score(y, predicted)
    with open(os.path.join(script_dir, config['output_model_path'], 'latestscore.txt'), 'w') as f:
        f.write(f'F1 score: {flscore}')

    return flscore


if __name__ == "__main__":
    try:
        score_model()
    except Exception as e:
        logger.error(f"Error in training model: {e}")
