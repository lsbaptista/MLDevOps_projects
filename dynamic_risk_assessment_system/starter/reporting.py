
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
from logger import get_logger

logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")


with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(script_dir, config['test_data_path'])


def score_model(data_path=None):

    if data_path:
        test_df = pd.read_csv(data_path)
    else:
        test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_true = test_df['exited'].values

    y_pred = model_predictions(file_path=data_path)

    cm = metrics.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'Stayed', 'Exited'], yticklabels=['Stayed', 'Exited'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    output_path = os.path.join(
        script_dir, config['output_model_path'], 'confusionmatrix.png')
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Confusion matrix saved to {output_path}")


if __name__ == '__main__':
    score_model()
