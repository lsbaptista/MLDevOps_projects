
import pandas as pd
import numpy as np
import timeit
import pickle
import os
import json
import subprocess
import sys
from logger import get_logger

logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")

# Load config.json and get environment variables
with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(script_dir, config['output_folder_path'])
test_data_path = os.path.join(script_dir, config['test_data_path'])
model_path_prod = os.path.join(script_dir, config['prod_deployment_path'])
model_file_path = os.path.join(model_path_prod, "latestmodel.pkl")


def model_predictions(model_file_path=model_file_path, file_path=None):
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)

    if file_path:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(os.path.join(
            test_data_path, 'testdata.csv'))

    X = df.loc[:, ['lastmonth_activity', 'lastyear_activity',
                   'number_of_employees']].values.reshape(-1, 3)
    predictions = model.predict(X)
    return predictions.tolist()


def dataframe_summary(dataset_csv_path=dataset_csv_path, file_path=None):
    if file_path:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(os.path.join(
            script_dir, dataset_csv_path, 'finaldata.csv'))

    df = df.select_dtypes(include=[np.number])
    summary = {}
    summary['mean'] = df.mean().to_dict()
    summary['median'] = df.median().to_dict()
    summary['std_dev'] = df.std().to_dict()
    return summary


def missing_date(dataset_csv_path=dataset_csv_path):
    df = pd.read_csv(os.path.join(
        script_dir, dataset_csv_path, 'finaldata.csv'))
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    return missing_percentages.to_dict()


def execution_time():

    start_time = timeit.default_timer()
    os.system(f'python {os.path.join(script_dir, "ingestion.py")}')
    end_time = timeit.default_timer()
    ingestion_time = end_time - start_time
    start_time = timeit.default_timer()
    os.system(f'python {os.path.join(script_dir, "training.py")}')
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    return [ingestion_time, training_time]


def outdated_packages_list():
    """
    Prints a table showing all installed packages that are outdated.
    Each row includes the package name, current version, and latest version.
    """
    try:

        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        outdated_packages = json.loads(result.stdout)

        if not outdated_packages:
            return ["All packages are up to date."]

        return [
            {
                "name": package['name'],
                "current_version": package['version'],
                "latest_version": package['latest_version']
            }
            for package in outdated_packages
        ]

    except subprocess.CalledProcessError as e:
        print("An error occurred while checking for outdated packages:")
        print(e.stderr)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
