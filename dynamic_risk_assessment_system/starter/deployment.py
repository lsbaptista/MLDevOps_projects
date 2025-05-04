import shutil
import os
import json
from logger import get_logger

logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")


with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(script_dir, config['output_folder_path'])
model_path = os.path.join(script_dir, config['output_model_path'])
prod_deployment_path = os.path.join(script_dir, config['prod_deployment_path'])


def store_model_into_pickle():

    try:
        files_to_copy = {
            os.path.join(model_path, 'trainedmodel.pkl'): 'latestmodel.pkl',
            os.path.join(model_path, 'latestscore.txt'): 'latestscore.txt',
            os.path.join(dataset_csv_path, 'ingestedfiles.txt'): 'ingestedfiles.txt'
        }

        os.makedirs(prod_deployment_path, exist_ok=True)

        for src, dest_name in files_to_copy.items():
            dest = os.path.join(prod_deployment_path, dest_name)
            shutil.copyfile(src, dest)
            logger.info(f"Copied {src} to {dest}")

        logger.info("Model deployment completed successfully.")

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    try:
        store_model_into_pickle(
            model_path, dataset_csv_path, prod_deployment_path)
    except Exception as e:
        logger.error(f"Error in model deployment: {e}")
