import os
import json
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
from logger import get_logger

logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")

with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']

ingested_file_path = os.path.join(
    script_dir, prod_deployment_path, 'ingestedfiles.txt')
source_data_folder = os.path.join(script_dir, input_folder_path)
score_file_path = os.path.join(
    script_dir, prod_deployment_path, 'latestscore.txt')
file_path = os.path.join(script_dir, output_folder_path, 'finaldata.csv')
prod_model = os.path.join(
    script_dir, prod_deployment_path, 'latestmodel.pkl')


with open(ingested_file_path, 'r') as f:
    ingested_files = [line.split(',')[1] for line in f.read().splitlines()]


source_files = [f for f in os.listdir(
    source_data_folder) if f.endswith('.csv')]
new_files = [file for file in source_files if file not in ingested_files]


if new_files:
    logger.info("New files found. Proceeding with ingestion.")
    ingestion.merge_multiple_dataframes()
else:
    logger.info("No new files found. Exiting process.")
    exit()

with open(score_file_path, 'r') as f:
    line = f.read().strip()
    old_score = float(line.split(":")[1].strip())

logger.info("Scoring the current deployed model on new data.")

new_score = scoring.score_model(
    model_path=prod_model, data_path=file_path)

logger.info(f"Old score: {old_score}, New score: {new_score}")

if new_score < old_score or new_files:
    logger.info(
        "Model drift detected or new data detected. Proceeding with retraining and redeployment.")
    logger.info("Retraining the model.")
    training.train_model()

    logger.info("Re-deploying the model.")
    deployment.store_model_into_pickle()

    logger.info("Running diagnostics...")
    diagnostics.model_predictions()
    diagnostics.dataframe_summary()
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()


logger.info("Generating confusion matrix report...")
reporting.score_model(data_path=file_path)

logger.info("Calling API endpoints and saving responses.")
# Need to start the Flask server app.py before making API calls
os.system(f'python {os.path.join(script_dir, "apicalls.py")}')

logger.info("Pipeline execution completed successfully.")
