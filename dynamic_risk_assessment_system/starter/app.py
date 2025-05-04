from flask import Flask, jsonify, request
import os
import json
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_date, outdated_packages_list
from scoring import score_model
from logger import get_logger

logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(script_dir, config['output_folder_path'])
prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']

ingested_file_path = os.path.join(
    script_dir, prod_deployment_path, 'ingestedfiles.txt')
source_data_folder = os.path.join(script_dir, input_folder_path)
score_file_path = os.path.join(
    script_dir, prod_deployment_path, 'latestscore.txt')
file_path = os.path.join(script_dir, dataset_csv_path, 'finaldata.csv')
prod_model = os.path.join(
    script_dir, prod_deployment_path, 'latestmodel.pkl')

prediction_model = None


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    try:
        input_json = request.get_json()
        file_path = input_json.get("filepath")
        logger.info(f"Received file path: {file_path}")
        full_file_path = os.path.join(script_dir, file_path)
        if not file_path or not os.path.isfile(full_file_path):
            return jsonify({"error": "Valid file path not provided"}), 400

        predictions = model_predictions(file_path=full_file_path)

        logger.info(f"Predictions made successfully.")
        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    try:
        f1_score = score_model(model_path=prod_model, data_path=file_path)
        return jsonify({"f1_score": f1_score}), 200
    except Exception as e:
        logger.exception("Error during scoring")
        return jsonify({"error": str(e)}), 500


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    try:
        stats = dataframe_summary()
        return jsonify(stats), 200
    except Exception as e:
        logger.exception("Error getting summary stats")
        return jsonify({"error": str(e)}), 500


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    try:

        timing = execution_time()

        missing = missing_date()

        outdated = outdated_packages_list()

        return jsonify({
            "execution_time": timing,
            "missing_data": missing,
            "outdated_packages": outdated
        }), 200

    except Exception as e:
        logger.exception("Error running diagnostics")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
