# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict customer churn using machine learning techniques. 
It demonstrates the end-to-end process of data ingestion, exploratory data analysis, 
feature engineering, model training, evaluation, and deployment. The pipeline integrates 
various ML and DevOps best practices to ensure scalability and reliability.

## Files and data description
The project contains the following main files and directories:

churn_library.py: Library of functions for data processing and modeling.
churn_script_logging_and_tests.py: Script for logging and testing.
constants.py: A file containing constant values used throughout the project.
data/: Directory containing input datasets.
images/: Visualizations and plots.
models/: Saved model files.
logs/: Logging output.
churn_notebook.ipynb: Jupyter notebooks with the code that need to be refactor.
Guide.ipynb: Getting started and troubleshooting tips
requirements_py3.10: A file listing all the necessary Python packages required to run the project.
Dockerfile: A file to build a Docker image for the project.
.devcontainer/: A folder containing configuration files for the development container (including devcontainer.json).
README.md: Project documentation.

## Running Files
Use a code editor like VSCode.
### Create a Virtual Environment with Python 3.10 (using pip)
Run the following command to create a virtual environment:
python3.10 -m venv myenv
Activate the Virtual Environment:
Windows:
myenv\Scripts\activate
Linux/Mac:
source myenv/bin/activate
Run the following command to install the dependencies
pip install -r requirements_py3.10.txt
### Alternative: Create a Conda Environment
conda create --name myenv python=3.10 --file requirements_py3.10.txt -c conda-forge
conda activate myenv
### Optionally, you can use Docker to build and run the project:
docker build -t churn-predictor .
docker run churn-predictor

Run the main script to perform data processing, model training, and evaluation:
python3 churn_library.py
This will generate model artifacts and result images in the models and images folder.

Run the testing script to ensure all functions work correctly:
python3 -m unittest churn_script_logging_and_tests.py
The tests will output results to the console, allowing you to verify the functionality of the main script,
and the logs will be written to the logs/churn_library.log.

## Results and Outputs
The output images and plots will be stored in the images folder.
Test results will be displayed on the console and logs will be written to the logs folder (churn_library.log).