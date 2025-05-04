import pandas as pd
import os
import json
from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Script directory: {script_dir}")

with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
outputlocation = os.path.join(
    script_dir, output_folder_path, 'ingestedfiles.txt')


def merge_multiple_dataframes():
    filenames = os.listdir(os.path.join(script_dir, input_folder_path))
    if not filenames:
        logger.warning("No files found in the input folder.")
        return

    dataframes = []

    with open(outputlocation, 'w') as MyFile:
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(
                    script_dir, input_folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    dataframes.append(df)

                    dateTimeObj = datetime.now()
                    thetimenow = dateTimeObj.strftime('%Y/%m/%d %H:%M:%S')
                    record_line = ','.join([
                        input_folder_path,
                        filename,
                        str(len(df.index)),
                        thetimenow
                    ]) + '\n'
                    MyFile.write(record_line)
                    logger.info(f"Processed file: {filename}")
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")
            else:
                logger.warning(f"Skipping non-CSV file: {filename}")

    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)

        output_file_path = os.path.join(
            script_dir, output_folder_path, 'finaldata.csv')
        merged_df.to_csv(output_file_path, index=False)
        logger.info(f"Merged data written to {output_file_path}")
    else:
        logger.warning("No CSV files were processed; nothing to merge.")


if __name__ == "__main__":
    try:
        merge_multiple_dataframes()
    except Exception as e:
        logger.exception(f"An error occurred during ingestion: {e}")
