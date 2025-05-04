import logging
import os


def get_logger(name=__name__, log_file='ingestion.log', level=logging.INFO):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_path)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console)

    return logger
