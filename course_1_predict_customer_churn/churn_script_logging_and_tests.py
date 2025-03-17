'''
This module contains unit tests for the churn_library module.
It tests each function in the churn_library module and logs the results.

Author: Leonel Baptista
Date: 2025-03-16
'''

import logging
import os
import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import churn_library as cl
import constants


try:
    logging.basicConfig(
        filename='./logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
except Exception as exc:
    print(f"Error setting up logging: {exc}")


class TestChurnLibrary(unittest.TestCase):
    '''
    Test class for the churn_library module.
    It tests each function in the churn_library module.
    '''

    @classmethod
    def setUpClass(cls):
        '''
        Set up the data for the tests
        '''
        try:
            cls.data_frame = pd.read_csv(constants.DATA_FILE_PATH)
            cl.perform_eda(cls.data_frame)
            cls.df_encoded = cl.encoder_helper(cls.data_frame)
            cls.x_train, cls.x_test, cls.y_train, cls.y_test = cl.perform_feature_engineering(
                cls.df_encoded)
            (cls.lr_train_preds, cls.lr_test_preds, cls.rf_train_preds,
             cls.rf_test_preds, cls.y_train, cls.y_test,
             cls.rf_model, cls.lr_model) = cl.train_models(
                cls.x_train, cls.x_test, cls.y_train, cls.y_test)
            logging.info("Setting up data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Setting up data: The file wasn't found")
            raise err
        except Exception as exc:
            logging.error("Setting up data: %s", exc)
            raise exc

    def test_import_data(self):
        '''
        Test the import_data function from churn_library
        '''
        try:
            logger.info("Running test_import_data")
            data_frame = cl.import_data(constants.DATA_FILE_PATH)
            self.assertIsInstance(data_frame, pd.DataFrame)
            self.assertFalse(data_frame.empty)
            logging.info("test_import_data: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_import_data: The file doesn't appear to have rows and columns")
            logging.error(f"AssertionError: {err}")
            raise err
        except Exception as exc:
            logging.error("test_import_data: %s", exc)
            raise exc

    def test_perform_eda(self):
        '''
        Test the perform_eda function from churn_library
        '''
        try:
            logger.info("Running test_perform_eda")
            self.assertTrue(os.path.exists(
                f"{constants.EDA_IMAGES_PATH}/churn_distribution.png"))
            self.assertTrue(os.path.exists(
                f"{constants.EDA_IMAGES_PATH}/customer_age_distribution.png"))
            self.assertTrue(os.path.exists(
                f"{constants.EDA_IMAGES_PATH}/marital_status_distribution.png"))
            self.assertTrue(os.path.exists(
                f"{constants.EDA_IMAGES_PATH}/total_transaction_distribution.png"))
            self.assertTrue(os.path.exists(
                f"{constants.EDA_IMAGES_PATH}/heatmap.png"))
            logging.info("test_perform_eda: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_perform_eda: One or more EDA images are missing")
            logging.error(f"AssertionError: {err}")
            raise err
        except Exception as exc:
            logging.error("test_perform_eda: %s", exc)
            raise exc

    def test_encoder_helper(self):
        '''
        Test the encoder_helper function from churn_library
        '''
        try:
            logger.info("Running test_encoder_helper")
            df_encoded = cl.encoder_helper(self.data_frame)
            for cat in constants.CAT_COLUMNS:
                self.assertIn(f"{cat}_{constants.RESPONSE}",
                              df_encoded.columns)
            logging.info("test_encoder_helper: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_encoder_helper: One or more encoded columns are missing")
            logging.error(f"AssertionError: {err}")
            raise err
        except Exception as exc:
            logging.error("test_encoder_helper: %s", exc)
            raise exc

    def test_perform_feature_engineering(self):
        '''
        Test the perform_feature_engineering function from churn_library
        '''
        try:
            logger.info("Running test_perform_feature_engineering")
            x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
                self.df_encoded)
            self.assertEqual(x_train.shape[0], y_train.shape[0])
            self.assertEqual(x_test.shape[0], y_test.shape[0])
            self.assertEqual(x_train.shape[1], len(constants.KEEP_COLS))
            logging.info("test_perform_feature_engineering: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_perform_feature_engineering: Feature engineering output shapes are incorrect")
            logging.error(f"AssertionError: {err}")
            raise err
        except Exception as exc:
            logging.error("test_perform_feature_engineering: %s", exc)
            raise exc

    def test_train_models(self):
        '''
        Test the train_models function from churn_library
        '''
        try:
            logger.info("Running test_train_models")
            self.assertIsInstance(self.rf_model, RandomForestClassifier)
            self.assertIsInstance(self.lr_model, LogisticRegression)
            self.assertTrue(os.path.exists(
                f"{constants.MODEL_SAVE_PATH}/rfc_model.pkl"))
            self.assertTrue(os.path.exists(
                f"{constants.MODEL_SAVE_PATH}/logistic_model.pkl"))
            self.assertTrue(os.path.exists(
                f"{constants.RESULTS_IMAGES_PATH}/roc_curve_result.png"))
            logging.info("test_train_models: SUCCESS")
        except AssertionError as err:
            logging.error("test_train_models: Model training or saving failed")
            logging.error(f"AssertionError: {err}")
            raise err
        except Exception as exc:
            logging.error("test_train_models: %s", exc)
            raise exc

    def test_classification_report_image(self):
        '''
        Test the classification_report_image function from churn_library
        '''
        try:
            logger.info("Running test_classification_report_image")
            cl.classification_report_image(
                self.y_train,
                self.y_test,
                self.lr_train_preds,
                self.rf_train_preds,
                self.lr_test_preds,
                self.rf_test_preds)
            self.assertTrue(os.path.exists(
                f"{constants.RESULTS_IMAGES_PATH}/rf_results.png"))
            self.assertTrue(os.path.exists(
                f"{constants.RESULTS_IMAGES_PATH}/logistics_results.png"))
            logging.info("test_classification_report_image: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_classification_report_image: Classification report images are missing")
            logging.error(f"AssertionError: {err}")
            raise err
        except Exception as exc:
            logging.error("test_classification_report_image: %s", exc)
            raise exc

    def test_feature_importance_plot(self):
        '''
        Test the feature_importance_plot function from churn_library
        '''
        try:
            logger.info("Running test_feature_importance_plot")
            cl.feature_importance_plot(
                self.rf_model, self.x_test, constants.RESULTS_IMAGES_PATH)
            self.assertTrue(os.path.exists(
                f"{constants.RESULTS_IMAGES_PATH}/feature_importances.png"))
            logging.info("test_feature_importance_plot: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_feature_importance_plot: Feature importance plot is missing")
            logging.error(f"AssertionError: {err}")
            raise err
        except Exception as exc:
            logging.error("test_feature_importance_plot: %s", exc)
            raise exc


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader.loadTestsFromTestCase(TestChurnLibrary))
    runner = unittest.TextTestRunner()
    runner.run(suite)
