'''
This library contains functions to import data, perform exploratory data analysis,
encode categorical columns, perform feature engineering, train models,
and store model results. The library also contains functions to store
classification reports and feature importance plots.

Author: Leonel Baptista
Date: 2025-03-16
'''

import os
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import constants

sns.set_theme()
matplotlib.use('Agg')


def import_data(pth=constants.DATA_FILE_PATH) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)

    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    pd.set_option('display.max_columns', None)

    print(data_frame.head())
    print(data_frame.shape)
    print(data_frame.isnull().sum())
    print(data_frame.describe())

    categorical_columns = data_frame.select_dtypes(
        include=['category', 'object']).columns
    print(f"Categorical columns: {categorical_columns}")

    quant_columns = data_frame.select_dtypes(include=['int', 'float']).columns
    print(f"Quantitative columns: {quant_columns}")

    os.makedirs(f"{constants.EDA_IMAGES_PATH}", exist_ok=True)

    plt.figure(figsize=(20, 10))
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    data_frame['Churn'].hist()
    plt.savefig(f"{constants.EDA_IMAGES_PATH}/churn_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig(
        f"{constants.EDA_IMAGES_PATH}/customer_age_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        f"{constants.EDA_IMAGES_PATH}/marital_status_distribution.png",
        dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(
        f"{constants.EDA_IMAGES_PATH}/total_transaction_distribution.png",
        dpi=300)
    plt.close()

    numeric_df = data_frame.select_dtypes(include=['number'])
    plt.figure(figsize=(20, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='GnBu', linewidths=2)
    plt.savefig(f"{constants.EDA_IMAGES_PATH}/heatmap.png", dpi=300)
    plt.close()


def encoder_helper(
        data_frame,
        category_lst=None,
        response=constants.RESPONSE):
    '''
    Helper function to encode categorical columns with the proportion of the response variable.

    This function takes a dataframe and a list of categorical columns, and for each categorical
    column, it calculates the mean of the response variable for each category. It then creates
    a new column in the dataframe with these mean values.
            category_lst: list of str, columns that contain categorical features
    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
            variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for each categorical feature encoded with
            the mean of the response variable
    '''
    if category_lst is None:
        category_lst = constants.CAT_COLUMNS

    for cat in category_lst:
        cat_groups = data_frame.groupby(cat)[response].mean()
        data_frame[f"{cat}_{response}"] = data_frame[cat].map(cat_groups)

    return data_frame


def perform_feature_engineering(
        data_frame,
        keep_cols=None,
        response=constants.RESPONSE,
        test_size=constants.TEST_SIZE,
        random_state=constants.RANDOM_STATE):
    '''
    input:
              data_frame: pandas dataframe
              keep_cols: list of columns to keep for X
              response: string of response name [optional argument that could be used for
              naming variables or index y column]
              test_size: float, proportion of data to use for testing
              random_state: int, random state for reproducibility

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    if keep_cols is None:
        keep_cols = constants.KEEP_COLS

    y_data = data_frame[response]
    x_data = pd.DataFrame()
    x_data = data_frame[keep_cols].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              tuple containing model predictions and the trained models
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=constants.RANDOM_STATE)
    lrc = LogisticRegression(
        solver='lbfgs', max_iter=constants.LOGISTIC_REGRESSION_MAX_ITER)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    lrc.fit(x_train, y_train)
    train_preds_lr = lrc.predict(x_train)
    test_preds_lr = lrc.predict(x_test)

    joblib.dump(cv_rfc.best_estimator_,
                f'{constants.MODEL_SAVE_PATH}/rfc_model.pkl')
    joblib.dump(lrc, f'{constants.MODEL_SAVE_PATH}/logistic_model.pkl')

    plot_roc_curve(lrc, cv_rfc.best_estimator_, x_test, y_test)

    return (train_preds_lr, test_preds_lr, train_preds_rf,
            test_preds_rf, y_train, y_test, cv_rfc.best_estimator_, lrc)


def plot_roc_curve(lrc_model, rfc_model, x_test_data, y_test_data):
    '''
    Creates and saves ROC curve for the models
    input:
            lrc_model: trained logistic regression model
            rfc_model: trained random forest model
            x_test_data: test features
            y_test_data: test target
    output:
            None
    '''
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        lrc_model, x_test_data, y_test_data, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(
        rfc_model, x_test_data, y_test_data, ax=ax, alpha=0.8)
    plt.savefig(
        f"{constants.RESULTS_IMAGES_PATH}/roc_curve_result.png", dpi=300)
    plt.close()


def classification_report_image(y_train_data,
                                y_test_data,
                                train_preds_lr,
                                train_preds_rf,
                                test_preds_lr,
                                test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train_data: training response values
            y_test_data:  test response values
            train_preds_lr: training predictions from logistic regression
            train_preds_rf: training predictions from random forest
            test_preds_lr: test predictions from logistic regression
            test_preds_rf: test predictions from random forest

    output:
             None
    '''
    report_rf_train = classification_report(y_train_data, train_preds_rf)
    report_rf_test = classification_report(y_test_data, test_preds_rf)
    report_lr_train = classification_report(y_train_data, train_preds_lr)
    report_lr_test = classification_report(y_test_data, test_preds_lr)

    plt.figure(figsize=(6, 6), dpi=300)

    plt.text(0.01, 1.15, 'Random Forest Train', fontsize=12,
             fontproperties='monospace', ha='left')
    plt.text(0.01, 0.80, report_rf_train, fontsize=10,
             fontproperties='monospace', ha='left')
    plt.text(0.01, 0.70, 'Random Forest Test', fontsize=12,
             fontproperties='monospace', ha='left')
    plt.text(0.01, 0.35, report_rf_test, fontsize=10,
             fontproperties='monospace', ha='left')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{constants.RESULTS_IMAGES_PATH}/rf_results.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6), dpi=300)

    plt.text(0.01, 1.15, 'Logistic Regression Train',
             fontsize=12, fontproperties='monospace', ha='left')
    plt.text(0.01, 0.80, report_lr_train, fontsize=10,
             fontproperties='monospace', ha='left')
    plt.text(0.01, 0.70, 'Logistic Regression Test',
             fontsize=12, fontproperties='monospace', ha='left')
    plt.text(0.01, 0.35, report_lr_test, fontsize=10,
             fontproperties='monospace', ha='left')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        f"{constants.RESULTS_IMAGES_PATH}/logistics_results.png", dpi=300)
    plt.close()


def feature_importance_plot(model, x_test_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_test_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    plt.figure(figsize=(12, 6))

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, x_test_data)
        shap_values = explainer.shap_values(x_test_data)
        shap.summary_plot(shap_values, x_test_data,
                          plot_type="bar", show=False)
        plt.title("Feature Importance - Logistic Regression")

    elif isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [x_test_data.columns[i] for i in indices]

        plt.bar(range(x_test_data.shape[1]), importances[indices])
        plt.title("Feature Importance - Random Forest")
        plt.ylabel('Importance')
        plt.xticks(range(x_test_data.shape[1]), names, rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_pth}/feature_importances.png",
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    imported_df = import_data(r"./data/bank_data.csv")
    perform_eda(imported_df)
    encoded_df = encoder_helper(imported_df)
    train_x, test_x, train_y, test_y = perform_feature_engineering(encoded_df)
    lr_train_preds, lr_test_preds, rf_train_preds, rf_test_preds, train_y, test_y, \
        rfc_model, lr_model = train_models(
            train_x, test_x, train_y, test_y)
    classification_report_image(
        train_y,
        test_y,
        lr_train_preds,
        rf_train_preds,
        lr_test_preds,
        rf_test_preds)
    feature_importance_plot(rfc_model, test_x, constants.RESULTS_IMAGES_PATH)
    feature_importance_plot(lr_model, test_x, constants.RESULTS_IMAGES_PATH)
    print("Done")
