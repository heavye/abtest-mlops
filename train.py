#This script is modified from the mlflow github repo file -> train.py 

# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Get url from DVC
import dvc.api

path = 'data/AdSmartABdata.csv'
# repo ='https://github.com/heavye/abtest-mlops'
repo ='C:/Users/euelf/Desktop/10_AC_CHALLENGES/Week_2/AB_Hypothesis_Testing/abtest-mlops'
version = 'version4'

data_url = dvc.api.get_url(
    path = path,
    repo = repo,
    rev = version
)

mlflow.set_experiment('hjhhh')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the AdSmartABdata csv file from remote repositery
    data = pd.read_csv(data_url, sep=",")

    

    # Split the data into training and test sets. (0.75, 0.25) split
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["awareness"], axis=1)
    test_x = test.drop(["awareness"], axis=1)
    train_y = train[["awareness"]]
    test_y = test[["awareness"]]

    

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        #Log data params
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_cols', data.shape[1])

        #Log artifacts: columns used for modeling
        cols_x = pd.DataFrame(list(train_x.columns))
        cols_x.to_csv('features.csv', header = False, index = False)
        mlflow.log_artifact('features.csv')

        cols_y = pd.DataFrame(list(train_y.columns))
        cols_y.to_csv('targets.csv', header = False, index = False)
        mlflow.log_artifact('targets.csv')

        mlflow.sklearn.log_model(lr, "model")
