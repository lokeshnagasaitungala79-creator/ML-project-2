import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.utils import evaluate_models , save_object


class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join("artifats" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.ModelTrainer  = ModelTrainerConfig()

    def initiate_model_trainer(self , train_arr , test_arr):
        try:
            logging.info("splitting the data into the train and test")
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1],
            )

            models = {
                "RandomForestRegreesor" : RandomForestRegressor(),
                "DecisionTreeRegressor"  : DecisionTreeRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "LinearRegression" : LinearRegression(),
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoostRegressor" : CatBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report : dict = evaluate_models(
                X_train = X_train,
                X_test = X_test,
                y_train = X_train,
                y_test = X_test,
                models = models
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report , key = model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model found")

            logging.info(f"best model found: {best_model_name} with r2_score: {best_model_score}")

            #saving the file
            save_object (
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            prediction = best_model.predict(X_test)
            r2_score = r2_score(y_test , prediction)
        except Exception as e:
            raise CustomException(e , sys)



            
