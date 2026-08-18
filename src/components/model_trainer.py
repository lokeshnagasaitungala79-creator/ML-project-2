import os
import sys
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:

            logging.info("Splitting the data into train and test")

            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            # Models
            models = {

                "RandomForestRegressor":
                    RandomForestRegressor(),

                "DecisionTreeRegressor":
                    DecisionTreeRegressor(),

                "AdaBoostRegressor":
                    AdaBoostRegressor(),

                "GradientBoostingRegressor":
                    GradientBoostingRegressor(),

                "LinearRegression":
                    LinearRegression(),

                "KNeighborsRegressor":
                    KNeighborsRegressor(),

                "XGBRegressor":
                    XGBRegressor(),

                "CatBoostRegressor":
                    CatBoostRegressor(verbose=False)
            }

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models
            )

            # Print all model scores
            print("\n================ MODEL SCORES ================\n")

            for model_name, score in model_report.items():
                print(f"{model_name}: {score}")

            print("\n===============================================\n")

            # Find best model
            best_model_score = max(model_report.values())

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[best_model_name]

            print(f"Best Model: {best_model_name}")
            print(f"Best R2 Score: {best_model_score}")

            logging.info(
                f"Best model found: {best_model_name} "
                f"with R2 score: {best_model_score}"
            )

            # Do not stop model saving because score is below 0.6
            # We can improve the model later.

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                ),
                exist_ok=True
            )

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                f"Best model saved at: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            # Test prediction
            prediction = best_model.predict(X_test)

            test_model_score = r2_score(
                y_test,
                prediction
            )

            print(f"\nFinal Test R2 Score: {test_model_score}")

            return test_model_score

        except Exception as e:

            raise CustomException(e)