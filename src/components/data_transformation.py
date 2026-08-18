import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            logging.info("Numerical and categorical pipelines created successfully")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str
    ):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded train and test datasets")

            train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
            test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "Churn"
            
            input_features_train_df = train_df.drop(columns=[target_column_name], errors="ignore")
            input_features_test_df = test_df.drop(columns=[target_column_name], errors="ignore")

            target_feature_train_df = train_df[target_column_name].map({"Yes": 1, "No": 0}).fillna(0)
            target_feature_test_df = test_df[target_column_name].map({"Yes": 1, "No": 0}).fillna(0)

            logging.info("Applying preprocessing object on training dataframe")
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)

            logging.info("Applying preprocessing object on testing dataframe")
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Preprocessing completed successfully")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)