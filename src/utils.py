import os
import sys
import pickle

from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    """Saves a Python object to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """Loads a pickled Python object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Train and evaluate multiple regression models using R2 score."""
    try:
        report = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)