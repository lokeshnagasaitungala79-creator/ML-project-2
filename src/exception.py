import sys
import os
from src.logger import logging


def error_message_detail(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    return (
        f"Error occurred in Python script: [{file_name}] "
        f"line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )


class CustomException(Exception):

    def __init__(self, error_message: Exception):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero error caught")
        raise CustomException(e) 