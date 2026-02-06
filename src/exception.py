# This is exception file which contains CustomException class which will be used to raise custom exceptions

# the sys library is responsible for handling all the exceptions raised while running the program as it contains all the data related to errors raised
import sys
from logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_traceback = error_detail.exc_info()
    filename = exc_traceback.tb_frame.f_code.co_filename
    line_no = exc_traceback.tb_lineno
    error_message = f"Error occurred in python script {filename}\n at line number {line_no}\n Error message: {str(error)}"

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return str(self.error_message)


if __name__ == "__main__":
    try:
        a =  1 / 0
    except Exception as e:
        logging.info("Division by zero error")
        raise CustomException(e, sys)




