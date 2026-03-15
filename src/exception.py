import sys

# Sys library is responsible for handling all the exceptions occured in the running of the code as it has all the data and information related to the exceptions raised
from src.logger import logging


# this error detail param we will get from sys
def error_message_detail(error, error_detail: sys):
    # okay so this below function will give us 3 objects, only the last one is useful for us as that will tell use on which file and which line the error occurred and is of which type

    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occurred in python script: [{0}]\n line number: [{1}]\n error message[{2}]".format(
        filename, line_number, str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return str(self.error_message)


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys)
