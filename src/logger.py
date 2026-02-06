# This is Logging while which will be used to log steps.
import logging
import os
import datetime

LOG_FILE = f"{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%s')}.log"
logs_path = os.path.join(os.getcwd(), 'Logs', LOG_FILE)
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
