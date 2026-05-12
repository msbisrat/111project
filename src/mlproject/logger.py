# Logging is used to track what your code is doing while it runs—especially useful 
# for debugging, monitoring, and understanding errors.
# It’s a powerful alternative to just using print() statements.

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd()  , "logs" , LOG_FILE) # getcurrent working directory path

os.makedirs(log_path , exist_ok = True) # this code is used to make directory 
# (exist_ok = True means if the directory already exists, it won't raise an error)
LOG_FILE_PATH  = os.path.join(log_path , LOG_FILE) # log_path is logs folder_name anf LOG_FILE is the name of the log file

logging.basicConfig(
    filename = LOG_FILE_PATH ,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)
