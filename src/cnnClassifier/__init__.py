# here we create the custom LOGGER 
# we create it inside the src folder constructor file , which makes it easy to use 

import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),   # save logs in log folder
        logging.StreamHandler(sys.stdout)    # print logs in console
    ]
)

logger = logging.getLogger("cnnClassifierLogger")