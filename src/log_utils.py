"""
This utils file contains helpers for tracking events when the code is run.
"""
import sys
import logging
import os
import sys
from datetime import datetime
from functools import partial

from .params import (
    TMP_DIR,
    LOGS_DIR
)

from . import file_utils as f


def _start_logger_if_necessary(name, logging_level, logpath, file_logging):
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.setLevel(logging_level)
        formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
        if file_logging:
            fh = logging.FileHandler(logpath)
            fh.setLevel(logging_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Override the except hook to also log the exception
        def handle_exception(exc_type, exc_value, exc_traceback):
            logger.error("Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback))
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = handle_exception
    return logger

def get_logger(name: str, debug=False, return_getter=False, file_logging=True):
    """
    This is a util to get the logger for use. It will write out logs to a file in the logs folder
    with name NAME_TIMESTAMP.txt where NAME is the argument `name` and TIMESTAMP is the current
    time written in the formay year_month_day_hour_minute_second (e.g. 2022_08_11_18_47_25).
    
    TODO: I'm not sure if this will cause any unexpected results when using parallelization. Scikit-learn
    and companion libraries use `joblib`, which will use `loky` as its backend. `loky` appears to be based on python's
    `multiprocessing`. If this becomes an issue, I can investigate using multiprocessing safe queues. `loky` appears to
    have one implemented: https://github.com/joblib/loky/blob/4d21f8f4ebbe712221d9adc99aea045c61ee6d68/loky/backend/queues.py
    
    To get around issues with logging and parallelization, I have also allowed returning a "getter" for the log
    that will return the logger. This way, processes can still retrieve the logger after they have been spawned.
    """
    logging_level = logging.DEBUG if debug else logging.INFO
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logname = '_'.join([name, timestamp]) + '.txt'
    
    logdir = f.safe_open_dir(TMP_DIR) if debug else f.safe_open_dir(LOGS_DIR)
    logpath = os.path.join(logdir, logname)
    
    logger_getter = partial(_start_logger_if_necessary, name, logging_level, logpath, file_logging)
    logger = logger_getter()
    
    if file_logging:
        logger.info(f'Logfile will be written to {logpath}')
    if debug:
        logger.debug(f'Running in debug mode')
    
    

    
    
    if return_getter:
        return logger, logger_getter
    else:
        return logger

class _PrintLogger:
    """
    Implements the python logging API and just prints the output message.
    Useful as a default parameter for code so I don't always have to pass the log
    when I'm not doing logging.
    """
    
    def info(self, msg):
        print(msg)
    
    def debug(self, msg):
        print(msg)
    
    def warning(self, msg):
        print(msg)
    
    def error(self, msg):
        print(msg)

PrintLogger = _PrintLogger()