"""
This utils file contains helpers for dealing with
errors in our code.
"""
import logging


def error_with_logging(msg: str, logger: logging.Logger, error_type):
    """Logs an error with logger.error before raising it.

    We use this method as a shorthand because we often want to log an error
    and raise it. The advantage of logging it is our logger may be setup
    to write to other streams in addition to `sys.stdout` (where `print`
    writes to), such as a file.

    Parameters
    ----------
    msg : str
        The message to describe the error. This message should
        try to be as clear as possible about what the believed cause
        of the error is.
    logger : logging.Logger
        The logger to log the message with.
    error_type: callable exception
        The error to raise (e.g. ValueError). The message will be passed
        to the error_type.

    Raises
    -------
    error_type
        Raises the error specified by error_type.
    """
    logger.error(msg)
    raise error_type(msg)
