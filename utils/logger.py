import logging


def get_root_logger(name: str = 'shapematch', log_level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger with a single stream handler.

    Repeated calls return the same configured logger (handlers are only added once).
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False
    return logger


