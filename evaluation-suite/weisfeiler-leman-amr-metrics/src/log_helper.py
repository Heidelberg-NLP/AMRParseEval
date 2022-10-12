import logging

def set_get_logger(name, level=50):
    """
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    logger.addHandler(handler)
    """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s', level=level)
    logger = logging.getLogger(name)
    return logger


def get_logger(name):
    return logging.getLogger("root")
