import logging

def get_logger(name: str = __name__):
    """Create and returns a logger object
    """

    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(name)
