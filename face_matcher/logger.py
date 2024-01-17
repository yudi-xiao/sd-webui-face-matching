import logging
import sys

logger = logging.getLogger('FaceMatcher')
logger.propagate = False
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
