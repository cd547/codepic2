import logging
import os
from datetime import datetime
from config import LOG_DIR


def init_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    logger = logging.getLogger("BarcodeSystem")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(fmt)
        file_handler.setFormatter(fmt)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger
