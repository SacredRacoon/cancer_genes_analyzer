import logging
import sys
from pathlib import Path

_LOGGER_INITIALIZED = False

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    global _LOGGER_INITIALIZED

    root_logger = logging.getLogger()

    if not _LOGGER_INITIALIZED:
        root_logger.setLevel(level)
        
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        _LOGGER_INITIALIZED = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger