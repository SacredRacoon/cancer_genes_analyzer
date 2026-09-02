# src/utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Настраивает именной логгер И корневой логгер.
    Благодаря этому logger.info() внутри ЛЮБОГО модуля проекта будет работать.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # === 1. Настраиваем именной логгер (например, "GA_Glioma") ===
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
<<<<<<< HEAD
    if logger.hasHandlers():
        logger.handlers.clear()

=======
    # Очищаем старые хендлеры, чтобы не дублировать при перезапуске
    if logger.hasHandlers():
        logger.handlers.clear()

    # Консоль
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

<<<<<<< HEAD
=======
    # Файл
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

<<<<<<< HEAD
=======
    # === 2. КЛЮЧЕВОЙ МОМЕНТ: Настраиваем корневой логгер ===
    # Это заставит ВСЕ логгеры в проекте (включая src.evaluation.visualizer)
    # использовать те же настройки консоли и файла
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.addHandler(console_handler)
    if log_file:
        root_logger.addHandler(file_handler)

    return logger