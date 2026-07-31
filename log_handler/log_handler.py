import os
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
os.environ['KIVY_LOG_MODE'] = 'PYTHON'


DEFAULT_LOG_FORMAT = "%(asctime)s - [%(name)-25s] - [%(levelname)-8s] - %(message)s"


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.log_buffer = deque(maxlen=capacity)

    def emit(self, record):
        log_message = self.format(record)
        self.log_buffer.append(log_message)

    def get_logs(self):
        return list(self.log_buffer)


def _to_level(level):
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def setup_logger(
    logger_name,
    logfile,
    level="INFO",
    console=True,
    max_bytes=5 * 1024 * 1024,
    backup_count=5,
    propagate=False,
):
    logger = logging.getLogger(logger_name)
    resolved_level = _to_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = propagate

    for handler in list(logger.handlers):
        if getattr(handler, "_prompt_hmi_handler", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    log_dir = os.path.dirname(logfile)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if max_bytes and max_bytes > 0:
        file_handler = RotatingFileHandler(
            logfile,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler._prompt_hmi_handler = True

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(resolved_level)
        stream_handler._prompt_hmi_handler = True
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
