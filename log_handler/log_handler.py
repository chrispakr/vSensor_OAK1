import os
import logging
import logging.config
import threading
from logging.handlers import RotatingFileHandler
from collections import deque


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity):
        super().__init__()
        self.log_buffer = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record):
        log_message = self.format(record)
        with self._lock:
            self.log_buffer.append(log_message)

    def get_logs(self):
        with self._lock:
            return list(self.log_buffer)


def setup_logger(logger_name, logfile, level=logging.INFO, ring_capacity=100):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger, next(
            (h for h in logger.handlers if isinstance(h, RingBufferHandler)), None
        )

    logger.setLevel(level)

    os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - [%(name)-30s] - [%(levelname)-8s] - %(message)s')

    file_handler = RotatingFileHandler(logfile, maxBytes=10_000_000, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    ring_handler = RingBufferHandler(ring_capacity)
    ring_handler.setLevel(level)
    ring_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addHandler(ring_handler)

    return logger, ring_handler