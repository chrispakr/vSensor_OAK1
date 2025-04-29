import os
import logging
import logging.config
from collections import deque
os.environ['KIVY_LOG_MODE'] = 'PYTHON'

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


def setup_logger(logger_name, logfile):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.INFO)
    # create console handler with a higher log level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - [%(name)-30s] - [%(levelname)-8s] - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger