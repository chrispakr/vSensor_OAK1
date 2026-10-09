import os
import time
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
os.environ['KIVY_LOG_MODE'] = 'PYTHON'


DEFAULT_LOG_FORMAT = "%(asctime)s - [%(name)-25s] - [%(levelname)-8s] - %(message)s"
DEFAULT_RING_BUFFER_MAX_AGE = 7 * 24 * 60 * 60  # 1 week, in seconds
DEFAULT_RING_BUFFER_MAX_ENTRIES = 20000  # safety cap against unbounded growth during a log burst
_HANDLER_MARKER = "_prompt_hmi_handler"


class RingBufferHandler(logging.Handler):
    """Keeps recent log lines in memory for a limited time window.

    Entries older than `max_age` seconds are evicted on the next emit/read.
    `max_entries` is a hard safety cap so a burst of log messages can't grow
    memory usage without bound even within the time window.
    """

    def __init__(self, max_age=DEFAULT_RING_BUFFER_MAX_AGE, max_entries=DEFAULT_RING_BUFFER_MAX_ENTRIES):
        super().__init__()
        self.max_age = max_age
        self.log_buffer = deque(maxlen=max_entries or None)

    def emit(self, record):
        log_message = self.format(record)
        self.log_buffer.append((time.time(), log_message))
        self._evict_expired()

    def _evict_expired(self):
        if not self.max_age:
            return
        cutoff = time.time() - self.max_age
        while self.log_buffer and self.log_buffer[0][0] < cutoff:
            self.log_buffer.popleft()

    def get_logs(self):
        self._evict_expired()
        return [message for _, message in self.log_buffer]


def _to_level(level):
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def setup_logger(
    logger_name,
    logfile,
    level:str="INFO",
    console:bool=True,
    max_bytes:int=5 * 1024 * 1024,
    backup_count:int=5,
    propagate:bool=False,
    ring_buffer_max_age:int=DEFAULT_RING_BUFFER_MAX_AGE,
    ring_buffer_max_entries:int=DEFAULT_RING_BUFFER_MAX_ENTRIES,
):
    logger = logging.getLogger(logger_name)
    resolved_level = _to_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = propagate

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
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
    setattr(file_handler, _HANDLER_MARKER, True)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(resolved_level)
        setattr(stream_handler, _HANDLER_MARKER, True)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if ring_buffer_max_entries and ring_buffer_max_entries > 0:
        ring_handler = RingBufferHandler(ring_buffer_max_age, ring_buffer_max_entries)
        ring_handler.setLevel(resolved_level)
        setattr(ring_handler, _HANDLER_MARKER, True)
        ring_handler.setFormatter(formatter)
        logger.addHandler(ring_handler)
        logger._ring_buffer_handler = ring_handler

    return logger


def get_ring_buffer_handler(logger_name):
    """Return the RingBufferHandler attached to the given logger, if any."""
    logger = logging.getLogger(logger_name)
    return getattr(logger, "_ring_buffer_handler", None)


def get_recent_logs(logger_name):
    """Return the most recent formatted log lines kept in memory for the given logger."""
    handler = get_ring_buffer_handler(logger_name)
    return handler.get_logs() if handler else []
