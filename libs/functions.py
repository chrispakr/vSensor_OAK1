import time
import os
import fnmatch
import platform
import subprocess

def str2bool(v):
    v = str(v)
    return v.lower() in ("yes", "true", "t", "1")


def bool2Int(value):
    if value:
        return 1
    else:
        return 0


def wait_for_ping(host, interval=2.0, logger=None):
    """Block until a single ICMP ping to `host` succeeds, retrying every `interval` seconds."""
    if platform.system() == "Windows":
        cmd = ["ping", "-n", "1", "-w", "1000", host]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", host]
    while True:
        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                return
        except OSError as e:
            if logger is not None:
                logger.error(f"ping command failed: {e}")
        if logger is not None:
            logger.warning(f"no ping response from {host}, retrying in {interval}s...")
        time.sleep(interval)


def print_separator():
    print("####################################################################################################################")

class Ticker:
    def __init__(self):
        self.t = time.perf_counter()

    def __call__(self):
        dt = time.perf_counter() - self.t
        self.t = time.perf_counter()
        return 1000 * dt


class ValueHandler:
    def __init__(self, value):
        self._value = value
        self._last_value = value
        self.new_value_available = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if val is not None:
            self._last_value = self._value
            self._value = val
            if self._value != self._last_value:
                self.new_value_available = True
            else:
                self.new_value_available = False

    @property
    def previous_value(self):
        return self._last_value

    @previous_value.setter
    def previous_value(self, val):
        pass

    @property
    def new_value_available(self):
        if self._value != self._last_value:
            return True
        else:
            return False

    @new_value_available.setter
    def new_value_available(self, val):
        pass

    # def reset(self):
    #     self._last_value = self._value


class ValueHandlerInt:
    def __init__(self, value=None):
        self._value = value
        self._last_value = value
        self.new_value_available = False

    # def __get__(self):
    #     return int(self._value)

    def __int__(self):
        return int(self._value)

    def __set__(self, value):
        self._last_value = self._value
        self._value = value
        if self._value != self._last_value:
            self.new_value_available = True
        else:
            self.new_value_available = False

class WriteOnChange:
    """Writes a value to a symbol (e.g. an ADS symbol) only if it changed.

    Every ADS write is a network round trip (plus a verify read), so writing unchanged values every
    frame causes a lot of useless traffic. The value is still re-written every `refresh_interval`
    seconds, in case a write failed or the value was changed on the PLC side.
    """
    def __init__(self, symbol, refresh_interval=1.0):
        self._symbol = symbol
        self._refresh_interval = refresh_interval
        self._last_value = None
        self._last_write_time = 0.0

    def write(self, value):
        now = time.monotonic()
        if value != self._last_value or (now - self._last_write_time) >= self._refresh_interval:
            self._symbol.value = value
            self._last_value = value
            self._last_write_time = now


class IntervalTimer:
    def __init__(self, interval):
        self.interval = interval
        self.last_time = time.perf_counter()

    def reset(self):
        self.last_time = time.perf_counter()

    def is_time_to_update(self):
        now = time.perf_counter()
        if now - self.last_time > self.interval:
            self.last_time = now
            return True
        else:
            return False

def delete_old_files(directory, days=7, pattern="log_*.log*", logger=None):
    """Delete files in `directory` matching `pattern` that were last modified more than `days` ago.

    Uses the modification time (not ctime): on Linux ctime is the inode change time, and
    rotated backups (log_X.log.1, ...) keep the mtime of their last write.
    """
    cutoff = time.time() - (days * 86400)

    try:
        filenames = os.listdir(directory)
    except OSError as e:
        if logger is not None:
            logger.error(f"cannot list log directory {directory}: {e}")
        return

    for filename in filenames:
        if not fnmatch.fnmatch(filename, pattern):
            continue
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff:
                os.remove(file_path)
                if logger is not None:
                    logger.info(f"deleted old log file: {file_path}")
        except OSError as e:
            if logger is not None:
                logger.warning(f"cannot delete {file_path}: {e}")