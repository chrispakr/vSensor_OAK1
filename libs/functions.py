import time
import os
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

def delete_old_files(directory, days=3):
    now = time.time()
    cutoff = now - (days * 86400)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            file_date = filename.split("-")
            # print(file_date)


            # # Löschen Sie die Datei, wenn sie älter als 'cutoff' ist
            # if file_creation_time < cutoff:
            #     print(f"Lösche: {file_path}")
            #     os.remove(file_path)