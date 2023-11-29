import time

def str2bool(v):
    v = str(v)
    return v.lower() in ("yes", "true", "t", "1")


def bool2Int(value):
    if value:
        return 1
    else:
        return 0


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