import time

def str2bool(v):
    v = str(v)
    return v.lower() in ("yes", "true", "t", "1")


def bool2Str(value):
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
