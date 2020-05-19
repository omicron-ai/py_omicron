import threading


class AtomicCounter:
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        with self._lock:
            self.value += num
            return self.value

    def __str__(self):
        return f"{self.value}"

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__name__}: " \
               f"{self.value} at {hex(id(self))}>"


class Origin:
    def __init__(self):
        self.type = "O"

    def __str__(self):
        return f"Origin "

    def __repr__(self):
        return f"<ORIGIN>"
