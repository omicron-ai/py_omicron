from omicron.utilities import AtomicCounter
from collections import OrderedDict


class Agenda:
    def __init__(self):
        self.store = OrderedDict()
        self.processed = OrderedDict()
        self.counter = AtomicCounter()

    def add_signal(self, signal):
        sid = f"G-{self.counter.value}"
        self.store[sid] = signal
        self.counter.increment()
        return sid

    def process_signal(self, sid):
        signal = self.store[sid]
        self.processed[sid] = signal
        del self.store[sid]
        return signal

    def get_next_signal(self):
        sid = [(i[0], i[1]) for i in self.store.items()][0][0]
        signal = self.process_signal(sid)
        return sid, signal
