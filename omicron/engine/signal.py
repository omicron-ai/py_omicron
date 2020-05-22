import uuid

from omicron import OMICRON_NAMESPACE
from omicron.utils.utilities import AtomicCounter
from enum import Enum

class Signal:

    class Status(Enum):
        RECEIVED = "RECEIVED"
        CONSUMED = "CONSUMED"

    def __init__(self, seed):
        self.id = uuid.uuid5(OMICRON_NAMESPACE, f"{seed}")
        self.type = "SIGNAL"
        pass


class Plan:
    def __init__(self, seed, rep):
        self.id = uuid.uuid5(OMICRON_NAMESPACE, f"{seed}")
        self.turn = rep['turn']
        self.intent = rep['intent']
        self.agent = rep['agent']
        self.directed_to = rep['directed_to']
        self.sem_slot = rep['sem_slot']
        self.tokens = rep['tokens']
        self.topic_tokens = rep['tokens']
        self.type = "P"

    def __str__(self):
        return f"_plan_{self.turn}_"

    def __repr__(self):
        return f"<_PLAN_{self.id}>_"


class Turn:
    def __init__(self, seed, turn):
        self.index = turn['turn']
        self.id = uuid.uuid5(OMICRON_NAMESPACE, f"{seed}")
        self.directed_to = f'1' if f"{seed}" == f'0' else f'0'
        self.text = turn['text']
        self.type = "T"

    def __str__(self):
        return f"_turn_{self.index}_"

    def __repr__(self):
        # return f"<_TURN_{self.id}_>"
        return f"<{self.__class__.__module__}.{self.__class__.__name__}: " \
               f"TURN-{self.index} at {hex(id(self))}>"


class Topic:

    def __init__(self, constituents, user, index, cutoff):
        self.id = f"TOPIC-{index}"
        self.constituents = constituents
        self.posted_by = user
        self.index = index
        self.cutoff = cutoff
        self._counter = AtomicCounter()

    @property
    def count(self):
        return self._counter.value

    def increment_counter(self, value=1):
        self._counter.increment(value)

    def increment_cutoff(self, value=1):
        self.cutoff = self.cutoff + value

    def constituents(self):
        return self.constituents

    def intersect(self, topic: 'Topic'):
        for c in self.constituents():
            if c in topic.constituents():
                return True
        return False

    def rep(self):
        s = f"{self.id}:\n"
        for c in self.constituents():
            s = f"{s}\t{c[0]} : [{c[1]}]\n"
        s = f"{s}\tCOUNT : {self.count}"
        return s

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        # return f"{self.id}"
        return f"<{self.__class__.__module__}.{self.__class__.__name__}: " \
               f"TOPIC-{self.index} at {hex(id(self))}>"