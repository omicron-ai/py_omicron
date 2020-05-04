from typing import Union
from collections import OrderedDict
import networkx as nx
import threading
import uuid
from omicron import stanza_nlp, OMICRON_NAMESPACE
from omicron.utils import INDENT
from omicron.nlp import tokens


class Agent:
    def __init__(self, _agent: int):
        self._xid = _agent
        self._id = uuid.uuid5(OMICRON_NAMESPACE, f"{_agent}")
        self._memory = Memory((self.xid, self.id))
        _heartbeat_tempo = 0.25

    def input(self, _repr):
        # turn['tokens'] = tokens(turn['text'])
        # take in as input a dialog turn from another agent.
        # process turn, determine dialog act/ask, etc.
        # update memory with input turn
        print(f"{INDENT}AGENT {self.xid}\t<INPUT>:\t{_repr['text']}")

    def output(self, _repr: dict):
        # generate and output a dialog turn from a semantic representation
        # update memory with output turn
        print(f"{INDENT}AGENT {self.xid}\t<OUTPUT>:\t{_repr['intent']}({_repr['semantic_slot']})")

    @property
    def id(self):
        return self._id

    @property
    def xid(self):
        return self._xid

    @property
    def memory(self):
        return self._memory

    @property
    def sentinel(self):
        return self._sentinel

    def __str__(self):
        return f"_agent_{self.xid}_"

    def __repr__(self):
        return f"<_AGENT_{self.id}_>"


class Sentinel:
    def __init__(self, seed):
        self._id = uuid.uuid5(OMICRON_NAMESPACE, f"{seed[1]}")
        self._agent = OrderedDict(zip(['SIMPLE', 'UUID'], seed))

    @property
    def id(self):
        return self._id

    @property
    def agent(self):
        return self._agent

    def __str__(self):
        return f"_agent_{self.agent['SIMPLE']}_"

    def __repr__(self):
        return f"<_AGENT_{self.id}_>"


class Turn:
    def __init__(self, seed, turn):
        self._index = turn['turn']
        self._id = uuid.uuid5(OMICRON_NAMESPACE, f"{seed[1]}")
        self._intent = ""
        self._slot = ""
        self._agent = OrderedDict(zip(['SIMPLE', 'UUID'], seed))
        self._directed_to = f"1" if f"{self.agent['SIMPLE']}" == f"0" else f"0"
        self._text = ""
        self._terminals = ""


    @property
    def intent(self):
        return self._intent

    @intent.setter
    def intent(self, value):
        self._intent = value

    @property
    def id(self):
        return self._id

    @property
    def index(self):
        return self._index

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, value):
        self._agent = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, value):
        self._terminals = value

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        return f"{self.id}"


class Topic:

    def __init__(self, constituents, user, index, cutoff):
        self._id = f"TOPIC-{index}"
        self._constituents = constituents
        self._posted_by = user
        self._topic_index = index
        self._cutoff = cutoff
        self._counter = AtomicCounter()

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def user(self):
        return self._posted_by

    @user.setter
    def user(self, value):
        self._posted_by = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value

    @property
    def count(self):
        return self._counter.value

    def increment_counter(self, value=1):
        self._counter.increment(value)

    def increment_cutoff(self, value=1):
        self.cutoff = self.cutoff + value

    def constituents(self):
        return self._constituents

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
        return f"{self.id}"


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
        return f"{self.value}"


class Memory(nx.MultiDiGraph):
    def __init__(self, seed):
        super(Memory, self).__init__()
        self.sentinel = Sentinel(seed)
        self.context = Context()
        self.turn_count = AtomicCounter()
        self.origin = Origin()
        super().add_node(self.origin, label=f"{self.origin}")
        super().add_node(self.sentinel, label=f"SELF_{self.sentinel}")
        super().add_edge(self.sentinel, self.origin, tag="IN")

    @property
    def sentinel(self):
        return self._sentinel

    @sentinel.setter
    def sentinel(self, value):
        self._sentinel = value

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value

    @property
    def count(self):
        return self._turn_count

    @count.setter
    def count(self, value):
        self._turn_count = value

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value


class Context:
    def __init__(self, n: int = 10, debug: bool = False):
        self._store = []
        self._n = n
        self._topic_index = AtomicCounter()
        if debug:
            print(f"STORE: {self.store}\nN: {self.n}\nTOPIC INDEX: {self.index}")

    def update(self, topic: Union[Topic, None] = None, debug: bool = False):
        # update topic memory with current topic
        output = {"LINK": False,
                  "TOPIC": None}
        if debug: print("\n\n\\/ ========== DEBUG ========== \\/\n")

        if topic:
            if debug:
                print(f"TOPIC: {topic.id}\n")
                [print(t.rep()) for t in self.store]

            found = False
            for t in self.store:
                if topic.intersect(t):
                    output = {"LINK": True,
                              "TOPIC": t}
                    t.increment_cutoff(value=2)
                    found = True
                    if debug: print(f"\nFOUND INTERSECTION: {t.id}"
                                    f"\nTOPIC CUTOFF: {t.cutoff}"
                                    f"\nTOPIC COUNT: {t.count}")
                    break
            if not found:
                if debug: print(f"NEW TOPIC: {topic.id}")
                self.add_topic(topic)
        else:
            if debug: print(f"TOPIC IS NONE.")
        self.advance(debug=debug)
        if debug: print("\n/\\ ========== DEBUG ========== /\\\n")
        return output

    def advance(self, debug: bool = False):
        loss = len(self.store)
        for i, t in enumerate(self.store):
            t.increment_counter()
            if t.count == t.cutoff:
                self.remove_topic(i)
        loss -= len(self.store)
        if debug: print(f"STORE LOSS: {loss}")

    def add_topic(self, topic: Topic, debug: bool = False):
        self._store.append(topic)

    def remove_topic(self, index: int, debug: bool = False):
        del self._store[index]

    @property
    def store(self):
        return self._store

    def get_next_index(self):
        self._topic_index.increment()
        return self.index

    @property
    def index(self):
        return self._topic_index

    @property
    def n(self):
        return self._n


class Origin:
    def __str__(self):
        return f"Origin "

    def __repr__(self):
        return f"<ORIGIN>"
