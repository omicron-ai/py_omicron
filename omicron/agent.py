#    Copyright (C) 2020 by
#    Ivan Leon <leoni@rpi.edu>
#    All rights reserved.
#    MIT license.
"""Create dialog agents for training and production.

See module train for a mechanism for training agents.
See module build for a mechanism for building agents for production.
"""

from typing import Union
from collections import OrderedDict
import networkx as nx
import uuid
from omicron import stanza_nlp, OMICRON_NAMESPACE
from omicron.constants import INDENT, IMAGE_DIR
from omicron.nlp import get_tokens, get_topics
from omicron.utilities import AtomicCounter, Origin
from omicron.agenda import Agenda
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab


class Agent:
    """The core omicron agent class."""

    def __init__(self, _agent_index: int):
        """
        Initialize empty agent from an agent index. There can only be one agent per index per world.

        :param _agent_index: integer describing agent index in world.
        """
        self.xid = _agent_index
        self.id = uuid.uuid5(OMICRON_NAMESPACE, f"{_agent_index}")
        self.memory = Memory((self.xid, self.id))
        self.sentinel = self.memory.sentinel
        self.agenda = Agenda()
        self.signals = []
        self.signal_count = AtomicCounter()

    def handle(self, _signal=None):
        """
        Handle agent signal. Signals are either input or output representations. If
        the signal is None, handle the next signal on agenda.

        :param _signal: representation of utterance to input or output
        """
        if _signal:
            self.agenda.add_signal(_signal)

        sid, signal = self.agenda.get_next_signal()

        if signal['mode'] == 'output':
            self.output(signal['turn'])
        if signal['mode'] == 'input':
            self.input(signal['turn'])

    def input(self, _signal, _verbose: bool = False):
        """
        Input utterance from dialog partner.

        :param _signal: signal representation to input
        :param _verbose: write process log to console
        """
        input_turn = Turn(self.xid, {'turn': _signal['turn'], 'text': _signal['text']})
        input_tokens = get_tokens(input_turn.text)
        input_topics = get_topics(input_tokens)
        input_rep = Plan(self.xid, {'turn': _signal['turn'],
                                    'intent': _signal['__intent__'],
                                    'sem_slot': _signal['__semantic_slot__'],
                                    'agent': _signal['agent'],
                                    'directed_to': input_turn.directed_to,
                                    'tokens': input_tokens,
                                    'topics': input_topics})

        self.memory.add_input(input_turn, input_rep)

        if _verbose:
            print(f"{INDENT}AGENT {self.xid}\t<INPUT>:\t{_signal['text']}")

    def output(self, _signal: dict, _verbose: bool = False):
        """
        Output utterance to dialog partner.

        :param _signal: signal representation to output
        :param _verbose: write process log to console
        """
        output_turn = Turn(self.xid, {'turn': _signal['turn'], 'text': _signal['text']})
        output_tokens = get_tokens(output_turn.text)
        output_topics = get_topics(output_tokens)
        output_rep = Plan(self.xid, {'turn': _signal['turn'],
                                     'intent': _signal['intent'],
                                     'sem_slot': _signal['semantic_slot'],
                                     'agent': _signal['agent'],
                                     'directed_to': output_turn.directed_to,
                                     'tokens': output_tokens,
                                     'topics': output_topics})

        self.memory.add_output(output_turn, output_rep)

        if _verbose:
            print(f"{INDENT}AGENT {self.xid}\t<OUTPUT>:\t{_signal['intent']}({_signal['semantic_slot']}) -> {_signal['text']}")

    def render_memory(self, filename: str = None):
        """Render memory as a DOT graph. (WIP)
        """

        # val_map = {'A': 1.0,
        #            'D': 0.5714285714285714,
        #            'H': 0.0}

        # val_map = {'A': 1.0,
        #            'T': 0.6,
        #            'R': 0.6,
        #            'O': 0.0}
        #
        # values = [val_map.get(node.type, 0.45) for node in self.memory.nodes()]
        # edge_labels=dict([((u,v,),d['label'])
        #                   for u,v,d in self.memory.edges(data=True)])
        # red_edges = [('C','D'),('D','A')]
        # edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]

        # pos=nx.spring_layout(self.memory)
        # nx.draw_networkx_edge_labels(self.memory,pos,edge_labels=edge_labels)
        # nx.draw(G,pos, node_color = values, node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
        # nx.draw(self.memory,pos, node_color = values, node_size=1500,edge_cmap=plt.cm.Reds)
        # pylab.show()
        if filename is None:
            filename = f"a{self.xid}_memory.png"
        A = nx.nx_agraph.to_agraph(self.memory)
        A.layout(prog="dot")
        # print(A)
        A.draw(f"{IMAGE_DIR}/{filename}")
        # pos = nx.nx_agraph.graphviz_layout(self.memory, prog='dot')
        # pass

    def __str__(self):
        return f"_agent_{self.xid}_"

    def __repr__(self):
        return f"<_AGENT_{self.id}_>"


class Sentinel:
    """Abstract in-memory agent representation."""
    def __init__(self, seed):
        self.id = uuid.uuid5(OMICRON_NAMESPACE, f"{seed[1]}")
        self.agent = OrderedDict(zip(['SIMPLE', 'UUID'], seed))
        self.type = "A"

    def __str__(self):
        return f"_agent_{self.agent['SIMPLE']}_"

    def __repr__(self):
        return f"<_AGENT_{self.id}_>"


class Memory(nx.MultiDiGraph):
    """Internal graph representing agent memory."""
    def __init__(self, seed):
        super(Memory, self).__init__(rankdir="LR", mode="scale")
        self.sentinel = Sentinel(seed)
        self.agents = {}
        self.context = Context()
        self.turn_count = AtomicCounter()
        self.origin = Origin()
        super().add_node(self.origin, label=f"{self.origin}")
        super().add_node(self.sentinel, label=f"SELF_{self.sentinel}")
        super().add_edge(self.sentinel, self.origin, tag='IN')

    def add_agent(self, agent_index):
        agent_id = uuid.uuid5(OMICRON_NAMESPACE, f"{agent_index}")
        sentinel = Sentinel((agent_index, agent_id))
        self.agents[sentinel.agent['SIMPLE']] = sentinel
        self.add_node(sentinel, label=f"AGENT_{sentinel}")
        self.add_edge(sentinel, self.origin, tag='IN')

    def add_input(self, turn, rep):
        if rep.directed_to not in self.agents.keys():
            self.add_agent(rep.directed_to)

        self.add_node(turn)
        self.add_node(rep)
        self.add_edge(self.agents[rep.agent], turn, label="GENERATES")
        self.add_edge(turn, self.sentinel, label="INPUT")
        self.add_edge(self.sentinel, rep, label="PRODUCES")
        self.add_edge(rep, turn, label="REPRESENTS")

    def add_output(self, turn, rep):
        if rep.directed_to not in self.agents.keys():
            self.add_agent(rep.directed_to)
        self.add_node(turn)
        self.add_node(rep)
        self.add_edge(self.sentinel, turn, label="GENERATES")
        self.add_edge(turn, self.agents[rep.directed_to], label="INPUT")
        self.add_edge(self.sentinel, rep, label="PRODUCES")
        self.add_edge(rep, turn, label="REPRESENTS")


class Context:
    """Internal representation of dialog context."""
    def __init__(self, n: int = 10, debug: bool = False):
        self._store = []
        self._n = n
        self._topic_index = AtomicCounter()
        if debug:
            print(f"STORE: {self.store}\nN: {self.n}\nTOPIC INDEX: {self.index}")

    def update(self, topic: Union['Topic', None] = None, debug: bool = False):
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
                    if debug:
                        print(f"\nFOUND INTERSECTION: {t.id}"
                              f"\nTOPIC CUTOFF: {t.cutoff}"
                              f"\nTOPIC COUNT: {t.count}")
                    break
            if not found:
                if debug:
                    print(f"NEW TOPIC: {topic.id}")
                self.add_topic(topic)
        else:
            if debug:
                print(f"TOPIC IS NONE.")
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
        if debug:
            print(f"STORE LOSS: {loss}")

    def add_topic(self, topic: 'Topic', debug: bool = False):
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


