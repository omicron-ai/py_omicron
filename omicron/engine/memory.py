import uuid
from collections import OrderedDict

import networkx as nx

from omicron import OMICRON_NAMESPACE
from omicron.engine.context import Context
from omicron.utils.utilities import AtomicCounter, Origin


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