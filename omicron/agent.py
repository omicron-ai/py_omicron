"""
    Create dialog agents for training and production.

    Copyright (c) 2020 Ivan Leon
    Contact <leoni@rpi.edu>
"""

import networkx as nx
import uuid
from omicron import OMICRON_NAMESPACE
from omicron.engine.memory import Memory
from omicron.engine.signal import Plan, Turn
from omicron.utils.constants import INDENT, IMAGE_DIR
from omicron.utils.nlp import get_tokens, get_topics
from omicron.utils.utilities import AtomicCounter
from omicron.engine.agenda import Agenda


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
            print(f"{INDENT}AGENT {self.xid}\t<INPUT>:"
                  f"\t{_signal['text']}")

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
            print(
                f"{INDENT}AGENT {self.xid}\t<OUTPUT>:"
                f"\t{_signal['intent']}({_signal['semantic_slot']}) -> {_signal['text']}")

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


