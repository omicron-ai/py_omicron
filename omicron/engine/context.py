from typing import Union

from omicron.utils.utilities import AtomicCounter


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