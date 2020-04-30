import networkx as nx
import threading


class User:
    def __init__(self, _user: str, _index: int):
        self._username = _user
        self._id = f"U-{_index}"

    @property
    def name(self):
        return self._username

    @name.setter
    def name(self, value):
        self._username = value

    @property
    def id(self):
        return self._id

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        return f"{self.id}"


class Post:

    def __init__(self, _post: tuple, _conv_index: int):
        self._conv_index = _conv_index
        self._id = f"P-{self.index}"
        self._dialog_act = _post[0]
        self._posted_by = _post[1]
        self._text = _post[2]
        self._terminals = _post[3]

    @classmethod
    def build(cls, _post, _index, debug=False):
        attribute = _post.attrib
        if debug:
            print(f"\n{attribute['user']} says: {_post.text}. \nIt is a {attribute['class']}")
        _p = (attribute['class'],
              attribute['user'],
              _post.text,
              [(t.attrib['pos'], t.attrib['word']) for t in _post.findall('./terminals/t')])
        return Post(_p, _index)

    @property
    def dialog_act(self):
        return self._dialog_act

    @dialog_act.setter
    def dialog_act(self, value):
        self._dialog_act = value

    @property
    def id(self):
        return self._id

    @property
    def user(self):
        return self._posted_by

    @user.setter
    def user(self, value):
        self._posted_by = value

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
        self._id = f"T-{index}"
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
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class TopicContext:
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


class TopicGraph(nx.MultiDiGraph):

    def __init__(self):
        super(TopicGraph, self).__init__()
        self.context = TopicContext()