

class Mechanism:

    def __init__(self):
        """initialize mechanism here with default properties and methods for all Mechanisms.

        Any method called from this class should raise a NotImplementedError
        """
        pass


class InternalMechanism(Mechanism):
    """Internal Mechanism object
    """
    @classmethod
    def build(cls):
        pass


class InputMechanism(Mechanism):
    """Internal Mechanism object
    """
    @classmethod
    def build(cls):
        pass


class OutputMechanism(Mechanism):
    """Internal Mechanism object
    """
    @classmethod
    def build(cls):
        pass
