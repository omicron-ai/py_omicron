from omicron.agent import Agent, AtomicCounter
from omicron.data import process_data, build_dialog
from omicron.utils import A0, A1, COMPOSITE, INDENT


def demo(verbose: bool = False):
    _data = process_data()
    _a0_script, _a1_script, _data = build_dialog(_data)
    _global_dialog_index = AtomicCounter()
    _continue = True

    agents = {"0": Agent(0), "1": Agent(1)}
    script = {"0": _a0_script, "1": _a1_script}

    _slate = []

    def _process_turn(_script: dict, _i: int):

        _a0_turn = _script["0"][_i]
        _a1_turn = _script["1"][_i]

        _global_turn = {}
        if _a0_turn['mode'] == "output":
            _global_turn["output"] = {"agent": agents["0"],
                                      "turn": _a0_turn}
            _global_turn["input"] = {"agent": agents["1"],
                                     "turn": _a1_turn}
        elif _a1_turn['mode'] == "output":
            _global_turn["output"] = {"agent": agents["1"],
                                      "turn": _a1_turn}
            _global_turn["input"] = {"agent": agents["0"],
                                     "turn": _a0_turn}

        # print(f"OUTPUT: {_global_turn['output']}")
        # print(f"INPUT: {_global_turn['input']}")
        _global_turn["output"]["agent"].handle(_global_turn["output"]["turn"])
        _global_turn["input"]["agent"].handle(_global_turn["input"]["turn"])

        # if mode.upper() == "INPUT":
        #     _agent.input(_script[_i][mode])
        # if mode.upper() == "OUTPUT":
        #     _slate.append(_agent.output(_script[_i][mode]))
        # elif mode.upper() == "INPUT":

    while _continue:
        d_index = _global_dialog_index.value
        if verbose:
            print(f"\n{INDENT}---\n")
            [print(f"{INDENT}{e}: {_data[d_index][e]}") for e in _data[d_index] if e != "tokens" and e != "agent"]
            print(f"\n{INDENT}{'Agent 0 -> Agent 1' if _data[d_index]['agent'] == f'{0}' else 'Agent 1 -> Agent 0'}\n")

        _process_turn(script, d_index)

        _global_dialog_index.increment()
        if _global_dialog_index.value >= len(_data):
            _continue = False

    agents["0"].render_memory()
    agents["1"].render_memory()


if __name__ == '__main__':
    demo()