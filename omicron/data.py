from collections import OrderedDict
from omicron.utils import SRC_PATH
from omicron.nlp import tokens, get_topic
import json
from pprint import pprint

header = ['turn', 'agent', 'text', 'tokens', 'intent', 'semantic_slot',]


def process_data(filedir: str = SRC_PATH, prettyprint: bool = False):
    def _process_row(_index, _row):
        _row = [el.strip() for el in _row.split('\t') if el != '']
        _row.insert(0, _index)
        _row.insert(3, tokens(_row[2], False))
        return _row

    _data = []
    with open(filedir, 'r') as file:
        turnindex = 0
        for line in file:
            if line != '\n':
                row = _process_row(turnindex, line)
                _data.append(OrderedDict(zip(header, row)))
                turnindex += 1
    del _data[-1]

    return _data


def write_files(_data, prettyprint: bool = False):
    from omicron.utils import JSON_PATH, SEM_PATH
    with open(JSON_PATH, 'w') as jsonfile:
        json.dump(_data, jsonfile, indent=2)

    with open(SEM_PATH, 'w') as semfile:
        for turn in _data:
            s = (f"turn: {turn['turn']}\n"
                 f"\ttext: {turn['text']}\n"
                 f"\trepresentation: ({turn['agent']}) -> "
                 f"{turn['intent']}({turn['semantic_slot']})\n"
                 f"\ttopics: {get_topic(turn)}\n\n")
            #    f"\tturn: {turn}\n\n")
            if prettyprint:
                print(s)
            semfile.write(s)

    # with open(COMP_DIR, 'w') as compfile:
    #     for turn in data:


def build_dialog(_data, _to_file: bool = False):
    from omicron.utils import A0, A1, COMPOSITE

    def _input_turn(_turn):
        input_turn = OrderedDict({"turn": _turn["turn"],
                                  "text": _turn["text"],
                                  "__tokens__": _turn["tokens"],
                                  "__intent__": _turn["intent"],
                                  "__semantic_slot__": _turn["semantic_slot"]})
        return input_turn

    _a0 = [{"mode": "output", "turn": t} if t["agent"] == "0" else {"mode": "input", "turn": _input_turn(t)} for t in _data]
    _a1 = [{"mode": "output", "turn": t} if t["agent"] == "1" else {"mode": "input", "turn": _input_turn(t)} for t in _data]
    if _to_file:
        with open(f"{A0}", 'w') as _a0_file:
            json.dump(_a0, _a0_file, indent=2)
        with open(f"{A1}", 'w') as _a1_file:
            json.dump(_a1, _a1_file, indent=2)
        with open(f"{COMPOSITE}", 'w') as _dialog_file:
            json.dump(_data, _dialog_file, indent=2)

    return _a0, _a1, _data


def build_script(_data, _to_file: bool = False):
    from omicron.utils import SCRIPT_DIR


    pass




if __name__ == '__main__':
    data = process_data()
    _, _, _ = build_dialog(data, True)
