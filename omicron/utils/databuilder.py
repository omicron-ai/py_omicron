from collections import OrderedDict
from nltk.corpus import stopwords
from omicron.manage import ROOT_DIR, DATA_DIR, SRC_DIR, DST_DIR
import stanza
import json


header = ['turn', 'agent', 'text', 'tokens', 'intent', 'semantic_slot']
nlp = stanza.Pipeline('en')


def process_data():
    data = []
    with open(SRC_DIR, 'r') as file:
        turnindex = 0   
        for line in file:
            if line != '\n':
                row = process_row(turnindex, line)
                data.append(OrderedDict(zip(header, row)))
                turnindex += 1
    del data[-1]
    with open(DST_DIR, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=2)

    process_conversation()


def process_row(index, row):
    _row = [el.strip() for el in row.split('\t') if el != '']
    _row.insert(0, index)
    _row.insert(3, tokens(_row[2], False))
    return _row


def process_conversation(prettyprint: bool = True):
    with open(DST_DIR, 'r') as file:
        data = json.load(file)
        with open(f"{DATA_DIR}/sample_dialog_representation.txt", 'w') as ff:
            for turn in data:
                s = (f"turn: {turn['turn']}\n"
                     f"\ttext: {turn['text']}\n"
                     f"\trepresentation: ({turn['agent']}) -> "
                     f"{turn['intent']}({turn['semantic_slot']})\n"
                     f"\ttopics: {get_topic(turn)}\n\n")
                if prettyprint:
                    print(s)
                ff.write(s) 


def tokens(text: str, prettyprint: bool = False):
    text = text.strip('"')
    _tokens = {}
    for s in nlp(f"{text}").sentences:
        if prettyprint:
            print(f"\nSentence: {s.text}")
        for w in s.words:
            _tokens[f"{w.lemma}"] = w.to_dict()
            if prettyprint:
                print(f"\t{w.lemma}:")
                for item in _tokens[w.lemma]:
                    print(f"\t\t{item}:\t{_tokens[w.lemma][item]}")
    return _tokens


def get_topic(turn: dict):
    swords = []  # set(stopwords.words("english"))
    postags = ["NN", "NNP", "NNS", "NNPS", "PRP", "VB", "VBG", "VBD", "VBN", "VBP"]
    tkns = turn['tokens']
    tpcs = [w for w in tkns if tkns[w]['text'] not in swords and tkns[w]['xpos'] in postags]
    return tpcs


if __name__ == '__main__':
    process_data()

    

