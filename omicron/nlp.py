
from omicron import stanza_nlp


def tokens(text: str, prettyprint: bool = False):
    text = text.strip('"')
    _tokens = {}
    for s in stanza_nlp(f"{text}").sentences:
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
