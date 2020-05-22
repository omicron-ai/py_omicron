from omicron import stanza_nlp


def get_tokens(text: str, verbose: bool = False) -> dict:
    """
    Get CoreNLP Tokens from text using Stanford Stanza NLP.

    See https://stanfordnlp.github.io/stanza/data_objects#token for information.

    :param text: text string to generate Tokens
    :param verbose: write process log to console
    :return: dictionary of CoreNLP Tokens
    """
    text = text.strip('"')
    _tokens = {}
    for s in stanza_nlp(f"{text}").sentences:
        if verbose:
            print(f"\nSentence: {s.text}")
        for w in s.words:
            _tokens[f"{w.lemma}"] = w.to_dict()
            if verbose:
                print(f"\t{w.lemma}:")
                for item in _tokens[w.lemma]:
                    print(f"\t\t{item}:\t{_tokens[w.lemma][item]}")
    return _tokens


def get_topics(tkns: dict) -> list:
    """
    Get simple local utterance topics from Tokens dict.

    :param tkns: dictionary of CoreNLP Tokens
    :return: dictionary of local utterance topics as CoreNLP Tokens
    """
    swords = []  # set(stopwords.words("english"))
    postags = ["NN", "NNP", "NNS", "NNPS", "PRP", "VB", "VBG", "VBD", "VBN", "VBP"]
    return [tkns[w] for w in tkns if tkns[w]['text'] not in swords and tkns[w]['xpos'] in postags]
