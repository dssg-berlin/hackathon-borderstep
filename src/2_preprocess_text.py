from collections import Counter

import pandas as pd
import numpy as np
import spacy


def lemmatize_stopwords(doc):
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]


def preprocess_text(text):
    if isinstance(text, list) and len(text) > 0:
        joined_text = ''.join(np.stack(text)[0])
        return lemmatize_stopwords(nlp(joined_text))
    else:
        return None


def naive_lang_detect(text: list) -> 'str':
    """Naive language detector. Uses 'en' and 'de_core_news_sm' spacy packages
    to compare which contains the majority of the text terms.
    Any non-matching input (eg. None, empty list, etc) default to `?`
    Prints relevant message if language package is missing

    Arguments:
    -------
    text: list of str. The input text.

    Returns:
    -------
    language: str"""

    detection_score = Counter('?')
    languages = {
        'en': 'en',
        'de': 'de_core_news_sm'
    }
    try:
        _ = text[0]
    except (IndexError, TypeError):
        # non-list provided
        return '?'
    for lang, package in languages.items():
        try:
            nlp = spacy.load(package)
        except NameError:
            print(f"Cannot load package {package} in spacy. "
                  f"Try `python -m spacy download {package}`")
            return '?'
        for word in ','.join(text).split():
            if word in nlp.vocab:
                detection_score[lang] += 1
    return detection_score.most_common(1)[0][0]


def naive_lang_evaluate(text: list, package='de_core_news_sm') -> float:
    """Naive language evaluator. Uses provided spacy package to evaluate the
    ratio of words in given text that exist in package vocab.
    Any non-matching input (eg. None, empty list, etc) default to 1
    Prints relevant message if language package is missing.

    Arguments:
    -------
    text: list. The input list of texts.

    Returns:
    -------
    language: str"""

    try:
        _ = text[0]
    except (IndexError, TypeError):
        # non-list provided
        return 1
    try:
        nlp = spacy.load(package)
        _ = nlp.vocab
    except NameError:
        print(f"Cannot load package {package} or package has no vocab. "
              f"Try `python -m spacy download {package}")
    tokens = ','.join(text).split()
    matched = sum(map(lambda x: x in nlp.vocab, tokens))
    return matched / len(tokens)


data_dir = '../data/processed/DSSG/'
input_file = 'GEMO_2016.pkl.gz'
output_file = 'GEMO_2016_prep.pkl.gz'

nlp = spacy.load('de_core_news_sm')
df_text = pd.read_pickle(data_dir + input_file)
df_text['lemma_words'] = df_text.text.apply(preprocess_text)
df_text['lang'] = df_text.lemma_words.apply(naive_lang_detect)
df_text.to_pickle(data_dir + output_file)
