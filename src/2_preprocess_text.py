import pandas as pd
import numpy as np
import spacy


def lemmatize_stopwords(doc):
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop ]

def preprocess_text(text):
    if isinstance(text,list) and len(text)>0:
        joined_text = ''.join(np.stack(text)[0])
        return lemmatize_stopwords(nlp(joined_text))
    else:
        return None


data_dir = '../data/processed/DSSG/'
input_file = 'GEMO_2016.pkl.gz'
output_file = 'GEMO_2016_prep.pkl.gz'

nlp = spacy.load('de_core_news_sm')
df_text = pd.read_pickle(data_dir + input_file)
df_text['lemma_words'] = df_text.text.apply(preprocess_text)
df_text.to_pickle(data_dir + output_file)
