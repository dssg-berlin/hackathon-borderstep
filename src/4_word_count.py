import pandas as pd
import numpy as np
import os
import re
from collections import Counter

# Extract keywords
data_dir = 'data/processed/DSSG'
keyword_filename = os.path.join(
    data_dir, 'GEMO-Schlagwortkatalog_20180312.xlsx')
dfk = pd.read_excel(keyword_filename, skiprows=1)

dfk.columns = ['domain', 'egss', 'field', 'keywords', 'extra', 'notes']
dfk = dfk.drop(['field', 'notes'], axis=1)

dfk = dfk.fillna('')


def extract_keywords(ds):
    content = ds.keywords + ', ' + ds.extra
    content = re.sub('[\(\)]', ',', content)
    content = content.split(',')
    content = [a.strip() for a in content if a.strip() != '']
    content = [re.sub('\*', '\w*', x) for x in content]
    return content


d = {}
for k, ds in dfk.iterrows():

    if ds.domain is not '':
        domain = ds.domain
        egss = ds.egss if ds.egss != '' else domain
        d[domain] = {}
        d[domain][egss] = extract_keywords(ds)
    else:
        if ds.egss is not '':
            egss = ds.egss
            d[domain][egss] = extract_keywords(ds)
        else:
            d[domain][egss] += extract_keywords(ds)

# webs = pd.read_pickle('data/processed/DSSG/GEMO_2016.pkl.gz')
webs = pd.read_pickle('data/processed/merge_2015_2016.pkl')
webs = webs.reset_index(drop=True)

def count_keywords(keyword):
    """Count occuring keywords in text
    Keyword Arguments:
    keyword --
    """
    if len(keyword) == 0:
        return []

    r = re.compile('|'.join(r'\b%s\b' % w for w in keyword), re.I)
    rec = []
    for entry, row in webs.iterrows():
        if isinstance(row['text'], float):
            continue

        description = " ".join(row['text'])
        counts = Counter(re.findall(r, description))
        if len(counts) > 0:
            rec.append((entry, counts))
    return rec


ds_list = []
for domain, egss in d.items():
    for column, keyword in egss.items():
        entries = count_keywords(keyword)
        ds = pd.Series({k: v for k, v in entries})
        ds.name = column
        ds_list.append(ds)

df = pd.concat(ds_list, axis=1)
df = df.reindex(np.arange(len(webs)))
df = df.fillna('')
df_value_sum = df.applymap(lambda d: sum(d.values()) if isinstance(d, dict) else 0)
# df_count = df.applymap(len)
# df_words = df.applymap(lambda d: list(d.keys()) if isinstance(d, dict) else 0)

df.columns = ['{}_dict'.format(c) for c in df.columns]
df_value_sum.columns = ['{}_value_sum'.format(c) for c in df_value_sum.columns]

df_out = pd.concat([webs, df, df_value_sum], axis=1)

dir_out = 'data/processed'
pd.to_pickle(df, os.path.join(dir_out, 'word_count_2015_2016.pkl'))

# pd.to_pickle(df, os.path.join(dir_out, 'word_counts_2015_2016_dicts.pkl'))
# pd.to_pickle(df_value_sum, os.path.join(
#     dir_out, 'word_counts_2015_2016_value_sum.pkl'))

# pd.to_pickle(df_count, os.path.join(dir_out, 'word_counts_2015_2016_counts.pkl'))
# pd.to_pickle(df_words, os.path.join(dir_out, 'word_counts_2015_2016_words.pkl'))
