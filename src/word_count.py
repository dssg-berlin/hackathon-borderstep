import os
import re
from collections import Counter
import pandas as pd
import numpy as np


def keyword_filter(entry):
    """From entries in the excel cell describe keywords
    Keyword Arguments:
    entry --
    """
    negatives_p = entry.split('NICHT')

    keyword = {"desired_keywords": split_cons(negatives_p[0])}
    if len(negatives_p) == 2:
        n_content = split_cons(negatives_p[1])
        keyword["avoid_keywords"] = n_content

    return keyword


def split_cons(content):
    """
    Keyword Arguments:
    content --
    """
    content = content.split(',')
    content = [a.strip() for a in content if a.strip() != '']
    return [re.sub(r'\*', r'\w*', x) for x in content]


def extract_keywords(ds):
    content = ds.keywords + ', ' + ds.extra
    content = re.sub('[\(\)]', ',', content)
    return keyword_filter(content)


def join_keywords(old, new):
    if old['desired_keywords'] and new['desired_keywords']:
        old['desired_keywords'] += new['desired_keywords']

    if old.get('avoid_keywords', None):
        if new.get('avoid_keywords', None):
            old['avoid_keywords'] += new['avoid_keywords']

    elif new.get('avoid_keywords', None):
        old['avoid_keywords'] = new['avoid_keywords']

    return old


def define_classes(dfk):
    """From the dataframed excel file of categories return a dictionary
    Keyword Arguments:
    dfk --
    """

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
                d[domain][egss] = join_keywords(d[domain][egss],
                                                extract_keywords(ds))
    return d


def count_keywords(keyword, text):
    """Count occuring keywords in text
    Keyword Arguments:
    keyword --
    """
    if len(keyword) == 0:
        return []

    target = re.compile('|'.join(r'\b%s\b' % w for w in keyword), re.I)
    counts = Counter(re.findall(target, text))
    return counts


def find_keywords(keywords, webs):
    """
    Keyword Arguments:
    webs --
    """
    rec = []
    for entry, row in webs.iterrows():
        if isinstance(row['text'], float):
            continue

        description = " ".join(row['text'])

        if keywords.get('avoid_keywords', None):
            unwanted = count_keywords(keywords.get(
                'avoid_keywords', None), description)
            if unwanted:
                continue

        counts = count_keywords(keywords['desired_keywords'], description)

        if len(counts) > 0:
            rec.append((entry, counts))
    return rec


if __name__ == '__main__':
    # Extract keywords
    data_dir = '../data/processed/DSSG'
    keyword_filename = os.path.join(
        data_dir, 'GEMO-Schlagwortkatalog_20180312.xlsx')
    dfk = pd.read_excel(keyword_filename, skiprows=1)

    dfk.columns = ['domain', 'egss', 'field', 'keywords', 'extra', 'notes']
    dfk = dfk.drop(['field', 'notes'], axis=1)

    dfk = dfk.fillna('')
    classes = define_classes(dfk)

    webs = pd.read_pickle('data/processed/merge_2015_2016.pkl')
    webs = webs.reset_index(drop=True)

    ds_list = []
    for domain, egss in classes.items():
        for column, keywords in egss.items():
            entries = find_keywords(keywords, webs)
            ds = pd.Series({k: v for k, v in entries})
            ds.name = column
            ds_list.append(ds)

    df = pd.concat(ds_list, axis=1)
    df = df.reindex(np.arange(len(webs)))
    df = df.fillna('')
    df_value_sum = df.applymap(lambda d: sum(
        d.values()) if isinstance(d, dict) else 0)
    # df_count = df.applymap(len)
    # df_words = df.applymap(lambda d: list(d.keys()) if isinstance(d, dict) else 0)

    df.columns = ['{}_dict'.format(c) for c in df.columns]
    df_value_sum.columns = ['{}_value_sum'.format(
        c) for c in df_value_sum.columns]

    df_out = pd.concat([webs, df, df_value_sum], axis=1)

    dir_out = 'data/processed'
    pd.to_pickle(df_out, os.path.join(dir_out, 'word_count_2015_2016.pkl'))

    # pd.to_pickle(df, os.path.join(dir_out, 'word_counts_2015_2016_dicts.pkl'))
    # pd.to_pickle(df_value_sum, os.path.join(
    #     dir_out, 'word_counts_2015_2016_value_sum.pkl'))

    # pd.to_pickle(df_count, os.path.join(dir_out, 'word_counts_2015_2016_counts.pkl'))
    # pd.to_pickle(df_words, os.path.join(dir_out, 'word_counts_2015_2016_words.pkl'))
