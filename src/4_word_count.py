import pandas as pd
import numpy as np
import os
import re
from collections import Counter

# Extract keywords
data_dir = '../data/processed/DSSG'
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
    content = [re.sub('\*', '.*', x) for x in content]
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

webs = pd.read_pickle('../data/processed/DSSG/GEMO_2016.pkl.gz')


def count_keywords(keyword):
    """Count occuring keywords in text
    Keyword Arguments:
    keyword --
    """
    if len(keyword) == 0:
        return []

    r = re.compile('|'.join('r\b%s\b' % w for w in keyword), re.I)
    rec = []
    for entry, row in webs.iterrows():
        if isinstance(row['text'], float):
            continue

        for par in row['text']:
            counts = Counter(re.findall(r, par))
            if len(counts) > 0:
                rec.append(entry)
    return rec


for domain, egss in d.items():
    for label, keyword in egss.items():
        entries = count_keywords(keyword)
        if len(entries) > 0:
            print(domain, label, entries)
