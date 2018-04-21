import pandas as pd
import os

# Extract keywords
data_dir = '../data/processed/DSSG'
keyword_filename = os.path.join(data_dir, 'GEMO-Schlagwortkatalog_20180312.xlsx')
dfk = pd.read_excel(keyword_filename, skiprows=1)

dfk.columns = ['domain', 'egss', 'field', 'keywords', 'extra', 'notes']
dfk = dfk.drop(['field', 'notes'], axis=1)

dfk = dfk.fillna('')

d = {}
for k, ds in dfk.iterrows():

    if ds.domain is not '':
        domain = ds.domain
        egss = ds.egss if ds.egss != '' else domain
        d[domain] = {}
        d[domain][egss] = ds.keywords + ', ' + ds.extra
    else:
        if ds.egss is not '':
            egss = ds.egss
            d[domain][egss] = ds.keywords + ', ' + ds.extra
        else:
            d[domain][egss] += ', ' + ds.keywords + ', ' + ds.extra


for k, v in d.items():
    print('{}: {}'.format(k, len(v)))
