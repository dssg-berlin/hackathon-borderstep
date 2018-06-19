import pandas as pd
import os
from sklearn import model_selection, linear_model, metrics
from sklearn import naive_bayes
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from functools import partial
import spacy

pd.options.display.max_colwidth = 100
pd.options.display.width = 160
pd.options.display.max_columns = 20


def join_dicts(dicts):
    dr = {}
    for d in dicts:
        if len(d) > 0:
            dr.update(d)
    return dr


def get_document_counts(list_of_dicts):
    """Every dict in the input list gives word frequencies for a single website."""

    d_all = defaultdict(int)
    for d in list_of_dicts:
        if len(d) > 0:
            for k in d.keys():
                d_all[k] += 1

    df3 = pd.Series(d_all)
    df3 = df3.sort_values(ascending=False)

    return df3


def transform_keys(counts, key_func):
    c = Counter()
    for k, v in counts.items():
        c[key_func(k)] += v
    return c


def combine_dicts(dicts):
    if len(dicts) == 0:
        return {}
    counter = Counter(dicts[0])
    for d in dicts[1:]:
        counter.update(d)
    return counter


nlp = spacy.load('de_core_news_sm')


def to_lemma(keyword):
    doc = nlp(keyword)
    if len(doc) != 1:
        print(keyword, len(doc))
        return ''
    else:
        token = doc[0]
        return token.lemma_


dir_data = 'data/processed'
df = pd.read_pickle(os.path.join(dir_data, 'word_count_full.pkl'))
df['depth'] = df['deepth']
df = df.drop(columns=['Zusätzlich_value_sum'])
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
df = df[df.url_2 != 'www.fahrschule-greenpoint.de']
df['is_green'] = df.code_green <= 2

# From keywords file.
class_names = sorted([c[:-10] for c in df.columns if c.endswith('value_sum')])

# print([c for c in class_names if c not in df.columns])

common_classes = ['cepa_1', 'cepa_2', 'cepa_3', 'cepa_4', 'cepa_5', 'cepa_6',
                  'cepa_7', 'cepa_8', 'cepa_9', 'crema_10', 'crema_11a',
                  'crema_11b', 'crema_12', 'crema_13a', 'crema_13b', 'crema_13c',
                  'crema_14', 'crema_15', 'crema_16']
for c in common_classes:
    df[c] = ~df[c].isnull() * 1


website_cols = ['name_company', 'url_2', 'is_green', 'code_green'] + common_classes
dict_cols = [c for c in df.columns if c.endswith('_dict')]
df2 = df[website_cols + dict_cols].groupby(website_cols).agg(join_dicts).reset_index()


keyword = 'Umwelt'
col = 'übergreifend_dict'
df3 = df2.copy()
df3[keyword] = df2[col].apply(lambda row: keyword in row)

## Check which words are most common.
df3 = get_document_counts(df2.übergreifend_dict)
print(df3.head())

## Select relevant dicts.
# dict_cols_selection = ['crema_13a_dict', 'crema_13b_dict', 'übergreifend_dict']
dict_cols_selection = ['{}_dict'.format(c) for c in common_classes]
df3 = df2.copy()
# df4 = df3[dict_cols_selection]

# Convert keys to lowercase.
cs = dict_cols_selection
df3.loc[:, cs] = df3.loc[:, cs].applymap(partial(transform_keys, key_func=lambda k: k.lower()))

# Apply lemmatizer to all keywords (only a very small effect).
keywords = sorted(set([x for xss in df3.loc[:, cs].applymap(lambda d: list(d.keys())).values
                       for xs in xss for x in xs]))
lemma_map = {k: to_lemma(k) for k in keywords}
lemma_map_effective = {k: v for k, v in lemma_map.items() if k != v}  # only 6 cases
df3.loc[:, cs] = df3.loc[:, cs].applymap(partial(transform_keys, key_func=lambda k: lemma_map.get(k, '')))

# Combine multiple dict columns into single one.
df3['combined_dict'] = df3.loc[:, cs].apply(lambda row: combine_dicts(row), axis=1)

## Create train test split.
df_train, df_test = model_selection.train_test_split(df3, train_size=0.8, random_state=1)

## Check which words occur in non-green websites.
df3t = get_document_counts(df_train.query('is_green == True').combined_dict).rename('is_green')
df3f = get_document_counts(df_train.query('is_green == False').combined_dict).rename('is_not_green')
df3 = pd.concat([df3t, df3f], axis=1, sort=True).fillna(0, downcast='infer')
df3['is_green_frac'] = df3['is_green'] / len(df2.query('is_green == True'))
df3['is_not_green_frac'] = df3['is_not_green'] / len(df2.query('is_green == False'))
df3 = df3[['is_green', 'is_not_green', 'is_green_frac', 'is_not_green_frac']]

print(df3.sort_values('is_green', ascending=False).head(20))
print(df3.sort_values('is_not_green', ascending=False).head(20))

df3['total'] = df3.is_green + df3.is_not_green
df3['green_ratio'] = df3.is_green / df3.is_not_green
print(df3.query('total >= 20').sort_values('green_ratio', ascending=False))

green_words = list(df3.query('total >= 20').index)
green_words = [w for w in green_words if w not in ['schutz']]

## Count websites with at least one green word.
df_train['has_green_word'] = df_train.combined_dict.apply(lambda d: any([x in green_words for x in list(d.keys())]))
print(df_train.groupby(['is_green', 'has_green_word']).size())
print(df_train.groupby(['has_green_word', 'is_green']).size())
print(df_train.has_green_word.value_counts())
print(df_train.has_green_word.value_counts(normalize=True))


## Try naive Bayes / logistic regression
for word in green_words:
    df_train[word] = df_train.combined_dict.apply(lambda d: word in d) * 1
    df_test[word] = df_test.combined_dict.apply(lambda d: word in d) * 1
df_X_train = df_train[green_words]
df_X_test = df_test[green_words]
df_y_train = df_train.is_green
df_y_test = df_test.is_green
# df_X = df3[green_words]
# df_y = df3.is_green

# df_X_train, df_X_test, df_y_train, df_y_test = model_selection.train_test_split(
#     df_X, df_y, test_size=0.2, random_state=1)

# model = naive_bayes.MultinomialNB()
model = linear_model.LogisticRegression()
model.fit(df_X_train, df_y_train)
y_pred_test = model.predict(df_X_test)

threshold = 0.7
print('evaluate on training set')
print(metrics.confusion_matrix(df_y_train, model.predict_proba(df_X_train)[:, 1] > threshold))
print(metrics.classification_report(df_y_train, model.predict_proba(df_X_train)[:, 1] > threshold))
print('evaluate on test set')
print(metrics.confusion_matrix(df_y_test, model.predict_proba(df_X_test)[:, 1] > threshold))
print(metrics.classification_report(df_y_test, model.predict_proba(df_X_test)[:, 1] > threshold))
# print(metrics.confusion_matrix(df_y_test, y_pred_test))
# print(metrics.classification_report(df_y_test, y_pred_test))

# Plot histogram of predicted probabilities, to be able to set threshold
if False:
    plt.hist(model.predict_proba(df_X_train)[:, 1], bins=100)
    plt.hist(model.predict_proba(df_X_test)[:, 1], bins=100)

# Investigate logistic regression coefficients.
if False:
    dfc = pd.DataFrame({'word': green_words, 'coef': model.coef_[0]})
    dfc.sort_values('coef', ascending=False)

# Investigate second peak
if False:
    df4 = df_train.copy()
    df4['proba'] = model.predict_proba(df_X_train)[:, 1]
    df4 = df4[['combined_dict', 'proba']]
    df4 = df4.query('0.23 < proba < 0.24')

if False:
    ## Show distribution of companies across classes:
    # print(df[common_classes + ['name_company']].groupby('name_company').agg(any).sum())
    print(df[common_classes + ['name_company']].drop_duplicates().sum())
    df[common_classes + ['name_company', 'code_green']].drop_duplicates().groupby('code_green').sum()

    df2 = df[common_classes + ['name_company', 'url_2', 'code_green']].drop_duplicates()


if False:
    df2 = df[['name_company', 'übergreifend_dict']].groupby('name_company').agg(join_dicts)

    d_all = defaultdict(int)
    for d in df2.übergreifend_dict:
        if len(d) > 0:
            for k in d.keys():
                d_all[k] += 1

    df3 = pd.Series(d_all)
    df3.sort_values(ascending=False).head(20)


if False:
    # Investigating the number of keywords per website.
    dict_cols_selection = ['{}_dict'.format(c) for c in common_classes]
    df3 = df2.copy()

    # Convert keys to lowercase.
    cs = dict_cols_selection
    df3.loc[:, cs] = df3.loc[:, cs].applymap(partial(transform_keys, key_func=lambda k: k.lower()))

    # Combine multiple dict columns into single one.
    df3['combined_dict'] = df3.loc[:, cs].apply(lambda row: combine_dicts(row), axis=1)

    df4 = df3.combined_dict[df3.combined_dict.apply(lambda x: len(x) > 0)].apply(len).value_counts().sort_index()
    df5 = df3[df3.combined_dict.apply(len) == 1]
    df5 = df3[df3.combined_dict.apply(len) > 100]

    df[df.url_2 == 'www.asc-abbruch.de'].T
    df5[['url_2', 'combined_dict', 'is_green', 'code_green']].head(20)
