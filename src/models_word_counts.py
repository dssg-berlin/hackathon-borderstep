import pandas as pd
import os
from sklearn import model_selection, linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt

dir_data = 'data/processed'
df = pd.read_pickle(os.path.join(dir_data, 'word_count_full.pkl'))
df = df[df.code_green != 2]
df['is_green'] = df.code_green <= 2
df['depth'] = df['deepth']
df = df.drop(columns=['Zusätzlich_value_sum'])

cs = [c for c in df.columns if c.endswith('sum')]
df2 = df[cs + ['url_2', 'text', 'depth', 'code_green', 'is_green']]
df2['text_len'] = df2['text'].apply(lambda s: len((' '.join(s)).split(' ')) if isinstance(s, list) else 1)

df3 = df2.groupby('url_2').sum()
# df3 = df3.reset_index()
df3['is_green'] = df3['is_green'] > 0.5
# df3[cs] = df3[cs] > 0.5  # not good

df3b = df2.groupby('url_2').size()
df3b.name = 'counts'
df3 = pd.concat([df3, df3b], axis=1)

print(df3.is_green.value_counts(normalize=True))
print(df3.is_green.value_counts())

df4 = df3[(df3.is_green) & (df3.counts < 100)]

df3[(df3.is_green) & (df3.counts < 100)].depth.hist(bins=100)
plt.figure()
df3[(~df3.is_green) & (df3.counts < 100)].depth.hist(bins=100)



# Investigate specific feature
feature = 'CREMA 10_value_sum'

df3[df3.is_green][feature].hist(bins=100)
plt.figure()
df3[(~df3.is_green) & (df3.counts < 100)].depth.hist(bins=100)



if True:
    df3['total'] = df3[cs].sum(axis=1)
    for c in cs:
        # df3[c] = df3[c] / df3['total']
        df3[c] = df3[c] / df3['text_len']
    df3 = df3.fillna(0)

    df_train, df_test = model_selection.train_test_split(df3, train_size=0.8)

    print(df_test.is_green.value_counts(normalize=True))
    print(df_test.is_green.value_counts())


    # Upsample green class

    df_train_x = df_train[cs]
    df_test_x = df_test[cs]
    df_train_y = df_train['is_green']
    df_test_y = df_test['is_green']

    model = linear_model.LogisticRegression()

    ratio = df_train_y.mean()

    sample_weight = np.zeros(len(df_train_y))
    sample_weight[df_train_y] = 1 / ratio
    sample_weight[~df_train_y] = 1 / (1 - ratio)
    sample_weight /= sample_weight.sum()

    model.fit(df_train_x, df_train_y, sample_weight=sample_weight)
    y_test_pred = model.predict(df_test_x)
    print(model.score(df_train_x, df_train_y))
    print(model.score(df_test_x, df_test_y))

    # y_pred = model.predict(df_test_x)

    df5 = pd.DataFrame({'scores': model.coef_[0], 'feature': cs})
    print(df5.sort_values('scores'))

    # df_train.sum(axis=1).sort_values(ascending=False).head(20)

    metrics.confusion_matrix(df_train_y, model.predict(df_train_x))
    metrics.confusion_matrix(df_test_y, y_test_pred)
    metrics.classification_report(df_test_y, y_test_pred)

    df6 = df[df['CREMA 10_value_sum'] > 0]
    df6 = df[df['CEPA 6_value_sum'] > 0]

    df6.text.iloc[0]
    df6.is_green.value_counts()

    for c in cs:
        dfx = df[df[c] > 0].is_green
        total = len(dfx)
        ratio = 0 if total == 0 else dfx.sum() / total
        # print(dfx)
        # ratio = 0 if len(dfx) == 0 or dfx[True] == 0 else dfx[True] / dfx.sum()
        print('{}: {}, {}'.format(c, dfx.sum(), ratio))
