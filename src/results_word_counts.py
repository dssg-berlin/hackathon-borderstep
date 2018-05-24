import pandas as pd
import os
from sklearn import model_selection, linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt

dir_data = 'data/processed'
df = pd.read_pickle(os.path.join(dir_data, 'word_count_full.pkl'))


feature_value_sum = 'CREMA 10_value_sum'
feature_dict = 'CREMA 10_dict'

df2 = df[['text', feature_dict, feature_value_sum]]
df3 = df2[df2[feature_dict].apply(len) > 0]

print(df3.head())
