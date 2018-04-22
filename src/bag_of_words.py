from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

df_fin = pd.read_pickle('hackathon-borderstep/data/processed/lemma_full.pkl')

training_data = df_fin[~df_fin.text_lemma.isnull()]


training_data_main_page= training_data[training_data.deepth==0]

X_raw = training_data_main_page.text_lemma.map(lambda x : ' '.join(x)).values
y = training_data_main_page.code_green.map(lambda x: 1 if x in [1,2] else 0)

# prepro
# bag of words
count_vect = CountVectorizer(min_df = 20, max_df = 50)
bag_model = count_vect.fit(X_raw)
X_train_counts = bag_model.transform(X_raw)
X_train_counts.shape
# tf_idf
tfidf_transformer = TfidfTransformer()
tfidf_model = tfidf_transformer.fit(X_train_counts)
X = tfidf_model.transform(X_train_counts)
X.shape

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# train and evaluate
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print('acc: ',acc)
confusion_matrix(y_test, y_pred)