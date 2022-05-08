import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from const import *
import string

print(M_PRO_INIT, '\n' + M_PRO_LOAD_DATA)
data = pd.read_csv(RAW_DATA_PATH)

print(M_PRO_RMV_PUNC)
clean_text = data[TEXT_COL_NAME].map(lambda x: x.lower().replace('\n', ''))

print(M_PRO_LE)
y = data[TARGET_COL].map({CLASS_0: 0, CLASS_1: 1})

print(M_PRO_VEC)
# every column is 1-2 words and the value is the number of appearance in Email
email_text_list = clean_text.tolist()
vectorizer = CountVectorizer(encoding='utf-8', decode_error='ignore', stop_words='english',
                             analyzer='word', ngram_range=(1, 2), max_features=500)
X_sparse = vectorizer.fit_transform(email_text_list)
X = pd.DataFrame(X_sparse.toarray(), columns=vectorizer.get_feature_names())

print(M_PRO_SPLIT_DATA)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(M_PRO_SAVE_DATA)
X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH, index=False)
y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)
