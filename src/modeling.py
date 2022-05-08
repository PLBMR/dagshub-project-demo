from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from const import *


print(M_MOD_INIT,'\n'+M_MOD_LOAD_DATA)
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH)
y_test = pd.read_csv(Y_TEST_PATH)

print(M_MOD_RFC)
rfc = RandomForestClassifier(n_estimators=1, random_state=0)
rfc.fit(X_train, y_train.values.ravel())
y_pred = rfc.predict(X_test)
print(M_MOD_SCORE, round(roc_auc_score(y_test, y_pred),3))