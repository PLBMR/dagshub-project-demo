from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from const import *
import dagshub

print(M_MOD_INIT,'\n'+M_MOD_LOAD_DATA)
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH)
y_test = pd.read_csv(Y_TEST_PATH)

print(M_MOD_RFC)
with dagshub.dagshub_logger() as logger:
    rfc = RandomForestClassifier(n_estimators=1, random_state=0)
    # log the model's parameters
    logger.log_hyperparams(model_class=type(rfc).__name__)
    logger.log_hyperparams({'model': rfc.get_params()})

    # Train the model
    rfc.fit(X_train, y_train.values.ravel())
    y_pred = rfc.predict(X_test)

    # log the model's performances
    logger.log_metrics(
        {f'roc_auc_score':round(roc_auc_score(y_test, y_pred), ROUND_LEV)})
    print(M_MOD_SCORE, round(roc_auc_score(y_test, y_pred), ROUND_LEV))
