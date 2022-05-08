import os

PREFIX = ".." if os.path.dirname(os.getcwd()) == "src" else ""

# Values
TEXT_COL_NAME = 'text'
TARGET_COL = 'label'
CLASS_0 = 'ham'
CLASS_1 = 'spam'

# Path
RAW_DATA_PATH = os.path.join(PREFIX, 'data/enron.csv')
X_TRAIN_PATH = os.path.join(PREFIX, 'data/X_train.csv')
X_TEST_PATH = os.path.join(PREFIX, 'data/X_test.csv')
Y_TRAIN_PATH = os.path.join(PREFIX, 'data/y_train.csv')
Y_TEST_PATH = os.path.join(PREFIX, 'data/y_test.csv')

# Messages
M_PRO_INIT = '[DEBUG] Preprocessing raw data'
M_PRO_LOAD_DATA = '     [DEBUG] Loading raw data'
M_PRO_RMV_PUNC = '     [DEBUG] Removing punctuation from Emails'
M_PRO_LE = '     [DEBUG] Label encoding target column'
M_PRO_VEC = '     [DEBUG] vectorizing the emails by words'
M_PRO_SPLIT_DATA = '     [DEBUG] Splitting data to train and test'
M_PRO_SAVE_DATA = '     [DEBUG] Saving data to file'

M_MOD_INIT = '[DEBUG] Initialize Modeling'
M_MOD_LOAD_DATA = '     [DEBUG] Loading data sets for modeling'
M_MOD_RFC = '     [DEBUG] Runing Random Forest Classifier'
M_MOD_SCORE = '     [INFO] Finished modeling with AUC Score:'
ROUND_LEV = 3
