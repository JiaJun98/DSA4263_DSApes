import syspend
import preprocess_class as pre
import pandas as pd
import numpy as np
import utility
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, auc, roc_curve
import pickle
import torch
from imblearn.over_sampling import SMOTE

def ttsplit(df):
    
    train, test = train_test_split(df, test_size = 0.2, random_state = 4263, stratify = df['Sentiment'])
    return [train, test]


def train_model(model_type, x_train, y_train, x_test, y_test):
    if model_type == 'LogisticRegression':
        logreg(x_train, y_train, x_test,  y_test)
    
    if model_type == 'RandomForest':
        rf(x_train, y_train, x_test, y_test)
    
    if model_type == 'XGBoost':
        xgboost(x_train, y_train, x_test, y_test)
    
    utility.custom_print('\n' + str(model_type) + ' model succesfully trained', logger = logger)

    

def logreg(x_train, y_train, x_test, y_test):
    logreg = pickle.load(open('models/lr_model.sav', 'rb'))
    logreg.fit(x_train, y_train)
    lr_pred = logreg.predict(x_test)
    utility.custom_print('LogisticRegression model succesfully loaded\n', logger = logger)
    utility.custom_print('---------------------------------\n',logger = logger)
    utility.churn_eval_metrics(lr_pred, y_test, logger)


def rf(x_train, y_train, x_test, y_test):
    rf = pickle.load(open('models/rf_model.sav', 'rb'))
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    utility.custom_print('RandomForest model succesfully loaded\n', logger = logger)
    utility.custom_print('---------------------------------\n',logger = logger)
    utility.churn_eval_metrics(rf_pred, y_test, logger)



def xgboost(x_train, y_train, x_test, y_test):
    xgb = pickle.load(open('models/xgb_model.sav', 'rb'))
    xgb.fit(x_train, y_train)
    xgb_pred = xgb.predict(x_test)
    utility.custom_print('XGBoost model succesfully loaded\n', logger = logger)
    utility.custom_print('---------------------------------\n',logger = logger)
    utility.churn_eval_metrics(xgb_pred, y_test, logger)



if __name__ == "__main__":
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'non_bert_sentiment_config.yml') #change when done
    config_file = utility.parse_config(config_path)
    log_path = config_file['model']['log_path']
    model_name = config_file['model']['model_name']
    model_type = config_file['model']['model_type']
    data_path = config_file['model']['data_path']
    data_file = config_file['model']['data_file']
    home_folder = os.getcwd() #change when done

    data_df = pd.read_csv(os.path.join(home_folder, data_path, data_file))
    logger = open(os.path.join(curr_dir, log_path), 'w')
    df = pre.Dataset(data_df)
    df.create_tfidf(root_words_option = 2, remove_stop_words = True, lower_case = True, ngrams = (1,2), min_doc = 0.05, max_doc = 0.95)
    tfidf = pd.DataFrame(df.tfidf[1].toarray())
    tfidf['Time'] = df.date
    tfidf['Sentiment'] = df.sentiments
    tfidf = tfidf.replace({'positive': 1, 'negative':0})
    train, test = ttsplit(tfidf)
    x_train = train.drop(['Time', 'Sentiment'], axis = 1)
    y_train = train['Sentiment'].to_numpy()
    x_test = test.drop(['Time', 'Sentiment'], axis = 1)
    y_test = test['Sentiment'].to_numpy()
    oversample = SMOTE(random_state = 4263)
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    utility.custom_print("Train_val dataset loaded",logger = logger)
    train_model(model_type, x_train, y_train, x_test, y_test)
    utility.custom_print('\n---------------------------------\n',logger = logger)

