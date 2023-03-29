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
    '''
    Returns a 80-20 train and test set for the given dataset. 
    @param df: List, numpy array, scipy-sparse matrix or pandas dataframe to operate the train-test split on
    '''
    train, test = train_test_split(df, test_size = 0.2, random_state = 4263, stratify = df['Sentiment'])
    return [train, test]


def predict_data(model_type, x_test):
    '''
    Predicts the given input data using the specified model_type and outputs the predictions (containing 0's or 1's) into a csv file.
        Parameters:
            model_type (str): A choice of 3 models are available. 'LogisticRegression', 'RandomForest' and 'XGBoost'
            x_test (pd.DataFrame, np.array): Input data to be predicted on
    '''
    if model_type == 'LogisticRegression':
        logreg = pickle.load(open(model_save_loc, 'rb'))
        lr_pred = pd.DataFrame({'Sentiment Prediction': logreg.predict(x_test)})
        lr_pred.to_csv(output_path)
    
    if model_type == 'RandomForest':
        rf = pickle.load(open(model_save_loc, 'rb'))
        rf_pred = pd.DataFrame({'Sentiment Prediction': rf.predict(x_test)})
        rf_pred.to_csv(output_path)
    
    if model_type == 'XGBoost':
        xgb = pickle.load(open(model_save_loc, 'rb'))
        xgb_pred = pd.DataFrame({'Sentiment Prediction': xgb.predict(x_test)})
        xgb_pred.to_csv(output_path)
    utility.custom_print(str(model_type) + ' has been succesfully predicted\n', logger = logger)


def train_model(model_type, x_train, y_train, x_test, y_test):
    '''
    Trains the chosen model using the respective data given
        Parameters:
            model_type (str): A choice of 3 models are available. 'LogisticRegression', 'RandomForest' and 'XGBoost'
            x_train (pd.DataFrame): Training data containing all variables to be used in training of the model
            y_train (np.array): An array of actual sentiment data for the train set to be validated on during the training of the model
            x_test (pd.DataFrame): Test data containing the same variables as x_train to be predicted on
            y_test (np.array): An array of actual sentiment data for the test set to validate the sentiment predictions and obtain test metrics
    '''
    if model_type == 'LogisticRegression':
        logreg(x_train, y_train, x_test,  y_test)
    
    if model_type == 'RandomForest':
        rf(x_train, y_train, x_test, y_test)
    
    if model_type == 'XGBoost':
        xgboost(x_train, y_train, x_test, y_test)


def logreg(x_train, y_train, x_test, y_test):
    '''
    When 'LogisticRegression' is chosen for parameter model_type in the train_model function, this function will be called to train a Logistic Regression model
        Parameters:
            x_train (pd.DataFrame): Training data containing all variables to be used in training of the model
            y_train (np.array): An array of actual sentiment data for the train set to be validated on during the training of the model
            x_test (pd.DataFrame): Test data containing the same variables as x_train to be predicted on
            y_test (np.array): An array of actual sentiment data for the test set to validate the sentiment predictions and obtain test metrics
    '''
    logreg = LogisticRegression(random_state = 4263, multi_class = 'multinomial', solver = 'saga', max_iter = 2000)
    logreg.fit(x_train, y_train)
    lr_pred = logreg.predict(x_test)
    utility.custom_print('LogisticRegression model succesfully trained\n', logger = logger)
    utility.custom_print('---------------------------------\n',logger = logger)
    utility.churn_eval_metrics(lr_pred, y_test, logger)
    utility.custom_print('\n',logger = logger)
    if save_model:
        pickle.dump(logreg, open(model_save_loc, 'wb')) 
        utility.custom_print('LogisticRegression model succesfully saved', logger = logger)
    else:
        utility.custom_print('Warning: LogisticRegression model has NOT been saved', logger = logger)


def rf(x_train, y_train, x_test, y_test):
    '''
    When 'RandomForest' is chosen for parameter model_type in the train_model function, this function will be called to train a Random Forest model
    via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.

    The available parameters to train are n_estimators and max_depth
        Parameters:
            x_train (pd.DataFrame): Training data containing all variables to be used in training of the model
            y_train (np.array): An array of actual sentiment data for the train set to be validated on during the training of the model
            x_test (pd.DataFrame): Test data containing the same variables as x_train to be predicted on
            y_test (np.array): An array of actual sentiment data for the test set to validate the sentiment predictions and obtain test metrics
    '''
    #Load training parameter range
    n_est_range = config_file['model']['rf_n_est']
    n_est = np.arange(n_est_range[0], n_est_range[1], n_est_range[2])
    max_d_range = config_file['model']['rf_max_d']
    max_d = np.arange(max_d_range[0], max_d_range[1], max_d_range[2])
    rf_grid = {'max_depth':max_d, 'n_estimators':n_est}
    #Execute grid search and fit model
    rf = RandomForestClassifier(random_state = 4263, criterion = 'entropy')
    rf_gscv = GridSearchCV(rf, rf_grid, return_train_score = True)
    rf_gscv.fit(x_train, y_train)
    rf_para = rf_gscv.best_params_
    rf = RandomForestClassifier(n_estimators = rf_para.get('n_estimators'), max_depth = rf_para.get('max_depth'), criterion = 'entropy', random_state = 4263)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    utility.custom_print(rf_gscv.best_params_, logger = logger)    
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    utility.custom_print('RandomForest model succesfully trained\n', logger = logger)
    utility.custom_print('---------------------------------\n',logger = logger)
    utility.churn_eval_metrics(rf_pred, y_test, logger)
    utility.custom_print('\n',logger = logger)
    if save_model:
        pickle.dump(rf, open(model_save_loc, 'wb'))
        utility.custom_print('RandomForest model succesfully saved', logger = logger)
    else:
        utility.custom_print('Warning: RandomForest model has NOT been saved', logger = logger)



def xgboost(x_train, y_train, x_test, y_test):
    '''
    When 'XGBoost' is chosen for parameter model_type in the train_model function, this function will be called to train a XGBoost model
    via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
    The available parameters to train are eta, max_depth, min_child_weight, n_estimators and colsample_bytree

        Parameters:
            x_train (pd.DataFrame): Training data containing all variables to be used in training of the model
            y_train (np.array): An array of actual sentiment data for the train set to be validated on during the training of the model
            x_test (pd.DataFrame): Test data containing the same variables as x_train to be predicted on
            y_test (np.array): An array of actual sentiment data for the test set to validate the sentiment predictions and obtain test metrics
    '''
    #Loading training parameter range
    eta_range = config_file['model']['xgb_eta']
    eta = np.arange(eta_range[0], eta_range[1], eta_range[2])
    max_d_range = config_file['model']['xgb_max_d']
    max_d = np.arange(max_d_range[0], max_d_range[1], max_d_range[2])
    min_weight_range = config_file['model']['xgb_min_weight']
    min_weight = np.arange(min_weight_range[0], min_weight_range[1], min_weight_range[2])
    n_est_range = config_file['model']['xgb_n_est']
    n_est = np.arange(n_est_range[0], n_est_range[1], n_est_range[2])
    sample_range = config_file['model']['xgb_sample']
    sample = np.arange(sample_range[0], sample_range[1], sample_range[2])
    xgb_grid = {'eta':eta, 'max_depth':max_d, 'min_child_weight':min_weight, 'colsample_bytree':sample, 'n_estimators':n_est}
    #Execute grid search and fit model
    xgb = XGBClassifier(random_state = 4263, eval_metric = roc_auc_score)
    xgb_gscv = GridSearchCV(xgb, xgb_grid, return_train_score = True)
    xgb_gscv.fit(x_train, y_train)
    xgb_para = xgb_gscv.best_params_
    xgb = XGBClassifier(eta = xgb_para.get('eta'), max_depth = xgb_para.get('max_depth'), min_child_weight = xgb_para.get('min_child_weight'),
                    colsample_bytree = xgb_para.get('colsample_bytree'), n_estimators = xgb_para.get('n_estimators'), random_state = 4263, eval_metric = roc_auc_score)
    xgb.fit(x_train, y_train)
    xgb_pred = xgb.predict(x_test)
    utility.custom_print('XGBoost model succesfully trained\n', logger = logger)
    utility.custom_print(xgb_para, logger = logger)
    utility.custom_print('---------------------------------\n',logger = logger)
    utility.churn_eval_metrics(xgb_pred, y_test, logger)
    utility.custom_print('\n',logger = logger)
    if save_model:
        pickle.dump(xgb, open(model_save_loc, 'wb'))
        utility.custom_print('XGBoost model succesfully saved', logger = logger)
    else:
        utility.custom_print('Warning: XGBoost model has NOT been saved', logger = logger)


if __name__ == "__main__":
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'non_bert_sentiment_config.yml') #change when done
    config_file = utility.parse_config(config_path)
    log_path = config_file['model']['log_path']
    model_name = config_file['model']['model_name']
    model_type = config_file['model']['model_type']
    model_save_loc = os.path.join(curr_dir, config_file['model']['model_save_loc'])
    data_path = config_file['model']['data_path']
    data_file = config_file['model']['data_file']
    is_train = config_file['model']['is_train']
    save_model = config_file['model']['save_model']
    output_path = os.path.join(curr_dir, config_file['model']['output_path'])
    home_folder = os.path.abspath(os.path.join(curr_dir,'../..')) #change when done
    data_df = pd.read_csv(os.path.join(home_folder, data_path, data_file))
    logger = open(os.path.join(curr_dir, log_path), 'w')
    df = pre.Dataset(data_df)
    df.create_tfidf(root_words_option = 2, remove_stop_words = True, lower_case = True, ngrams = (1,2), min_doc = 0.05, max_doc = 0.95)
    tfidf = pd.DataFrame(df.tfidf[1].toarray())
    tfidf['Time'] = df.date
    tfidf['Sentiment'] = df.sentiments
    tfidf = tfidf.replace({'positive': 1, 'negative':0})
    if is_train:
        train, test = ttsplit(tfidf)
        x_train = train.drop(['Time', 'Sentiment'], axis = 1)
        y_train = train['Sentiment'].to_numpy()
        x_test = test.drop(['Time', 'Sentiment'], axis = 1)
        y_test = test['Sentiment'].to_numpy()
        oversample = SMOTE(random_state = 4263)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
        utility.custom_print("Training dataset has been loaded successfully\n",logger = logger)
        utility.custom_print('---------------------------------',logger = logger)  
        train_model(model_type, x_train, y_train, x_test, y_test)
        utility.custom_print('\n---------------------------------\n',logger = logger)
    else:
        utility.custom_print("Data to be predicted has been loaded successfully",logger = logger)
        predict_data(model_type, tfidf)

