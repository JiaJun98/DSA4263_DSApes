import syspend
import preprocess_class as pre
import pandas as pd
import numpy as np
import utility
import os
import pickle
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from model_base_class import BaseModel


#Create Non-Bert Classifier Class
class NonBertClassifier(BaseModel):
    '''
    Non-Bert Model for Sentiment Analysis
    Parameters
    -----------
    model_name: str
        Non-Bert Model to train/predict. Choose from 'LogisticRegression', 'RandomForest', 'XGBoost'

    Attributes
    -----------
    model: str
        The type of model that will be used in training/prediction of the data ('LogisticRegression', 'RandomForest', 'XGBoost')
    data: pd.DataFrame
        A data frame with columns 'Time', 'Text' and 'Sentiment' ('Sentiment' only for train)
    x_train: pd.DataFrame
        Contains preprocessed bag of words features that will be used in training
    x_test: pd.DataFrame
        Contains preprocessed bag of words features that will be used in testing trained model
    y_train: np.ndarray
        True value of 'Sentiment' for each 'Text' to be used in training
    y_test: np.ndarray
        True value of 'Sentiment' for each 'Text' to be used to obtain evaluation metrics
    time: np.ndarray, pd.DataFrame
        Time at which Text was being produced
    text: np.ndarray, pd.DataFrame
        Text containing reviews to be preprocessed
    
    Methods
    -----------
    ttsplit():
        Returns a 80-20 train and test set for the given dataset. 
    predict(model_type, threshold):
        Predicts the given input data using the specified model_type and outputs the predictions (containing 0's or 1's) into a csv file.
    train(model_type):
        Trains the chosen model using the respective data given
    logreg():
        When 'LogisticRegression' is chosen for parameter model_type in the train function, this function will be called to train a Logistic Regression model
    rf():
        When 'RandomForest' is chosen for parameter model_type in the train function, this function will be called to train a Random Forest model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are n_estimators and max_depth
    xgboost():
        When 'XGBoost' is chosen for parameter model_type in the train function, this function will be called to train a XGBoost model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are eta, max_depth, min_child_weight, n_estimators and colsample_bytree
    '''
    def  __init__(self, data = None, model_name = None):
        self.model = model_name
        self.data = data
        self.time = None
        self.text = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None


    def ttsplit(self):
        '''
        Returns a 80-20 train and test set for the given dataset. 
        '''
        df = pre.Dataset(self.data)
        df.preprocessing_text(root_word_option = 2, remove_stop_words = True, lower_case = True, word_form = ['adverb', 'adjective'])
        df.create_bow(ngrams = (1,2), min_doc = 0.05, max_doc = 0.5)
        bow = pd.DataFrame(df.feature_engineer[2].toarray())
        bow['Sentiment'] = self.data['Sentiment']
        bow = bow.replace({'positive': 1, 'negative':0})
        train, test = train_test_split(bow, test_size = 0.2, random_state = 4263, stratify = bow.Sentiment)
        self.x_train = train.drop(['Sentiment'], axis = 1)
        self.y_train = train['Sentiment'].to_numpy()
        self.x_test = test.drop(['Sentiment'], axis = 1)
        self.y_test = test['Sentiment'].to_numpy()
        oversample = SMOTE()
        self.x_train, self.y_train = oversample.fit_resample(self.x_train, self.y_train)
        svd = TruncatedSVD(n_components=len(self.x_train.axes[1]), random_state=4263)
        svd.fit(self.x_train)
        exp = 0
        n = 0
        for i in svd.explained_variance_ratio_:
            if exp < 0.9:
                exp += i
                n += 1
            else:
                break
        print([n,exp])
        utility.custom_print('N Components for SVD: ' + str(n) + '\n', logger = logger)
        self.x_train = pd.DataFrame(TruncatedSVD(n_components = n, random_state = 4263).fit_transform(self.x_train))
        self.x_test = pd.DataFrame(TruncatedSVD(n_components = n, random_state = 4263).fit_transform(self.x_test))


    def predict(self, model_type, threshold):
        '''
        Predicts the given input data using the specified model_type and outputs the predictions (containing 0's or 1's) into a csv file.
        
        Parameters
        -----------
            model_type: str
                A choice of 3 models are available. 'LogisticRegression', 'RandomForest' and 'XGBoost'
            threshold: float
                Threshold to decide at what probability the sentiment would be considered positive
        '''
        self.data['Sentiment'] = 0
        self.time = self.data['Time']
        self.text = self.data['Text']
        df = pre.Dataset(self.data)
        df.preprocessing_text(root_word_option = 2, remove_stop_words = True, lower_case = True, word_form = ['adverb', 'adjective'])
        df.create_bow(ngrams = (1,2), min_doc = 0.05, max_doc = 0.5)
        self.data = pd.DataFrame(df.feature_engineer[2].toarray())
        self.data = pd.DataFrame(TruncatedSVD(n_components=n_svd).fit_transform(self.data))
        if model_type == 'LogisticRegression':
            logreg = pickle.load(open(model_save_loc, 'rb'))
            sentiment_proba = logreg.predict_proba(self.data)[:,1]
            sentiment = []
            for i in sentiment_proba:
                if i >= threshold:
                    sentiment.append(1)
                else:
                    sentiment.append(0)
            lr_pred = pd.DataFrame({'Text': self.text, 'Time': self.time, 'predicted_sentiment': sentiment, 'predicted_sentiment_probability': sentiment_proba})
            lr_pred.to_csv(output_path, index = False)
        
        if model_type == 'RandomForest':
            rf = pickle.load(open(model_save_loc, 'rb'))
            sentiment_proba = rf.predict_proba(self.data)[:,1]
            sentiment = []
            for i in sentiment_proba:
                if i >= threshold:
                    sentiment.append(1)
                else:
                    sentiment.append(0)
            rf_pred = pd.DataFrame({'Text': self.text, 'Time': self.time, 'predicted_sentiment': sentiment, 'predicted_sentiment_probability': sentiment_proba})
            rf_pred.to_csv(output_path, index = False)
        
        if model_type == 'XGBoost':
            xgb = pickle.load(open(model_save_loc, 'rb'))
            sentiment_proba = xgb.predict_proba(self.data)[:,1]
            sentiment = []
            for i in sentiment_proba:
                if i >= threshold:
                    sentiment.append(1)
                else:
                    sentiment.append(0)
            xgb_pred = pd.DataFrame({'Text': self.text, 'Time': self.time, 'predicted_sentiment': sentiment, 'predicted_sentiment_probability': sentiment_proba})
            xgb_pred.to_csv(output_path, index = False)
        utility.custom_print(str(model_type) + ' has been succesfully predicted\n', logger = logger)


    def train(self, model_type):
        '''
        Trains the chosen model using the respective data given
        
        Parameters
        -----------
            model_type : str
                A choice of 3 models are available. 'LogisticRegression', 'RandomForest' and 'XGBoost'
        '''
        if model_type == 'LogisticRegression':
            self.logreg()
        
        if model_type == 'RandomForest':
            self.rf()
        
        if model_type == 'XGBoost':
            self.xgboost()


    def logreg(self):
        '''
        When 'LogisticRegression' is chosen for parameter model_type in the train function, this function will be called to train a Logistic Regression model
        '''
        logreg = LogisticRegression(random_state = 4263, multi_class = 'multinomial', solver = 'saga', max_iter = 2000)
        logreg.fit(self.x_train, self.y_train)
        lr_pred = logreg.predict(self.x_test)
        lr_proba = logreg.predict_proba(self.x_test)[:,1]
        utility.custom_print('LogisticRegression model succesfully trained\n', logger = logger)
        utility.custom_print('---------------------------------\n',logger = logger)
        utility.churn_eval_metrics(lr_pred, self.y_test, logger)
        utility.custom_print('\n---------------------------------\n',logger = logger)
        utility.custom_print('Threshold parameter tuning\n', logger = logger)
        threshold, accuracy = utility.plot_pr_curve(lr_proba, self.y_test, plot_path)
        lr_pred_best = []
        for i in lr_proba:
            if i>=threshold:
                lr_pred_best.append(1)
            else:
                lr_pred_best.append(0)
        utility.custom_print('Prediction using best threshold for accuracy\n-------------------------\n', logger = logger)                
        utility.churn_eval_metrics(lr_pred_best, self.y_test, logger)
        utility.custom_print('Best threshold for accuracy: ' + str(threshold), logger = logger)
        utility.custom_print('Accuracy score at best threshold: ' + str(accuracy), logger = logger)
        if save_model:
            pickle.dump(logreg, open(model_save_loc, 'wb')) 
            utility.custom_print('LogisticRegression model succesfully saved', logger = logger)
        else:
            utility.custom_print('Warning: LogisticRegression model has NOT been saved', logger = logger)


    def rf(self):
        '''
        When 'RandomForest' is chosen for parameter model_type in the train function, this function will be called to train a Random Forest model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are n_estimators and max_depth
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
        rf_gscv.fit(self.x_train, self.y_train)
        rf_para = rf_gscv.best_params_
        rf = RandomForestClassifier(n_estimators = rf_para.get('n_estimators'), max_depth = rf_para.get('max_depth'), criterion = 'entropy', random_state = 4263)
        rf.fit(self.x_train, self.y_train)
        rf_pred = rf.predict(self.x_test)
        rf_proba = rf.predict_proba(self.x_test)[:,1]
        utility.custom_print(rf_gscv.best_params_, logger = logger)
        utility.custom_print('RandomForest model succesfully trained\n', logger = logger)
        utility.custom_print('---------------------------------\n',logger = logger)
        utility.churn_eval_metrics(rf_pred, self.y_test, logger)
        utility.custom_print('\n---------------------------------\n',logger = logger)
        utility.custom_print('Threshold parameter tuning\n', logger = logger)
        threshold, accuracy = utility.plot_pr_curve(rf_proba, self.y_test, plot_path)
        rf_pred_best = []
        for i in rf_proba:
            if i>=threshold:
                rf_pred_best.append(1)
            else:
                rf_pred_best.append(0)
        utility.custom_print('Prediction using best threshold for accuracy\n-------------------------\n', logger = logger)                
        utility.churn_eval_metrics(rf_pred_best, self.y_test, logger)
        utility.custom_print('Best threshold for accuracy: ' + str(threshold), logger = logger)
        utility.custom_print('Accuracy score at best threshold: ' + str(accuracy), logger = logger)
        if save_model:
            pickle.dump(rf, open(model_save_loc, 'wb'))
            utility.custom_print('RandomForest model succesfully saved', logger = logger)
        else:
            utility.custom_print('Warning: RandomForest model has NOT been saved', logger = logger)



    def xgboost(self):
        '''
        When 'XGBoost' is chosen for parameter model_type in the train function, this function will be called to train a XGBoost model
        via a grid search for the range of grid specified in the non_bert_sentiment_config.yml file.
        The available parameters to train are eta, max_depth, min_child_weight, n_estimators and colsample_bytree
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
        xgb_gscv.fit(self.x_train, self.y_train)
        xgb_para = xgb_gscv.best_params_
        xgb = XGBClassifier(eta = xgb_para.get('eta'), max_depth = xgb_para.get('max_depth'), min_child_weight = xgb_para.get('min_child_weight'),
                        colsample_bytree = xgb_para.get('colsample_bytree'), n_estimators = xgb_para.get('n_estimators'), random_state = 4263, eval_metric = roc_auc_score)
        xgb.fit(self.x_train, self.y_train)
        xgb_pred = xgb.predict(self.x_test)
        xgb_proba = xgb.predict_proba(self.x_test)[:,1]
        utility.custom_print('XGBoost model succesfully trained\n', logger = logger)
        utility.custom_print(xgb_gscv.best_params_, logger = logger)
        utility.custom_print('\n---------------------------------\n',logger = logger)
        utility.churn_eval_metrics(xgb_pred, self.y_test, logger)
        utility.custom_print('\n---------------------------------\n',logger = logger)
        utility.custom_print('Threshold parameter tuning\n', logger = logger)
        threshold, accuracy = utility.plot_pr_curve(xgb_proba, self.y_test, plot_path)
        xgb_pred_best = []
        for i in xgb_proba:
            if i>=threshold:
                xgb_pred_best.append(1)
            else:
                xgb_pred_best.append(0)
        utility.custom_print('Prediction using best threshold for accuracy\n-------------------------\n', logger = logger)                
        utility.churn_eval_metrics(xgb_pred_best, self.y_test, logger)
        utility.custom_print('Best threshold for accuracy: ' + str(threshold), logger = logger)
        utility.custom_print('Accuracy score at best threshold: ' + str(accuracy), logger = logger)
        if save_model:
            pickle.dump(xgb, open(model_save_loc, 'wb'))
            utility.custom_print('XGBoost model succesfully saved', logger = logger)
        else:
            utility.custom_print('Warning: XGBoost model has NOT been saved', logger = logger)


if __name__ == "__main__":
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    curr_dir = os.getcwd()
    home_folder = os.path.abspath(os.path.join(curr_dir,'../..'))
    config_path = os.path.join(curr_dir, 'non_bert_sentiment_config.yml')
    config_file = utility.parse_config(config_path)
    log_path = config_file['model']['log_path']
    model_name = config_file['model']['model_name']
    model_type = config_file['model']['model_type']
    model_save_loc = os.path.join(curr_dir, config_file['model']['model_save_loc'])
    data_path = config_file['model']['data_path']
    data_file = config_file['model']['data_file']
    threshold = config_file['model']['threshold']
    n_svd = config_file['model']['n_svd']
    is_train = config_file['model']['is_train']
    save_model = config_file['model']['save_model']
    plot_path = config_file['model']['plot_path']
    output_path = os.path.join(home_folder, config_file['model']['output_path'])
    plot_path = os.path.join(curr_dir, plot_path)
    logger = open(os.path.join(curr_dir, log_path), 'w')

    data_df = pd.read_csv(os.path.join(home_folder, data_path, data_file))
    df = NonBertClassifier(data = data_df, model_name = model_name)
    if is_train:
        df.ttsplit()
        utility.custom_print("Training dataset has been loaded successfully\n",logger = logger)
        utility.custom_print('---------------------------------',logger = logger)
        df.train(model_type)
        utility.custom_print('\n---------------------------------\n',logger = logger)
    else:
        utility.custom_print("Data to be predicted has been loaded successfully",logger = logger)
        df.predict(model_type, threshold)