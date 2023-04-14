import os
import numpy as np
import pytest
import syspend
import torch
import pandas as pd
from utility import parse_config
from sentimental_analysis.bert.train import BertClassifier
from sentimental_analysis.bert.dataset import data_loader
from transformers import AutoModelForSequenceClassification

pytestmark = pytest.mark.filterwarnings("ignore")

@pytest.fixture
def example_data():
    """Fixture for the training dataframe reviews.csv
    """
    config_path = os.path.join(os.getcwd(), 'bert_sentiment_config.yml')
    config_file = parse_config(config_path)
    train_file = config_file['model']['data_folder']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    data_df = pd.read_csv(os.path.join(home_folder,train_file))
    return data_df

@pytest.fixture
def example_config():
    """Fixture for the global variables by reading yml file
    """
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'bert_sentiment_config.yml')
    config_file = parse_config(config_path)
    model_name = config_file['model']['model_name']
    n_classes = int(config_file['model']['n_classes'])
    max_len = int(config_file['model']['max_len'])
    batch_size = int(config_file['model']['batch_size'])
    epochs = int(config_file['model']['epochs'])
    learning_rate = float(config_file['model']['learning_rate'])
    epsilon = float(config_file['model']['epsilon'])
    train_val_percentage = float(config_file['model']['train_val_percentage'])
    train_on_full_data = eval(str(config_file['model']['train_on_full_data']))
    no_of_kfolds = config_file['model']['noOfKFolds']
    model_path = os.path.join(curr_dir, config_file['model']['model_path'])
    logging_path = os.path.join(curr_dir,config_file['model']['log_path'])
    plot_path =  os.path.join(curr_dir,config_file['model']['plot_path'])
    return [config_path,model_name,n_classes,max_len,batch_size,epochs,
    learning_rate,epsilon,train_val_percentage,train_on_full_data,no_of_kfolds,
    model_path,logging_path,plot_path]

def test_init(example_data,example_config):
    """Testing initialisation of BERTClassifier to ensure that 
        only the columns Sentiment Text Time are inside and ensuriing number of classes is 2
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = example_config[1]
    n_classes = example_config[2]
    assert all(ele in list(example_data.columns) for ele in ['Sentiment', 'Text', 'Time'])
    assert n_classes == 2
    sentimental_classifier = BertClassifier(model_name, n_classes)
    sentimental_classifier.model.to(device)

def test_predict(example_data,example_config):
    """Testing predict function to test same length of class prediction and probabilities list
        Test if algo of probability > Threshold implies said prediction
        For simplicity we will be only using 20 rows of the original reviews
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shortened_df = example_data.head(20)
    texts =  shortened_df['Text'].tolist()
    x_preprocessed = np.array([text for text in texts])
    model_name = example_config[1]
    n_classes = example_config[2]
    max_len = example_config[3]
    model = BertClassifier(model_name, n_classes)
    model.model.to(device)
    test_dataloader = data_loader(model_name, max_len, x_preprocessed, len(texts))
    preds, probs = model.predict(test_dataloader, 0.5) #For testing purposes
    assert len(preds) == len(probs)
    for index, item in enumerate(preds):
        if probs[index] < 0.5:
            assert item == 'Negative'
        else:
            assert item == 'Positive'
