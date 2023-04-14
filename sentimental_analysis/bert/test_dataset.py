import pytest
import os
import pandas as pd
import syspend
from sentimental_analysis.bert.dataset import *
from utility import parse_config

@pytest.fixture
def example_data():
    data  =  pd.read_csv(os.path.join(os.getcwd(), '../..', 'data', 'reviews.csv'))
    return data

@pytest.fixture
def parameters():
    config_path = os.path.join(os.getcwd(), 'bert_sentiment_config.yml')
    config_file = parse_config(config_path)
    model_name = config_file['model']['model_name']
    max_len = int(config_file['model']['max_len'])
    batch_size = int(config_file['model']['batch_size'])
    return [model_name, batch_size,max_len]

@pytest.fixture
def single_sentence():
    """
    Fixture for test_data_loader. Gives one sentences to be used in the data_loader function
    Example usage:
    ```
    def test_data_loader(single_sentence):
        # test code that uses the `single_sentence` fixture
    ```
    """
    return [["bert-base-uncased", 4, 1] ,  ["I like this product"]]

@pytest.fixture
def mutiple_sentence():
    """
    Fixture for test_data_loader. Gives a series of test
    sentences to be used in the data_loader function
    Example usage:
    ```
    def test_data_loader(single_sentence):
        # test code that uses the `single_sentence` fixture
    ```
    """
    return [["bert-base-uncased", 6, 2], ["I like this product", "I hate this product really much"]]

def test_data_loader_single_sentence(single_sentence):
    """
    Testing data_loader function to ensure it returns the correct length for one sentence
    and does not exceed maximum length
    """
    parameters , sentences = single_sentence
    model_name, max_len, num_sentences = parameters
    assert len(sentences) == num_sentences
    for sentence in sentences:
        assert isinstance(sentence, str)
        assert isinstance(max_len, int)
        assert isinstance(model_name, str)
        assert len(model_name) > 0
        assert len(sentence) > 0
        assert len(sentence) <= 512

def test_data_loader_mutiple_sentence(mutiple_sentence):
    """Testing data_loader function to ensure it returns the correct length for mutiple sentences
    and does not exceed maximum length
    """
    parameters , sentences = mutiple_sentence
    model_name, max_len, num_sentences = parameters
    assert len(sentences) == num_sentences
    for sentence in sentences:
        assert isinstance(sentence, str)
        assert isinstance(max_len, int)
        assert isinstance(model_name, str)
        assert len(model_name) > 0
        assert len(sentence) > 0
        assert len(sentence) <= 512

def test_create_data_loader(example_data,parameters):
    """Testing if "Sentiment column in pandas dataframe and only contains string
    "positive" and "negative"
    """
    assert 'Sentiment' in example_data.columns
    sentiment_unique = list(set(list(example_data['Sentiment'])))
    assert all(elem in {'positive', 'negative'} for elem in sentiment_unique)
    tokenizer_name, batch_size,max_len = parameters
    create_data_loader(tokenizer_name, batch_size,max_len,example_data)

def test_full_data_loader(example_data,parameters):
    """Testing if "Sentiment column in pandas dataframe and only contains string
    "positive" and "negative"
    """
    assert 'Sentiment' in example_data.columns
    sentiment_unique = list(set(list(example_data['Sentiment'])))
    assert all(elem in {'positive', 'negative'} for elem in sentiment_unique)
    tokenizer_name, batch_size,max_len = parameters
    full_create_data_loader(tokenizer_name, batch_size,max_len,example_data)
