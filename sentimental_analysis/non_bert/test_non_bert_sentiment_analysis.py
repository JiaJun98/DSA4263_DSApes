import pytest
from train import NonBertClassifier
import pandas as pd
import numpy as np
import syspend
import os

data  =  pd.read_csv(os.path.join(os.getcwd(), '../..', 'data', 'reviews.csv'))
print(data)
@pytest.fixture
def example_data():
    data  =  pd.read_csv(os.path.join(os.getcwd(), '../..', 'data', 'reviews.csv'))
    return data

def test_init(example_data):
    reviews = NonBertClassifier(example_data)
    assert isinstance(reviews.data, pd.DataFrame)
    for i in reviews.data['Time']:
        assert isinstance(i, str)
    for i in reviews.data['Sentiment']:
        assert isinstance(i, str)
    for i in reviews.data['Text']:
        assert isinstance(i, str)

@pytest.fixture
def class_data(example_data):
    return NonBertClassifier(example_data)

def test_ttsplit(class_data):
    reviews = class_data
    reviews.ttsplit()
    assert isinstance(reviews.x_train, pd.DataFrame)
    assert isinstance(reviews.y_train, np.ndarray)
    assert isinstance(reviews.x_test, pd.DataFrame)
    assert isinstance(reviews.y_test, np.ndarray)


def test_predict():
    data = pd.read_csv(os.path.join(os.getcwd(), '../..', 'data', 'non_bert_sa_prediction.csv'))
    for i in data['Time']:
        assert isinstance(i, str)
    for i in data['predicted_sentiment']:
        assert isinstance(i, int)
    for i in data['Text']:
        assert isinstance(i, str)
    for i in data['predicted_sentiment_probability']:
        assert isinstance(i, float)
    
pytest.main()