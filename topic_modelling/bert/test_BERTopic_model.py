import pytest
from BERTopic_model import BERTopic_model
from preprocess_class import Dataset, create_datasets
import pandas as pd
import syspend
import sys
import os
sys.path.append("../..")
pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def train_dataset():
    df = pd.read_csv('../../data/reviews.csv')
    train, _ = create_datasets(df)
    return train


@pytest.fixture
def test_dataset():
    df = pd.DataFrame({'Time': ['9/4/2022', '4/5/2021', '13/5/2021', '18/3/2022', '23/4/2021'],
                       'Sentiment': ['Positive', 'Positive', 'Positive', 'Negative', 'Negative'],
                       'Text': ["I fell in love with this Keurig coffee. Amazing standards!",
                                "The milk complements the coffee well",
                                "Tea tarik is the best in Singapore",
                                "Where is the honey in the tea? So bland.",
                                "Never ever buy coffee from Brazil. I dun like that after taste"]})
    return Dataset(df)

@pytest.fixture
def model(train_dataset):
    model = BERTopic_model()
    model.train(train_dataset.data['Text'])
    return model

def test_predict(model, test_dataset):
    topics = model.predict(test_dataset.data['Text'])
    assert len(topics[0]) == len(test_dataset.data['Text'])

def test_evaluate(model, train_dataset):
    score = model.evaluate(train_dataset.data['Text'])
    assert (score > -1) and (score < 1)
