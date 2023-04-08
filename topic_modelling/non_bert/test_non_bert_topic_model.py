import pytest
from sklearn.decomposition import NMF
import sys
from io import StringIO
import os
sys.path.extend([".", "../.."])

from non_bert_topic_model import TopicModel
from preprocess_class import *

import warnings
warnings.filterwarnings("ignore")

curr_dir = os.getcwd()
pytest_dir = os.path.join(curr_dir, "pytest_TopicModel")
if not os.path.exists(pytest_dir):
    os.makedirs(pytest_dir)

@pytest.fixture
def train_dataset():
    df = pd.DataFrame({'Time': ['1/1/2021', '2/1/2021', '21/7/2021', '31/8/2022', '1/7/2022'], 
                       'Sentiment': ['Positive', 'Positive', 'Positive', 'Negative', 'Negative'],
                       'Text': ["had this cup of tea. nice tea.", "i love to have this coffee. this is my favourite coffee",
                                "this coffee is irreplaceable. where else to find this good coffee", "this tea is too bitter. awful tea",
                                "this coffee far too sweet. can't get olden standards coffee anymore"]})

    return Dataset(df)

@pytest.fixture
def test_dataset():
    df = pd.DataFrame({'Time': ['9/4/2022', '4/5/2021', '13/5/2021', '18/3/2022', '23/4/2021'],
                       'Sentiment': ['Positive', 'Positive', 'Positive', 'Negative', 'Negative'],
                       'Text': ["I fell in love with this Keurig coffee. Amazing!", "The milk complements the coffee well",
                                "Tea tarik is the best in Singapore", "Where is the honey in the tea? So bland.",
                                "Never ever buy coffee from Brazil. I dun like that after taste"]})
    return Dataset(df)

def test_set_topic_labels(train_dataset, monkeypatch):
    testModel = TopicModel(train_dataset = train_dataset, custom_print = False)
    sample_input = StringIO('testTopic')
    monkeypatch.setattr('sys.stdin', sample_input)
    testModel.set_topic_labels(0)
    assert testModel.topic_label == ['testTopic']

def test_get_input_text(train_dataset):
    testModel = TopicModel(train_dataset = train_dataset, custom_print = False)
    train_dataset.word_tokenizer(lower_case = False)
    expected_output = train_dataset.tokenized_words
    assert testModel.get_input_text(testModel.train_dataset, 0, remove_stop_words = False).tolist() == expected_output.tolist()

def test_preprocess_dataset(train_dataset):
    testModel = TopicModel(train_dataset = train_dataset, custom_print = False)
    testModel.preprocess_dataset(remove_stop_words = False, include_words = [], exclude_words = [])
    assert testModel.train_dataset.bow is not None

def test_display_topics(train_dataset):
    testModel = TopicModel(train_dataset = train_dataset, feature_engineer = "tfidf", custom_print = False)
    testModel.preprocess_dataset(remove_stop_words = False, include_words = [], exclude_words = [])
    training = NMF(n_components = 2, init = 'nndsvd', random_state = 4263, solver = 'cd')
    trained_model = training.fit_transform(testModel.train_dataset.tfidf[2])
    dir = os.path.join(os.getcwd(), "pytest_TopicModel")
    topic_key_words = testModel.display_topics(training.components_, testModel.train_dataset.tfidf[0].get_feature_names_out(), 
                                               num_top_words = 2, train_output_path = dir, training_model = "NMF")
    assert len(topic_key_words) == 2
    assert len(topic_key_words[0]) == 2

def test_train(train_dataset, monkeypatch):
    testModel = TopicModel(train_dataset = train_dataset, custom_print = False)
    testModel.preprocess_dataset(remove_stop_words = False, word_form = ['noun'])
    inputs = iter(['Virtual topic 1', 'Virtual topic 2'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    testModel.train("NMF", num_of_topics = 2, num_top_words = 1, num_top_documents = 1, train_output_path = pytest_dir)
    assert testModel.topic_label == ['Virtual topic 1', 'Virtual topic 2']

def test_predict(test_dataset):
    pickled_vectorizer = os.path.join(pytest_dir, "training_vectorizer_2.pk")
    pickled_model = os.path.join(pytest_dir, "training_model_2.pk")
    topic_label = os.path.join(pytest_dir, "topic_key_words.csv")
    testModel = TopicModel(test_dataset = test_dataset, pickled_vectorizer = pickled_vectorizer, pickled_model = pickled_model,
                           topic_label = topic_label)
    testModel.preprocess_dataset(remove_stop_words = False, word_form = ['noun'])
    #tbc





