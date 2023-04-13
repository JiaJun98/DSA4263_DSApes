"""Ensure non bert topic modelling functions are working"""
import os
import sys
sys.path.extend([".", "../.."])
from non_bert_topic_model import TopicModel
from preprocess_class import Dataset
import pandas as pd
from io import StringIO
import pytest
from sklearn.decomposition import NMF
pytestmark = pytest.mark.filterwarnings("ignore")

curr_dir = os.getcwd()
pytest_dir = os.path.join(curr_dir, "pytest_TopicModel")
if not os.path.exists(pytest_dir):
    os.makedirs(pytest_dir)

@pytest.fixture
def train_dataset():
    """Generate train dataset"""
    train_data = pd.DataFrame({'Time': ['1/1/2021', '2/1/2021',
                                        '21/7/2021', '31/8/2022', '1/7/2022'],
                       'Sentiment': ['Positive', 'Positive', 
                                     'Positive', 'Negative', 'Negative'],
                       'Text': ["had this cup of tea. nice tea.",
                                "i love to have this coffee. this is my favourite coffee",
                                "this coffee is irreplaceable. where else to find this good coffee",
                                "this tea is too bitter. awful tea",
                                "this coffee is too sweet. prefer olden standards coffee"]})

    return Dataset(train_data)

@pytest.fixture
def test_dataset():
    """Generate test dataset"""
    test_data = pd.DataFrame({'Time': ['9/4/2022', '4/5/2021', '13/5/2021',
                                       '18/3/2022', '23/4/2021'],
                       'Sentiment': ['Positive', 'Positive', 'Positive', 'Negative', 'Negative'],
                       'Text': ["I fell in love with this Keurig coffee. Amazing standards!", 
                                "The milk complements the coffee well",
                                "Tea tarik is the best in Singapore", 
                                "Where is the honey in the tea? So bland.",
                                "Never ever buy coffee from Brazil. I dun like that after taste"]})
    return Dataset(test_data)

def test_set_topic_labels(train_dataset, monkeypatch):
    """
    Ensures that topic_label attribute is updated upon running this method.
    """
    test_model = TopicModel(train_dataset = train_dataset, custom_print_in = False)
    sample_input = StringIO('testTopic')
    monkeypatch.setattr('sys.stdin', sample_input)
    test_model.set_topic_labels(0)
    assert test_model.topic_label == ['testTopic']

def test_modify_dataset_stop_words_list(train_dataset, test_dataset):
    """
    Ensures that the stop words list is replaced in both train and test dataset.
    """
    test_model = TopicModel(train_dataset = train_dataset, test_dataset = test_dataset,
                            custom_print_in = False)
    expected = ['i', 'am', 'happy']
    test_model.modify_dataset_stop_words_list(replace_stop_words_list = expected)

    assert test_model.train_dataset.stop_words_list == expected
    assert test_model.test_dataset.stop_words_list == expected

def test_preprocess_dataset(test_dataset):
    """
    Ensures that text of the dataset in the model class is preprocessed.
    """
    test_model = TopicModel(test_dataset = test_dataset, custom_print_in = False)
    test_model.preprocess_dataset(remove_stop_words = False)
    expected_output_0 = ["i", "fell", "in", "love", "with", "this", "keurig", "coffee",
                         "amazing", "standards"]
    assert test_model.test_dataset.preprocessed_text[0] == expected_output_0
    assert len(test_model.test_dataset.preprocessed_text) == 5

def test_generate_feature_engineer(train_dataset):
    """
    Ensures that the feature engineer of the dataset in the model
    """
    test_model = TopicModel(train_dataset = train_dataset, custom_print_in = False)
    test_model.preprocess_dataset(remove_stop_words = False)
    test_model.generate_feature_engineer()

    assert test_model.train_dataset.feature_engineer is not None

def test_train(train_dataset):
    """
    Ensures the model is trained on the dataset, output 2 files with the
    second file as text labelled to each topic number
    """
    test_model = TopicModel(train_dataset = train_dataset, feature_engineer_type = "tfidf",
                            custom_print_in = False)
    test_model.preprocess_dataset(remove_stop_words = False)
    test_model.generate_feature_engineer()
    output = test_model.train("NMF", 2, pytest_dir)

    assert len(output) == 2
    assert output[1].shape[0] == 5
    assert output[1].shape[1] == 3

def test_display_topics(train_dataset):
    """
    Ensures that the right number of topics and its corresponding topic key words are generated.
    """
    test_model = TopicModel(train_dataset = train_dataset, custom_print_in = False)
    test_model.preprocess_dataset(remove_stop_words = False)
    test_model.generate_feature_engineer()
    component, labelled_no_topics = test_model.train("NMF", 2, pytest_dir)
    topic_key_words = test_model.display_topics(training_model = "NMF",
                                                trained_topics = component, num_top_words = 2,
                                                train_output_path = pytest_dir)
    assert len(topic_key_words) == 2
    assert len(topic_key_words[0]) == 2

def test_predict(test_dataset):
    """
    Ensures that the correct text is being predicted on, correct number of columns
    and column names are outputted 
    and the text are labelled based on the input topic labels
    """
    pickled_vectorizer = os.path.join(pytest_dir, "training_tfidf_vectorizer_2.pk")
    pickled_model = os.path.join(pytest_dir, "training_NMF_model_2.pk")
    topic_label = os.path.join(pytest_dir, "topic_key_words.csv")
    test_model = TopicModel(test_dataset = test_dataset, pickled_vectorizer = pickled_vectorizer,
                            pickled_model = pickled_model, topic_label = topic_label,
                            custom_print_in = False, feature_engineer = "tfidf")
    test_model.preprocess_dataset(remove_stop_words = False, word_form = ['noun'])
    labelled_test = test_model.predict(test_output_path = pytest_dir, root_word_option = 0,
                                       remove_stop_words = False)
    output_topic_label = labelled_test['Topic_label'].unique().tolist()
    output_topic_label.sort()
    expected_topic_label = pd.read_csv(topic_label)['Topic_label'].tolist()
    expected_topic_label.sort()
    assert labelled_test['Text'].tolist() == test_dataset.text.tolist()
    assert list(labelled_test.columns) == ["Text", "Topic_label"]
    assert output_topic_label == expected_topic_label

def test_churn_eval_metrics(test_dataset, monkeypatch):
    """
    Ensures that the number of topics processed and the number of samples
    from each topic is included in the output file is correct.
    """
    pickled_vectorizer = os.path.join(pytest_dir, "training_tfidf_vectorizer_2.pk")
    pickled_model = os.path.join(pytest_dir, "training_NMF_model_2.pk")
    topic_label = os.path.join(pytest_dir, "topic_key_words.csv")
    test_model = TopicModel(test_dataset = test_dataset, pickled_vectorizer = pickled_vectorizer,
                            pickled_model = pickled_model,
                           topic_label = topic_label, custom_print_in = False,
                           feature_engineer = "tfidf")
    test_model.preprocess_dataset(remove_stop_words = False, word_form = ['noun'])
    labelled_test = test_model.predict(test_output_path = pytest_dir, root_word_option = 0,
                                       remove_stop_words = False)
    inputs = iter(["1", "0"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    test_model.churn_eval_metrics(labelled_test, 1, pytest_dir)

    churned_output = pd.read_csv(os.path.join(pytest_dir, "test_sample_labels.csv"))
    assert churned_output.shape[0] == 2
    assert churned_output['Topic_label'].value_counts().tolist()[0] == 1
    assert churned_output['Topic_label'].value_counts().tolist()[1] == 1
