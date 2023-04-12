"""Test Dataset class from preprocess_class.py"""
import sys
import pytest
import numpy as np
import pandas as pd
from preprocess_class import Dataset, create_datasets
sys.path.append(".")

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def example_data():
    """Sample data to test Dataset class"""
    data_dict = {'Sentiment': ['Positive', 'Positive', 'Positive', 'Positive',
                               'Positive', 'Positive', 'Positive', 'Positive',
                                'Positive', 'Positive', 'Negative', 'Negative',
                                'Negative', 'Negative', 'Negative'],
                'Time': ['1/1/2021', '2/1/2021', '21/7/2021', '31/8/2022',
                        '1/7/2022', '8/9/2021', '9/9/2029', '9/10/2012',
                        '10/8/2023', '9/4/2022', '4/5/2021', '13/5/2021',
                        '18/3/2022', '23/4/2021', '3/3/2021'],
                'Text': ["I like this drink", "The taste is just nice for me",
                        "<br />This ice cream is becoming my all-time favourite",
                        "This drink is not too  sweet. Fantastic ! ",
                        "Delicious fish with the ultimate flavor, best ever!",
                        "I love this chips! Great healthy snacks!</ br ahref:  https://iron.com>",
                        "No salt, no sugar. Yet this chips is still so flavourful!",
                        "I hate mash potatoes. But I still rate this 5 / 5. Taste perfect!",
                        "No better food than this mash potatoes!",
                        "Perfect 10. Marvellous potatoes!",
                        "THIS IS THE WORST POTATOES I HAVE EVER EATEN",
                        "<a href::item.com> This ice cream is too sweet!",
                        "Disgusting drink. Who will ever buy this? :(",
                        "Had a diarrhoea after drinking this juice. Not recommended",
                        "0 out of 100. Most nonsense food I ever eat!"]}
    return pd.DataFrame(data = data_dict)

def test_dataset(example_data):
    """Ensure that dataset attribute is updated when initialising Dataset class"""
    test = Dataset(example_data)
    assert test.data.shape[0] == example_data.shape[0]
    assert test.data.shape[1] == example_data.shape[1]
    assert test.data.columns.tolist() == example_data.columns.tolist()

def test_create_datasets(example_data):
    """
    Ensures that when create_datasets is runned multiple times,
    the same train and test datasets are generated
    Ensures that the train and test datasets have the same proportion
    of positive and negative comments
    """
    train_1, test_1 = create_datasets(example_data)
    train_2, test_2 = create_datasets(example_data)

    sentiment_counts_train = train_1.data['Sentiment'].value_counts()
    sentiment_counts_test = test_1.data['Sentiment'].value_counts()

    assert train_1.data.shape == train_2.data.shape
    assert test_1.data.shape == test_2.data.shape
    assert train_1.data['Text'].tolist() == train_2.data['Text'].tolist()
    assert test_1.data['Text'].tolist() == test_2.data['Text'].tolist()
    assert sentiment_counts_train['Positive'] == 10 * 0.8
    assert sentiment_counts_train['Negative'] == 5 * 0.8
    assert sentiment_counts_test['Positive'] == 10 * 0.2
    assert sentiment_counts_test['Negative'] == 5 * 0.2

def test_modify_stop_words_list(example_data):
    """
    Ensures that the function modifies the stop_words_list attribute in Dataclass
    """
    test = Dataset(example_data)
    test.modify_stop_words_list(replace_stop_words_list = ["I", "am", "very"],
                                include_words = ["happy"], exclude_words = ["very", "sad"])
    assert test.stop_words_list == ["I", "am", "happy"]

def test_word_tokenizer(example_data):
    """
    Ensures that each text is tokenised, with non alphabet characters such as numbers and punctuations removed
    """
    test = Dataset(example_data)
    expected_output_1 = ["hate", "potatoes", "rate", "Taste", "perfect"]

    test.word_tokenizer(False, ['noun', 'verb'])
    assert test.preprocessed_text[7] == expected_output_1

def test_removing_stop_words(example_data):
    """
    Ensures that the list does not contains any stopwords when lower cased
    """
    test = Dataset(example_data)

    expected_output_1 = ['hate', 'mash', 'potatoes', 'rate', 'taste', 'perfect']
    expected_output_2 = ["like", "drink"]

    test.word_tokenizer(True)
    test.removing_stop_words()
    assert test.preprocessed_text[7] == expected_output_1
    assert test.preprocessed_text[0] == expected_output_2

def test_stemming(example_data):
    """
    Ensures that tokenized words are stemmed accurately
    """
    test = Dataset(example_data)

    expected_output = ['hate', 'mash', 'potato', 'rate', 'Tast', 'perfect']

    test.word_tokenizer(lower_case = False)
    test.removing_stop_words()
    test.stemming(lower_case = False)
    assert test.preprocessed_text[7] == expected_output

def test_lemmatization(example_data):
    """
    Ensures that tokenized words are lemmatized accurately
    """
    test = Dataset(example_data)

    expected_output = ['i', 'love', 'this', 'chip', 'great', 'healthy', 'snack']

    test.word_tokenizer(lower_case = True)
    test.lemmatization()

    assert test.preprocessed_text[5] == expected_output

def test_processing_text(example_data):
    """
    Ensures that this function carries out the necessary preprocessing steps in sequence
    """
    test = Dataset(example_data)
    test1 = Dataset(example_data)
    test.preprocessing_text(root_word_option = 2, remove_stop_words = False,
                   lower_case = True, word_form = None)

    test1.word_tokenizer(lower_case = True)
    test1.lemmatization()

    assert test.preprocessed_text.tolist()[0] == test1.preprocessed_text.tolist()[0]

def test_create_bow(example_data):
    """
    Ensures that all documents are included in creating the bag of words
    """
    test = Dataset(example_data)

    test.preprocessing_text(root_word_option = 2, lower_case = True, remove_stop_words = True)
    test.create_bow(True)
    assert test.feature_engineer[2].toarray().shape[0] == test.data.shape[0]

def test_create_tfidf(example_data):
    """
    Ensures that all documents are included in creating tfidf
    Ensures that tokenized words are not lower cased as specified
    """
    test = Dataset(example_data)

    test.preprocessing_text(root_word_option = 1, lower_case = False, remove_stop_words = False)
    test.create_tfidf(False)
    assert test.feature_engineer[2].toarray().shape[0] == test.data.shape[0]
    assert test.feature_engineer[0].get_feature_names_out()[0].islower() is False

def test_create_doc2vec(example_data):
    """
    Ensures that all documents are vectorized
    Ensures that each document contain 100 dimensions
    """
    test = Dataset(example_data)

    test.preprocessing_text(0, False, True)
    test.create_doc2vec()

    assert len(test.doc2vec) == test.data.shape[0]
    assert len(test.doc2vec[0]) == 100

def test_create_word2vec(example_data):
    """
    Ensures that all the words are tagged with a corresponding number
    """
    test = Dataset(example_data)

    test.preprocessing_text(root_word_option = 2, lower_case = True, remove_stop_words = True)
    test.create_word2vec()
    expected = list(test.word2vec.keys())
    target = list(np.intersect1d(expected, list(test.feature_engineer[0].get_feature_names_out())))

    assert expected == target
