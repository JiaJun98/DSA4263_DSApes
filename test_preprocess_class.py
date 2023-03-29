#add pytest in environment.yml
import sys
sys.path.append(".")
from preprocess_class import *
from pandas.api.types import is_datetime64_any_dtype as is_datetime
# import warnings
import pytest

@pytest.fixture
def example_data():
    return pd.DataFrame(data = {'Sentiment': ['Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive',
                                              'Negative','Negative','Negative','Negative','Negative'],
                                'Time': ['1/1/2021', '2/1/2021', '21/7/2021', '31/8/2022', '1/7/2022', '8/9/2021', '9/9/2029', '9/10/2012', '10/8/2023', 
                                         '9/4/2022', '4/5/2021', '13/5/2021', '18/3/2022', '23/4/2021', '3/3/2021'],
                                'Text': ["I like this drink", "The taste is just nice for me", "<br /><br /> This ice cream is becoming my all-time favourite",
                                         "This drink is not too sweet. Fantastic!", "Delicious fish with the ultimate flavor, best  ever in a can for all purposes ! ",
                                         "I love this chips! Great healthy snacks!</ br ahref:  https://iron.com>", "No salt, no sugar. Yet this chips is still so flavourful!",
                                         "I hate mash potatoes. But I still rate this 5 / 5. Taste perfect!", "No better food than this mash potatoes!", "Perfect 10. Marvellous potatoes!",
                                         "THIS IS THE WORST POTATOES I HAVE EVER EATEN", "<a href::item.com> This ice cream is too sweet!",
                                         "Disgusting drink. Who will ever buy this? :(", "Had a diarrhoea after drinking this juice. Not recommended",
                                         "0 out of 100. Most nonsense food I ever eat!"]})

def test_dataset_date(example_data):
    """
    Ensure that the date attribute correctly reflects the 'Time' column in datetime format upon declaring Dataset class
    """
    test = Dataset(example_data)

    assert is_datetime(test.date)

def test_dataset_sentiments(example_data):
    """
    Ensure that the sentiments attribute correctly reflects the 'Sentiment' column upon declaring Dataset class
    """
    test = Dataset(example_data)

    expected_output = ['Positive'] * 10 + ['Negative'] * 5
    assert test.sentiments.tolist() == expected_output


def test_dataset_text(example_data):
    """
    Ensure that the text attribute correctly reflects the 'Text' column after removing all html tags from all the strings
    """
    test = Dataset(example_data)
    expected_output = ["I like this drink", "The taste is just nice for me", "This ice cream is becoming my all-time favourite",
                        "This drink is not too sweet. Fantastic!", "Delicious fish with the ultimate flavor, best  ever in a can for all purposes !",
                        "I love this chips! Great healthy snacks!", "No salt, no sugar. Yet this chips is still so flavourful!",
                        "I hate mash potatoes. But I still rate this 5 / 5. Taste perfect!", "No better food than this mash potatoes!", "Perfect 10. Marvellous potatoes!",
                        "THIS IS THE WORST POTATOES I HAVE EVER EATEN", "This ice cream is too sweet!",
                        "Disgusting drink. Who will ever buy this? :(", "Had a diarrhoea after drinking this juice. Not recommended",
                        "0 out of 100. Most nonsense food I ever eat!"]
    
    test_text = test.text.tolist()
    assert test_text == expected_output

def test_create_datasets(example_data):
    """
    Ensures that when create_datasets is runned multiple times, the same train and test datasets are generated
    Ensures that the train and test datasets have the same proportion of positive and negative comments
    """
    train_1, test_1 = create_datasets(example_data)
    train_2, test_2 = create_datasets(example_data)

    sentiment_counts_train = train_1.sentiments.value_counts()
    sentiment_counts_test = test_1.sentiments.value_counts()

    assert train_1.sentiments.tolist() == train_2.sentiments.tolist()
    assert test_1.sentiments.tolist() == test_2.sentiments.tolist()
    assert train_1.date.tolist() == train_2.date.tolist()
    assert test_1.date.tolist() == test_2.date.tolist()
    assert train_1.text.tolist() == train_2.text.tolist()
    assert test_1.text.tolist() == test_2.text.tolist()
    assert sentiment_counts_train['Positive'] == 10 * 0.8
    assert sentiment_counts_train['Negative'] == 5 * 0.8
    assert sentiment_counts_test['Positive'] == 10 * 0.2
    assert sentiment_counts_test['Negative'] == 5 * 0.2

def test_modify_stop_words_list(example_data):
    test = Dataset(example_data)
    test.modify_stop_words_list(replace_stop_words_list = ["DSA4263", "NUS", "Faculty of Science"], include_words = [], exclude_words = [])
    assert test.stop_words_list == ["DSA4263", "NUS", "Faculty of Science"]

def test_word_tokenizer(example_data):
    """
    Ensures that each text is tokenised, with non alphabet characters such as numbers and punctuations removed
    """
    test = Dataset(example_data)
    expected_output_1 = ["hate", "potatoes", "rate", "Taste", "perfect"]

    test.word_tokenizer(False, ['noun', 'verb'])
    assert test.tokenized_words[7] == expected_output_1

def test_sentence_tokenizer(example_data):
    """
    Ensures that each text is split into individual sentences
    """
    test = Dataset(example_data)
    expected_output = ["I hate mash potatoes.", "But I still rate this 5 / 5.", "Taste perfect!"]

    test.sentence_tokenizer()
    assert test.tokenized_sentence[7] == expected_output

def test_removing_stop_words(example_data):
    """
    Ensures that each text is tokenised, with non alphabet characters such as numbers and punctuations removed
    Ensures that the list does not contains any stopwords when lower cased
    Ensures that lower_case parameter enforced. 'Taste' is stored as 'taste'
    """
    test = Dataset(example_data)

    expected_output_1 = ['hate', 'mash', 'potatoes', 'rate', 'taste', 'perfect']
    expected_output_2 = ["like", "drink"]

    test.removing_stop_words(True)
    assert test.tokenized_no_stop_words[7] == expected_output_1
    assert test.tokenized_no_stop_words[0] == expected_output_2

def test_input_text(example_data):
    """
    Ensures that the right set of words is returned from input_text function
    """
    test = Dataset(example_data)

    test_output = test.input_text(0, True, False)
    expected_output = test.tokenized_no_stop_words

    assert test_output.tolist() == expected_output.tolist()

def test_stemming(example_data):
    """
    Ensures that tokenized words are stemmed accurately
    Ensures stop words are removed as specified
    Ensures that tokenized words are not lower cased
    """
    test = Dataset(example_data)

    expected_output = ['hate', 'mash', 'potato', 'rate', 'Tast', 'perfect']

    test.stemming(remove_stop_words = True, lower_case = False)
    assert test.stem[7] == expected_output

def test_lemmatization(example_data):
    """
    Ensures that tokenized words are lemmatized accurately
    Ensures that stop words remains as specified
    Ensures that tokenized words are lower cased
    """
    test = Dataset(example_data)

    expected_output = ['i', 'love', 'this', 'chip', 'great', 'healthy', 'snack']

    test.lemmatization(False, True)

    assert test.lemmatize[5] == expected_output

def test_create_bow(example_data):
    """
    Ensures that all documents are included in creating the bag of words
    """
    test = Dataset(example_data)

    test.create_bow(2, True, True)
    assert test.bow[1].toarray().shape[0] == len(test.text)

def test_create_tfidf(example_data):
    """
    Ensures that all documents are included in creating tfidf
    Ensures that tokenized words are not lower cased as specified
    """
    test = Dataset(example_data)

    test.create_tfidf(1, True, False)
    assert test.tfidf[1].toarray().shape[0] == len(test.text)
    assert test.tfidf[0].get_feature_names_out()[0].islower() == False

def test_create_doc2vec(example_data):
    """
    Ensures that all documents are vectorized
    Ensures that each document contain 100 dimensions
    """
    test = Dataset(example_data)

    test.create_doc2vec(0, False, True)

    assert len(test.doc2vec) == len(test.text)
    assert len(test.doc2vec[0]) == 100

def test_create_word2vec(example_data):
    """
    Ensures that all the words are tagged with a corresponding number
    """
    test = Dataset(example_data)

    test.create_word2vec()
    assert len(test.word2vec) == len(test.bow[0].get_feature_names_out())