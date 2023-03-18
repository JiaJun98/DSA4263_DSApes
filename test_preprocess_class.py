#add pytest in environment.yml
from preprocess_class import *
from pandas.api.types import is_datetime64_any_dtype as is_datetime

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

def test_datasets_date(example_data):
    test = Dataset(example_data)

    assert is_datetime(test.date)

def test_datasets_sentiments(example_data):
    test = Dataset(example_data)

    expected_output = ['Positive'] * 10 + ['Negative'] * 5
    assert test.sentiments.tolist() == expected_output


def test_datasets_text(example_data):
    """
    Confirm that it reads in the text from the dataset and removed all html tags
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



