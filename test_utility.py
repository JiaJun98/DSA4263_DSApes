import pytest
import syspend
from utility import *

LOGGER_PATH = os.path.join(os.getcwd(),'logs', "utility_test_logs.log")

@pytest.fixture
def example_yml_file():
    """
    Fixture for test_parse_config. Gives a series of test
    
    Example usage:
    
    ```
    def test_parse_config(example_yml_file):
        # test code that uses the `example_yml_file` fixture
    ```
    """
    return [("non_bert.yml", "yml"), ("model.yml", "yml"), ("bert_config.yml", "yml"), ("topic_modelling.yml", "yml")]

def test_parse_config(example_yml_file):
    """Testing string to read into yml parser"""
    for yml_file in example_yml_file:
        file,file_type = yml_file
        assert isinstance(file, str)
        assert file.split(".")[-1] == file_type

def test_churn_eval_metrics():
    """Testing churn_eval_metrics to ensure predicted and test is the same length"""
    logger = open(LOGGER_PATH, "w")
    Y_pred = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    Y_test = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    assert len(Y_pred) == len(Y_test) #Test same length
    custom_print("Testing churn_eval_metrics", logger = logger)
    churn_eval_metrics(Y_pred, Y_test, logger)
    logger.close()
    



    
