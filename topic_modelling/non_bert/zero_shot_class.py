"""Try out zero shot model on pre-defined topics"""
from re import sub
import os
import sys
import pandas as pd
import numpy as np
from transformers import pipeline
from nltk.tokenize import sent_tokenize
sys.path.append("../..")
from model_base_class import BaseModel
from utility import parse_config, seed_everything



classifier = pipeline(task="zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device=0)

# Do not recommend reading in pickled zero_shot model as the file is too big (1.5GB)
# import pickle
# file_path = os.path.join(os.getcwd(), "zero_shot_model.pk")
# pickle.load(open(file_path, 'rb'))


class PredictTopic(BaseModel):
    """
    Zero shot model prediction with pre specified topic labels, 
    advised to run on GPU for huge datasets.
    """

    def __init__(self, text):
        self.text = pd.read_csv(text)
        self.labels = None
        self.results = None

    def train(self, labels):
        """Pass in the labels to be used for zero shot model"""
        self.labels = labels

    def predict(self):
        """
        return the results of zero_shot modelling
        """
        self.preprocess_text()
        predictions = pd.DataFrame(classifier(
            self.text['sentences'].tolist(), self.labels))
        self.postprocess(predictions)
        return self.results

    def preprocess_text(self):
        """
        Simple preprocessing by removing html tags and apply sentence tokeniser for all text
        """
        self.text['Text'] = self.text['Text'].apply(
            lambda x: sub("<[^>]+>", " ", str(x)).strip())
        test_tokenized = pd.DataFrame({"index": self.text.index, "original text": self.text['Text'],
                                       "sentences": self.text['Text'].apply(sent_tokenize)})
        test_tokenized = test_tokenized.explode(
            "sentences").reset_index(drop=True)
        self.text = test_tokenized

    def postprocess(self, predictions):
        """
        Assign text to sentences with max topic score.
        Those topic score with less than 0.5 are reflected as unknown
        """
        predictions['labels'] = predictions['labels'].apply(lambda x: x[0])
        predictions['scores'] = predictions['scores'].apply(lambda x: x[0])
        predictions.insert(0, "original index", self.text['index'])
        predictions = predictions.groupby("original index")["scores"].max().reset_index().merge(
            predictions, how="left", on=["original index", "scores"])
        predictions.loc[predictions['scores'] < 0.5, "labels"] = np.nan
        predictions = self.text.merge(predictions, how="right", left_on="index",
                                      right_on="original index")
        predictions = predictions.loc[:, ['original text', 'labels']]
        self.results = predictions.drop_duplicates()


if __name__ == "__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'zero_shot_config.yml')
    config_file = parse_config(config_path)
    home_folder = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    data_path = config_file['data_path']
    data_path = os.path.join(home_folder, data_path)
    output_path = os.path.join(curr_dir, config_file['output_path'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    seed_everything()

    file_output = os.path.join(output_path, "zero_shot_sample_prediction.csv")

    new_feedback = PredictTopic(data_path)
    new_feedback.train(["cat", "dog", "condiments", "snacks", "carbohydrates", "family",
                       "sauce", "drinks", "healthy alternatives", "household"])
    new_feedback.predict().to_csv(file_output, index=False)
