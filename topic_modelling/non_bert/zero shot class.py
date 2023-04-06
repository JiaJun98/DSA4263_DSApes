import pandas as pd
import numpy as np
from re import sub
from transformers import pipeline
from nltk.tokenize import sent_tokenize

classifier = pipeline(task="zero-shot-classification", 
                      model="facebook/bart-large-mnli",
                      device=0) 

class PredictTopic:
    def __int__(self, text):
        self.text = text
        self.labels = ["cat", "dog", "condiments", "snacks", "carbohydrates", "family", 
                    "sauce", "drinks", "healthy alternatives", "household"]
        self.results = None
        
    def predict(self):
        self.preprocess_text()
        predictions = classifier(self.text['sentences'].tolist(), self.labels)
        self.postprocess(predictions)
        return self.results

    def preprocess_text(self):
        df = pd.DataFrame(self.text)
        df = df.apply(lambda x: sub("<[^>]+>", " ", x).strip())
        test_tokenized = pd.DataFrame({"index":df.index, "original text": df, "sentences":df.apply(sent_tokenize)})
        test_tokenized = test_tokenized.explode("sentences").reset_index(drop = True)
        self.text = test_tokenized
    
    def postprocess(self, predictions):
        predictions['labels'] = predictions['labels'].apply(lambda x: x[0])
        predictions['scores'] = predictions['scores'].apply(lambda x: x[0])
        predictions.insert(0, "original index", self.text['index'])
        predictions = predictions.groupby("original index")["scores"].max().reset_index().merge(predictions, how = "left", on = ["original index", "scores"])
        predictions.loc[predictions['scores'] < 0.5, "labels"] = np.nan
        self.results = predictions.loc[:, ['original text', 'labels']]
