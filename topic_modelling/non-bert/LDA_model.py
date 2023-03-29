import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append("../..")
from preprocess_class import *
from model_base_class import BaseModel

class LDAmodel(BaseModel):
    def __init__(self, dataset):
        self.dataset = dataset
        self.text = None
        self.model = None

    def train(self, num_of_topics, export_topics = False):
        lda = LatentDirichletAllocation(n_components= num_of_topics, random_state = 0)
        training_lda = lda.fit_transform(self.dataset.bow[1])
        self.model = lda
        topic_key_words, topic_doc = self.display_topics(lda.components_, training_lda, self.dataset.bow[0].get_feature_names_out(), self.dataset.text, 20, 10, export_topics)
        coherence = self.get_coherence(topic_key_words, self.text)
        return coherence

    def predict(self, new_data, export_topics = True):
        testing_lda = self.model.fit_transform(new_data)
        topic_key_words, topic_doc = self.display_topics(self.model.components_, testing_lda, new_data.bow[0].get_feature_names_out(), new_data.text, 20, 10, export_topics)
        coherence = self.get_Cv(topic_key_words, self.text)
        return coherence

    def preprocess_dataset(self, replace_stop_words_list = None, include_words = ['taste', 'flavor', 'amazon', 'price', 'minute', 'time', 'year'],
                            exclude_words = ["not", "no", "least", "less", "last", "serious", "too", "again", "against", "already", "always", "cannot", "few", "must", "only", "though"],
                            root_word_option = 0, remove_stop_words = True, lower_case = True, 
                            word_form = None, ngrams = (1,1), max_doc = 1, min_doc = 1):
        self.dataset.modify_stop_words_list(replace_stop_words_list, include_words, exclude_words)
        self.dataset.create_bow(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)
        if root_word_option == 0:
            if remove_stop_words:
                self.text = self.dataset.tokenized_no_stop_words
            else:
                self.text = self.dataset.tokenized_words
        
        elif root_word_option == 1:
            self.text = self.dataset.stem
        
        elif root_word_option == 2:
            self.text = self.dataset.lemmatize
        
        else:
            raise Exception("invalid root word option")

    def display_topics(self, H, W, feature_names, documents, no_top_words, no_top_documents, export_topics = False):
        topic_key_words = []
        topic_doc = []
        
        for topic_idx, topic in enumerate(H):
            # print ("Topic %d:" % (topic_idx))
            # print (" ".join([feature_names[i]
            #                 for i in topic.argsort()[:-no_top_words - 1:-1]]))
            topic_key_words.append([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]])
            top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
            curr_topic_doc = []
            for doc_index in top_doc_indices:
                curr_topic_doc.append(documents[doc_index])
            
            topic_doc.append(curr_topic_doc)
        
        num_of_topics = len(topic_key_words)
        if export_topics:
            pd.DataFrame(topic_key_words).to_csv("topic_key_words_{}.csv".format(num_of_topics))
            pd.DataFrame(topic_doc).to_csv("topic_doc_{}.csv".format(num_of_topics))
        return (topic_key_words, topic_doc)

    def get_coherence(self, topic_words, texts, measure = 'u_mass'):
        dictionary = corpora.Dictionary(texts)
        coherence_model = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence= measure)
        coherence = coherence_model.get_coherence()
        return coherence
    
    def compare_coherence_scores(self, start_topic_num, end_topic_num, step):
        # Show graph
        coherence_values = []

        for i in range(start_topic_num, end_topic_num, step):
            coherence_values.append(self.train(i))

        x = range(start_topic_num, end_topic_num, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()
        plt.savefig("coherence_values_comparison.csv")

    def dump_model(self):
        pickle.dump(self.model, 'lda_model.pk')

df = pd.read_csv("../../reviews.csv")
lda_model = LDAmodel(Dataset(df))
lda_model.preprocess_dataset(replace_stop_words_list= None,root_word_option= 2,remove_stop_words= True,lower_case= True, word_form= ["noun"], ngrams=(1,2), max_doc=0.9, min_doc=5)
lda_model.train(7, export_topics=True)
lda_model.dump_model()
print(lda_model.get_Cv)
# pickle.dump(lda_model, 'lda_model.pk')
# # then reload it with
# lda_model = pickle.load('lda_model.pk')
