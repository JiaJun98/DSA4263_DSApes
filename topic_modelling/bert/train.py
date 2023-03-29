from bertopic import BERTopic
import pandas as pd
import numpy as np
import syspend
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from bertopic.vectorizers import ClassTfidfTransformer
import sys
sys.path.append("../..")
import os
from utility import parse_config, seed_everything, custom_print
from preprocess_class import create_datasets
from model_base_class import BaseModel
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

class BERTopic_model(BaseModel):
    """
    BERTOpic model for topic modelling task
    """
    def __init__(self, embedding_model = None, clustering_model = None, vectorizer_model=None, 
                 ctfidf_model=None, dim_reduction_model=None, representation_model=None,
                 min_topic_size = 10):
        self.topic_model = None
        self.embedding_model = embedding_model
        self.clustering_model = clustering_model
        self.vectorizer_model = vectorizer_model
        self.ctfidf_model = ctfidf_model
        self.dim_reduction_model = dim_reduction_model
        self.representation_model =representation_model
        self.min_topic_size = min_topic_size
    
    def train(self, dataset):
        self.topic_model = BERTopic(embedding_model=self.embedding_model, ctfidf_model=self.ctfidf_model,\
                        vectorizer_model=self.vectorizer_model, \
                        min_topic_size= self.min_topic_size,
                        representation_model=self.representation_model, \
                        umap_model = self.dim_reduction_model, \
                        hdbscan_model = self.clustering_model, \
                        nr_topics= 'auto',
                        calculate_probabilities=False, verbose=True)
        self.topic_model.fit_transform(dataset.text)

    def get_coherence_score(dataset, topic_model):
        """
        Evaluation metric for model
        """
        documents = pd.DataFrame({"Document": dataset.text,
                                "ID": range(len(dataset.text)),
                                "Topic": topic_model.topics_})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                    for topic in range(len(set(topic_model.topics_))-1)]

        # Evaluate
        cm = CoherenceModel(topics=topic_words, 
                                        texts=tokens, 
                                        corpus=corpus,
                                        dictionary=dictionary, 
                                        coherence='c_npmi', #'u_mass', 'c_v', 'c_uci', 'c_npmi'
                                        topn=5)
        return cm.get_coherence()
    def evaluate(self,dataset):
        """
        Evaluate performance of model using coherence_score. As it is a unsupervised method, we need to manually check if the topics make sense as well
        """
        c_score = get_coherence_score(dataset, self.topic_model)
        return c_score
    
    def predict(self, dataset):
        return self.topic_model.transform(dataset.text)


###### Driver class
if __name__ == "__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'bert_topic_config.yml')
    config_file = parse_config(config_path)
    model_name = config_file['model']['model_name']
    embedding_model = eval(config_file['model']['embedding_model'])
    dim_reduction = eval(config_file['model']['dim_reduction'])
    clustering_model = eval(config_file['model']['clustering_model'])
    vectorizer_params = eval(config_file['model']['vectorizer_params'])    
    ctfidf_params = eval(config_file['model']['ctfidf_params'])
    representation_model = eval(config_file['model']['representation_model'])
    min_topic_size = int(config_file['model']['min_topic_size'])
    data_file = config_file['model']['data_folder']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    data_df = pd.read_csv(os.path.join(home_folder,data_file))
    # model_path = os.path.join(curr_dir, config_file['model']['model_path'])
    logging_path = os.path.join(curr_dir,config_file['model']['log_path'])
    image_path = os.path.join(curr_dir,config_file['model']['image_folder'])
    train, test = create_datasets(data_df)
    vectorizer_model = CountVectorizer(stop_words=train.stop_words_list, min_df = vectorizer_params['min_df'], max_df = vectorizer_params['min_df'],\
                                   ngram_range=vectorizer_params['ngram_range'])
    ctfidf_model = ClassTfidfTransformer(bm25_weighting= ctfidf_params["bm25_weighting"], reduce_frequent_words= ctfidf_params["reduce_frequent_words"])        
    logger = open(os.path.join(curr_dir, logging_path), 'w')
    # custom_print(f'Device availiable: {device}', logger = logger)
    custom_print('Training model',logger = logger)
    # seed_everything()
    custom_print('---------------------------------\n',logger = logger)
    # custom_print("Hyperparameters:",logger = logger)
    custom_print(f"model name: {model_name}",logger = logger)
    # custom_print(f"Number of vectorizer_params: {vectorizer_params}",logger = logger)
    # custom_print(f"number of classes: {n_classes}",logger = logger)
    # custom_print(f"max length: {dim_reduction}",logger = logger)
    # custom_print(f"batch size: {clustering_model}",logger = logger)
    # custom_print(f"learning rate: {ctfidf_params}",logger = logger)
    model = BERTopic_model(embedding_model = embedding_model,
                        dim_reduction_model=dim_reduction_model,
                        clustering_model = clustering_model,
                        vectorizer_model=vectorizer_model, 
                        ctfidf_model=ctfidf_model, 
                        representation_model=representation_model,
                        min_topic_size = min_topic_size)
    model.train(train)
    custom_print(f'Coherence score: {model.evaluate(train)}', logger=logger)
    custom_print(f'{model.topic_model.get_topic_info()}', logger=logger)
    
    for i in range(len(model.topic_model.topic_labels_)-1):
        custom_print(f'Words and score of topic {i}:\n {model.topic_model.get_topic(i)}',
                     logger=logger)
    fig = model.topic_model.visualize_documents(train.text)
    fig.write_html(f"{image_path}doc_viz.html")
    logger.close()