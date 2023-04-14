import pandas as pd
import syspend
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import sys
import os
from utility import parse_config, custom_print
from preprocess_class import create_datasets
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from BERTopic_model import BERTopic_model
sys.path.append("../..")

###### Driver class
if __name__ =="__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'bert_topic_config.yml')
    config_file = parse_config(config_path)
    data_file = config_file['data_folder']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    data_df = pd.read_csv(os.path.join(home_folder,data_file))
    train, test = create_datasets(data_df)
    for i in range(len(config_file)-3):
        cur_model = f'model_{i}'
        model_name = config_file[cur_model]['model_name']
        embedding_model = eval(config_file[cur_model]['embedding_model'])
        dim_reduction_model = eval(config_file[cur_model]['dim_reduction_model'])
        clustering_model = eval(config_file[cur_model]['clustering_model'])
        vectorizer_params = dict(config_file[cur_model]['vectorizer_params'])    
        ctfidf_params = dict(config_file[cur_model]['ctfidf_params'])
        representation_model = eval(config_file[cur_model]['representation_model'])
        min_topic_size = int(config_file[cur_model]['min_topic_size'])
        nr_topics = config_file[cur_model]['nr_topics']
        logging_path = os.path.join(curr_dir,config_file['log_path'],f'{model_name}.log')
        image_path = os.path.join(curr_dir,config_file['image_folder'])
        ctfidf_model = ClassTfidfTransformer(bm25_weighting= ctfidf_params["bm25_weighting"], reduce_frequent_words= ctfidf_params["reduce_frequent_words"])        
        vectorizer_model = CountVectorizer(stop_words=train.stop_words_list, min_df = vectorizer_params['min_df'], max_df = vectorizer_params['min_df'],\
                                    ngram_range=(1,vectorizer_params['ngram_range']))
        logger = open(os.path.join(curr_dir, logging_path), 'w')
        custom_print('Training model',logger = logger)
        custom_print('---------------------------------\n',logger = logger)
        custom_print(f"model name: {model_name}",logger = logger)
        model = BERTopic_model(embedding_model = embedding_model,
                            dim_reduction_model=dim_reduction_model,
                            clustering_model = clustering_model,
                            vectorizer_model=vectorizer_model, 
                            ctfidf_model=ctfidf_model, 
                            representation_model=representation_model,
                            min_topic_size = min_topic_size)
        if nr_topics == 'None':
            nr_topics = None
        model.train(train.data['Text'], probability=False, nr_topics=nr_topics)
        custom_print(f"Coherence score: {model.evaluate(train.data['Text'])}", logger=logger)
        custom_print(f'{model.topic_model.get_topic_info()}', logger=logger)

        for i in range(len(model.topic_model.topic_labels_)-1):
            custom_print(f'Words and score of topic {i}:\n {model.topic_model.get_topic(i)}',
                            logger=logger)
        fig = model.topic_model.visualize_documents(train.data['Text'])
        fig.write_html(f"{image_path}{model_name}_doc_viz.html")
        model.topic_model.save(f"{model_name}")
    logger.close()