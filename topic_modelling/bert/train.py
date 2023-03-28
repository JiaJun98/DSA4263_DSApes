from bertopic import BERTopic
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
import sys
sys.path.append("../..")
from utility import parse_config, seed_everything, custom_print
from preprocess_class import create_datasets
from model_base_class import BaseModel
from umap import UMAP
from hdbscan import HDBSCAN

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

    def evaluate(self,dataset):
        """
        Evaluate performance of model using coherence_score. As it is a unsupervised method, we need to manually check if the topics make sense as well
        """
        c_score = get_coherence_score(dataset, self.topic_model)
        return c_score
    
    def predict(self, dataset):
        return self.topic_model.transform(dataset.text)

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

###### Driver class
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     curr_dir = os.getcwd()
#     config_path = os.path.join(curr_dir, 'bert_sentiment_config.yml')
#     config_file = parse_config(config_path)
#     model_name = config_file['model']['model_name']
#     n_classes = int(config_file['model']['n_classes'])
#     max_len = int(config_file['model']['max_len'])
#     batch_size = int(config_file['model']['batch_size'])
#     epochs = int(config_file['model']['epochs'])
#     learning_rate = float(config_file['model']['learning_rate'])
#     epsilon = float(config_file['model']['epsilon'])
#     train_on_full_data = eval(str(config_file['model']['train_on_full_data']))
#     train_file = config_file['model']['data_folder']
#     isTrainer = config_file['model']['trainer']
#     home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
#     model_path = os.path.join(curr_dir, config_file['model']['model_path'])
#     logging_path = os.path.join(curr_dir,config_file['model']['log_path'])
        
#     data_df = pd.read_csv(os.path.join(home_folder,train_file))
#     logger = open(os.path.join(curr_dir, logging_path), 'w')
#     custom_print(f'Device availiable: {device}', logger = logger)
#     train_df, test_df = train_test_split(data_df, test_size = 0.2, random_state = 4263) #Trainer Arguments
#     train_dataset, val_dataset = full_bert_data_loader(model_name,max_len, batch_size, True, train_df) #Trainer Arguments
#     custom_print("Train_val dataset loaded",logger = logger)
#     custom_print('Training model',logger = logger)
#     seed_everything()
#     custom_print('---------------------------------\n',logger = logger)

#     custom_print("Hyperparameters:",logger = logger)
#     custom_print(f"model name: {model_name}",logger = logger)
#     custom_print(f"Number of epochs: {epochs}",logger = logger)
#     custom_print(f"number of classes: {n_classes}",logger = logger)
#     custom_print(f"max length: {max_len}",logger = logger)
#     custom_print(f"batch size: {batch_size}",logger = logger)
#     custom_print(f"learning rate: {learning_rate}",logger = logger)
    
#     if isTrainer:
#         trainer = train(model_name, train_dataset, val_dataset)
#         custom_print('Training complete!',logger = logger)
#         custom_print('Showing Training and Evaluation metrics....',logger = logger)
#         #https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
#         for obj in trainer.state.log_history:
#             for key,value in obj.items():
#                 custom_print(f'{key}: {value}', logger = logger)
#     else:
#         custom_print('Loading data.....',logger = logger)
#         sentimental_classifier = BertClassifier(model_name, n_classes)
#         sentimental_classifier.model.to(device)
#         custom_print('Model initialised!', logger = logger)
#         if not train_on_full_data:
#             train, test = train_test_split(data_df, test_size = 0.2, random_state = 4263)
#             custom_print(f"train size: {len(train)}",logger = logger)
#             custom_print(f"dev size: {len(test)}",logger = logger)
#             train_dataloader = create_data_loader(model_name, batch_size,max_len, train)
#             custom_print('Train data loaded!', logger = logger)
#             val_dataloader = create_data_loader(model_name, batch_size,max_len, test, predict_only=False)
#             sentimental_classifier.train(learning_rate, epsilon,train_dataloader, val_dataloader,epochs =1, evaluation=True, logger = logger)
#         else:
#             full_train_data = full_create_data_loader(model_name, batch_size,max_len, data_df)
#             custom_print('Full Data loaded!',logger = logger)
#             sentimental_classifier.train(learning_rate, epsilon,full_train_data, epochs =1, logger = logger)
#         custom_print('Saving model ...', logger = logger)
#         torch.save({'model_state_dict':sentimental_classifier.model.state_dict()}, model_path)
#     logger.close()