from bertopic import BERTopic
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
# import sys
import syspend
from preprocess_class import create_datasets
from model_base_class import BaseModel


# sys.path.append("../..")
class BERTopic_model(BaseModel):
    """
    BERTopic model for topic modelling. BERTopic is modular and the final topic model is dependent on the submodels chosen for each part of the task
    The parts of the model that an be modified is as follows: 
    1. Document embedding, 2. Dimensionality Reduction, 3. Clustering, 4. Tokenizer, 5. Weighting scheme 6. Representation Tuning (optional)
    """
    def __init__(self, embedding_model = None, dim_reduction_model=None,
                 clustering_model = None, vectorizer_model=None, 
                 ctfidf_model=None,  representation_model=None,
                 min_topic_size = 10):
        """
        @param embedding_model: Model to transform document into matrix of embedding
        @param dim_reduction_model: Dimensionality reduction algorithm to use
        @param clustering_model: Clustering algorithm to use
        @param vectorizer_model: Tokenizer to use
        @param ctfidf_model: weighting scheme to use
        @param representation_model: optional model to use to finetune the representations calculated using ctfidf
        """
        self.topic_model = None
        self.embedding_model = embedding_model
        self.dim_reduction_model = dim_reduction_model
        self.clustering_model = clustering_model
        self.vectorizer_model = vectorizer_model
        self.ctfidf_model = ctfidf_model
        self.representation_model =representation_model
        self.min_topic_size = min_topic_size
    
    def train(self, dataset, probability = False, nr_topics = 'auto'):
        """
        fit and transform the BERTopic model to the dataset
        @param dataset [Dataset]: Dataset for the model to be fit and transform on
        """
        self.topic_model = BERTopic(embedding_model=self.embedding_model, ctfidf_model=self.ctfidf_model,
                        vectorizer_model=self.vectorizer_model, 
                        min_topic_size= self.min_topic_size, 
                        representation_model=self.representation_model, 
                        umap_model = self.dim_reduction_model, 
                        hdbscan_model = self.clustering_model, 
                        nr_topics= nr_topics,
                        calculate_probabilities=probability, verbose=True)
        self.topic_model.fit_transform(dataset.text)

    def evaluate(self,dataset):
        """
        Evaluate performance of model using coherence_score. (Using normalise pointwise mutual information, range between -1 and 1, higher score is better)
        prints out coherence score and topic freqenucy
        @param dataset [Dataset]: Dataset to evaluate performance
        """
        c_score = self.get_coherence_score(dataset)
        return c_score
        
    def predict(self, dataset):
        '''
        Cluster the dataset into topics
        @param dataset Union[str,[Dataset]]: New dataset to predict
        @return prediction: Topic prediction for each document
        '''
        if type(dataset) == str:
            return self.topic_model.transform(dataset)
        else:
            return self.topic_model.transform(dataset.text)

    def load_model(self, path):
        '''
        Load previously trained topic model
        @param path [str]: path to model
        '''
        self.topic_model = BERTopic.load(path)
        
    def get_coherence_score(self, dataset):
        """
        Evaluation metric for model
        @param dataset [Dataset]: Training dataset
        @return c_score [float]: coherence score
        """
        documents = pd.DataFrame({"Document": dataset.text,
                                "ID": range(len(dataset.text)),
                                "Topic": self.topic_model.topics_})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.topic_model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = self.topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in self.topic_model.get_topic(topic)] 
                    for topic in range(len(set(self.topic_model.topics_))-1)]

        # Evaluate
        cm = CoherenceModel(topics=topic_words, 
                                        texts=tokens, 
                                        corpus=corpus,
                                        dictionary=dictionary, 
                                        coherence='c_npmi', #'u_mass', 'c_v', 'c_uci', 'c_npmi'
                                        topn=5)
        return cm.get_coherence()