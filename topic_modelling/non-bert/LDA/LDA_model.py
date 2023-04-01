import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
import gensim.corpora as corpora
# from nltk.tag import pos_tag
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append("../../..")
import syspend

from utility import parse_config, seed_everything, custom_print, churn_eval_metrics
from preprocess_class import Dataset, create_datasets
from model_base_class import BaseModel

class LDAmodel(BaseModel):
    def __init__(self, train_dataset = None, test_dataset = None, pickled_model = None, topic_label = None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.text = None
        self.model = None
        self.num_of_topics = None
        self.topic_label = pd.DataFrame(topic_label, columns = ['Topic_no', 'Topic_label']) if topic_label is not None else pd.DataFrame(columns = ['Topic_no', "Topic_label"])
        self.pickled_model = pickle.load(open(pickled_model, 'rb')) if pickled_model is not None else None

    def train(self, num_of_topics, num_top_words, num_top_documents, train_output_path, coherence_measure = 'c_v'):
        self.num_of_topics = num_of_topics
        custom_print("------ Training LDA model --------\n", logger = logger)
        lda = LatentDirichletAllocation(n_components= num_of_topics, random_state = 4263)
        training_lda = lda.fit_transform(self.train_dataset.bow[1])
        self.model = lda

        doc_topic_labels = pd.DataFrame(training_lda).idxmax(axis = 1)
        num_of_doc_per_topic = doc_topic_labels.value_counts().sort_index()

        custom_print("------ Distribution for number of documents in each topic --------\n", logger = logger)
        for i in range(len(num_of_doc_per_topic)):
            custom_print("Topic {}: {}".format(i, num_of_doc_per_topic[i]),logger = logger)

        topic_key_words = self.display_topics(lda.components_, training_lda, self.train_dataset.bow[0].get_feature_names_out(), 
                                                         self.train_dataset.text, num_top_words, num_top_documents, train_output_path)
        
        labelled_train_topics = pd.DataFrame({"Text": self.train_dataset.text, "Labelled_topic": doc_topic_labels})
        custom_print("------ Generating key words and sample documents for each topic ------------\n", logger = logger)

        for i in range(self.num_of_topics):
            curr_topic_key_words = ", ".join(topic_key_words[i])
            custom_print("\nTopic {}".format(i), logger = logger)
            custom_print(curr_topic_key_words, logger = logger)

            curr_topic_doc = labelled_train_topics.loc[labelled_train_topics['Labelled_topic'] == i, :]
            topic_samples = curr_topic_doc["Text"].sample(n = num_top_documents, random_state=4263)
            
            for text in topic_samples:
                custom_print("\n" + text, logger = logger)

            self.set_topic_labels(i)
            full_topic_path = os.path.join(train_output_path, "full_doc_{}.csv".format(self.topic_label.loc[i, "Topic_label"]))
            sampled_topic_path = os.path.join(train_output_path, "sample_doc_{}.csv".format(self.topic_label.loc[i, "Topic_label"]))

            curr_topic_doc.to_csv(full_topic_path)
            topic_samples.to_csv(sampled_topic_path)

        topic_key_words_df = pd.DataFrame(topic_key_words, index = self.topic_label['Topic_label'])
        topic_key_words_path = os.path.join(train_output_path, "topic_key_words.csv")
        topic_key_words_df.to_csv(topic_key_words_path)

        coherence = self.get_coherence(topic_key_words, self.text, coherence_measure)
        custom_print("Overall training coherence score: {}".format(coherence), logger = logger)
        custom_print("-------- End of training for {} topics ----------".format(num_of_topics), logger = logger)

    def predict(self, test_output_path, num_top_documents):
        if self.model is not None:
            testing_lda = self.model.fit_transform(self.test_dataset.bow[1])
        elif self.pickled_model is not None:
            testing_lda = self.pickled_model.fit_transform(self.test_dataset.bow[1])
        assigned_topic = pd.DataFrame(testing_lda).idxmax(axis = 1)
        num_of_doc_per_topic = assigned_topic.value_counts().sort_index()
        custom_print("------ Number of documents assigned to each topic--------\n", logger = logger)
        for i in range(len(num_of_doc_per_topic)):
            custom_print("Topic {}: {}".format(i, num_of_doc_per_topic[i]),logger = logger)
        
        custom_print("-------- Exporting labelled topics ----------", logger = logger)
        labelled_test = pd.DataFrame({"Text": self.test_dataset.text, "Topic_no": assigned_topic})
        labelled_test.merge(self.topic_label, how = "left", on = "Topic_no") 

        topic_accuracy = []
        for i in range(self.topic_label.shape[0]):
            curr_topic = labelled_test.loc[labelled_test['Topic_no']== i, "Text"]
            curr_topic_samples = curr_topic.sample(n=num_top_documents, random_state=4263)
            custom_print("Current topic: {}".format(self.topic_label.loc[i, "Topic_label"]), logger = logger)

            for sample in curr_topic_samples:
                custom_print("\n" + sample, logger = logger)

            correctly_labelled = input("Number of correct labels: ")
            try:
                correctly_labelled = int(correctly_labelled)
            
            except:
                raise ValueError
            
            if correctly_labelled < 0 or correctly_labelled > num_top_documents:
                raise Exception("Number of correct labels cannot exceed number of samples!")
            accuracy = int(correctly_labelled) / num_top_documents
            topic_accuracy.append(accuracy)
            custom_print("{} topic testing accuracy: {}".format(self.topic_label.loc[i, "Topic_label"], str(accuracy)), logger = logger)
        
        average_topic_accuracy = sum(topic_accuracy) / len(topic_accuracy)
        custom_print("-------- Average topic accuracy: {} --------".format(str(average_topic_accuracy)), logger = logger)
        custom_print("-------- Exporting labelled topics ----------", logger = logger)
        labelled_test_output_path = os.path.join(test_output_path, "full_labelled_test_dataset.csv")
        labelled_test.to_csv(labelled_test_output_path, index = False)
        custom_print("-------- End of predicting for {} topics ----------".format(num_of_topics), logger = logger)

    def preprocess_dataset(self, replace_stop_words_list = None, include_words = ['taste', 'flavor', 'amazon', 'price', 'minute', 'time', 'year'],
                            exclude_words = ["not", "no", "least", "less", "last", "serious", "too", "again", "against", "already", "always", "cannot", "few", "must", "only", "though"],
                            root_word_option = 0, remove_stop_words = True, lower_case = True, 
                            word_form = None, ngrams = (1,1), max_doc = 1, min_doc = 1):
        if self.train_dataset is not None:
            self.train_dataset.modify_stop_words_list(replace_stop_words_list, include_words, exclude_words)
            self.train_dataset.create_bow(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)

            if root_word_option == 0:
                if remove_stop_words:
                    self.text = self.train_dataset.tokenized_no_stop_words
                else:
                    self.text = self.train_dataset.tokenized_words
            
            elif root_word_option == 1:
                self.text = self.train_dataset.stem
            
            elif root_word_option == 2:
                self.text = self.train_dataset.lemmatize
            
            else:
                raise Exception("invalid root word option")
        if self.test_dataset is not None:
            self.test_dataset.modify_stop_words_list(replace_stop_words_list, include_words, exclude_words)
            self.test_dataset.create_bow(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)

    def display_topics(self, trained_topics, list_of_words, feature_names, documents, num_top_words, num_top_documents, train_output_path):
        topic_key_words = []
        
        for topic in trained_topics:
            topic_key_words.append([feature_names[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]])

        topic_vis_path = os.path.join(train_output_path, "topic_vis_{}.html".format(self.num_of_topics))
        panel = pyLDAvis.sklearn.prepare(self.model, self.train_dataset.bow[1], self.train_dataset.bow[0], mds='tsne')
        pyLDAvis.save_html(panel, topic_vis_path)
        return topic_key_words

    def get_coherence(self, topic_words, texts, measure = 'c_v'):
        dictionary = corpora.Dictionary(texts)
        coherence_model = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence= measure)
        coherence = coherence_model.get_coherence()
        return coherence

    def dump_model(self, train_output_path):
        train_output_path = os.path.join(train_output_path,'lda_model_{}.pk'.format(self.num_of_topics))
        pickle.dump(self.model, open(train_output_path, 'wb'))

    def set_topic_labels(self, topic_no):
        curr_topic_label = input("Label for Topic {}: ".format(topic_no))
        self.topic_label.loc[len(self.topic_label)] = [len(self.topic_label), curr_topic_label]

def train_test(train_dataset, train_output_path, num_of_topics, num_top_words, num_top_documents, coherence_measure, replace_stop_words_list, include_words, 
          exclude_words, root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc, test_dataset = None, test_output_path = None):
    LDA_model = LDAmodel(train_dataset=train_dataset, test_dataset = test_dataset)
    LDA_model.preprocess_dataset(replace_stop_words_list, include_words, exclude_words, root_word_option,
                                 remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)
    LDA_model.train(num_of_topics, num_top_words, num_top_documents, train_output_path, coherence_measure)
    LDA_model.dump_model(train_output_path)

    if test_dataset is not None:
        LDA_model.predict(test_output_path, num_top_documents)

def test(test_dataset, pickled_model, test_output_path, topic_label, num_top_documents, replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc):
    testLDA = LDAmodel(test_dataset=test_dataset, pickled_model=pickled_model, topic_label = topic_label)
    testLDA.preprocess_dataset(replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc)
    testLDA.predict(test_output_path=test_output_path, num_top_documents = num_top_documents)

if __name__ == "__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'LDA_topic_modelling_config.yml')
    config_file = parse_config(config_path)
    model_name = config_file['model']['model_name']
    train_file = config_file['model']['data_folder']
    isTrainer = config_file['model']['train']
    isTester = config_file['model']['test']
    num_of_topics = config_file['model']['num_of_topics']
    num_top_words = config_file['model']['num_top_words']
    num_top_documents = config_file['model']['num_top_documents']
    export_topics = config_file['model']['export_topics']
    topic_label = config_file['model']['topic_label']
    coherence_measure = config_file['model']['coherence_measure']
    replace_stop_words_list = config_file['model']['replace_stop_words_list']
    root_word_option = config_file['model']['root_word_option']
    include_words = config_file['model']['include_words']
    exclude_words = config_file['model']['exclude_words']
    remove_stop_words = config_file['model']['remove_stop_words']
    lower_case = config_file['model']['lower_case']
    word_form = config_file['model']['word_form']
    ngrams = (config_file['model']['ngrams_start'], config_file['model']['ngrams_end'])
    max_doc = config_file['model']['max_doc']
    min_doc = config_file['model']['min_doc']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../../..'))
    train_output_path = config_file['model']['train_output_path']
    if train_output_path is not None:
        train_output_path = os.path.join(curr_dir, train_output_path)
        if not os.path.exists(train_output_path):
            os.makedirs(train_output_path)
    test_output_path = config_file['model']['test_output_path']
    if test_output_path is not None:
        test_output_path = os.path.join(curr_dir, test_output_path)
        if not os.path.exists(test_output_path):
            os.makedirs(test_output_path)
    logging_path = os.path.join(curr_dir,config_file['model']['log_path'])
    # if not os.path.exists(logging_path):
    #     os.makedirs(logging_path)
    pickled_model = config_file['model']['pickled_model']
    if pickled_model is not None:
        pickled_model = os.path.join(curr_dir, pickled_model)

    data_df = pd.read_csv(os.path.join(home_folder,train_file))
    logger = open(os.path.join(curr_dir, logging_path), 'w')
    custom_print("Train dataset loaded",logger = logger)
    custom_print('Training model',logger = logger)
    seed_everything()
    custom_print('---------------------------------\n',logger = logger)

    if isTrainer and isTester:
        if isTester:
            train_dataset, test_dataset = create_datasets(data_df)
        else:
            train_dataset = Dataset(data_df)
            test_dataset = None
        train_test(train_dataset, train_output_path, num_of_topics, num_top_words, num_top_documents, coherence_measure, replace_stop_words_list, include_words, 
          exclude_words, root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc, test_dataset, test_output_path)
    
    elif isTester:
        test_dataset = Dataset(data_df)
        test(test_dataset, pickled_model, test_output_path, topic_label, num_top_documents, replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc)
        custom_print('Testing complete!',logger = logger)
    logger.close()

## yet to figure out why preprocess class is invoked many times at get_coherence function
## modify code to run gridsearchcv and create the graph



