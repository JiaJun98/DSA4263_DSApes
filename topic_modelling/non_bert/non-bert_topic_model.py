import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models import CoherenceModel
import gensim.corpora as corpora
# from nltk.tag import pos_tag
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append("../..")

from utility import parse_config, seed_everything, custom_print
from preprocess_class import Dataset, create_datasets
from model_base_class import BaseModel

class TopicModel(BaseModel):
    def __init__(self, train_dataset = None, test_dataset = None, pickled_model = None, topic_label = None, pickled_bow = None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.training_tokens = None
        self.model = pickle.load(open(pickled_model, 'rb')) if pickled_model is not None else None
        self.num_of_topics = None
        self.topic_label = pd.read_csv(topic_label).iloc[:,0].tolist() if topic_label is not None else []
        self.training_bow = pickle.load(open(pickled_bow, 'rb')) if pickled_bow is not None else None

    def train(self, training_model, num_of_topics, num_top_words, num_top_documents, train_output_path):
        self.num_of_topics = num_of_topics
        custom_print("------ Training model --------\n", logger = logger)
        if training_model == "LDA":
            training = LatentDirichletAllocation(n_components= num_of_topics, doc_topic_prior=0.5, topic_word_prior=0.5, random_state = 4263)

        elif training_model == "NMF":
            training = NMF(n_components = num_of_topics, init = 'nndsvd', random_state = 4263, solver = 'cd')   

        else:
            raise Exception("invalid model input!")
        fitted_model = training.fit(self.train_dataset.bow[2])
        trained_model = fitted_model.transform(self.train_dataset.bow[2])
        self.model = fitted_model
        self.training_bow = self.train_dataset.bow[1]

        training_bow_path = os.path.join(train_output_path,'training_bow_{}.pk'.format(self.num_of_topics))
        pickle.dump(self.train_dataset.bow[1], open(training_bow_path, 'wb'))

        doc_topic_labels = pd.DataFrame(trained_model).idxmax(axis = 1)
        num_of_doc_per_topic = doc_topic_labels.value_counts().sort_index()

        custom_print("------ Distribution for number of documents in each topic --------\n", logger = logger)
        for i in range(len(num_of_doc_per_topic)):
            custom_print("Topic {}: {}".format(i, num_of_doc_per_topic[i]),logger = logger)

        custom_print("------{} documents trained------".format(str(num_of_doc_per_topic.sum())), logger = logger)

        topic_key_words = self.display_topics(training.components_, self.train_dataset.bow[0].get_feature_names_out(), num_top_words, train_output_path, training_model)
        labelled_train_topics = pd.DataFrame({"Text": self.train_dataset.text, "Tokens": self.training_tokens, "Topic no": doc_topic_labels})
        custom_print("------ Generating key words and sample documents for each topic ------------\n", logger = logger)
        
        sample_doc_labels = pd.DataFrame()
        for i in range(self.num_of_topics):
            curr_topic_key_words = ", ".join(topic_key_words[i])
            custom_print("\nTopic {}".format(i), logger = logger)
            custom_print(curr_topic_key_words, logger = logger)

            curr_topic_doc = labelled_train_topics.loc[labelled_train_topics['Topic no'] == i, :]
            topic_samples = curr_topic_doc.sample(n = num_top_documents, random_state=4263)
            
            for text in topic_samples['Tokens']:
                custom_print("\n" + ", ".join(text), logger = logger)
            
            topic_samples = topic_samples.loc[:, ['Text', "Topic no"]]
            sample_doc_labels = pd.concat([sample_doc_labels, topic_samples])

            self.set_topic_labels(i)

        topic_labels = pd.DataFrame({"Topic label": self.topic_label}).reset_index()
        topic_labels.rename(columns = {"index":"Topic no"}, inplace = True)
        labelled_train_topics = labelled_train_topics.merge(topic_labels, how = "left", on = "Topic no").loc[:, ['Text', "Topic label"]]
        sample_doc_labels = sample_doc_labels.merge(topic_labels, how = "left", on = "Topic no").loc[:, ['Text', "Topic label"]]
        
        full_topic_path = os.path.join(train_output_path, "full_training_doc.csv")
        sampled_topic_path = os.path.join(train_output_path, "sample_training_doc.csv")

        labelled_train_topics.to_csv(full_topic_path, index = False)
        sample_doc_labels.to_csv(sampled_topic_path, index = False)

        topic_key_words_df = pd.DataFrame(topic_key_words, index = self.topic_label).reset_index()
        topic_key_words_df.rename(columns = {"index": 'Topic label'}, inplace = True)
        topic_key_words_path = os.path.join(train_output_path, "topic_key_words.csv")
        topic_key_words_df.to_csv(topic_key_words_path, index = False)

        custom_print("-------- End of training for {} topics ----------".format(num_of_topics), logger = logger)

    def predict(self, test_output_path, root_word_option):
        test_input = self.get_input_text(self.test_dataset, root_word_option)
        test_data_fitted = self.training_bow.transform(test_input.apply(lambda x: " ".join(x)))
        testing_labels = self.model.transform(test_data_fitted)

        assigned_topic = pd.DataFrame(testing_labels).idxmax(axis = 1)
        
        custom_print("-------- Exporting labelled topics ----------", logger = logger)
        labelled_test = pd.DataFrame({"Text": self.test_dataset.text, "Topic_no": assigned_topic})
        indexed_topic_label = pd.DataFrame({"Topic label":self.topic_label}).reset_index()

        labelled_test = labelled_test.merge(indexed_topic_label, how = "left", left_on = "Topic_no", right_on = "index")
        labelled_test = labelled_test.loc[:, ['Text', 'Topic label']]
        labelled_test_output_path = os.path.join(test_output_path, "full_labelled_test_dataset.csv")
        labelled_test.to_csv(labelled_test_output_path, index = False)
        
        custom_print("-------- End of predicting for {} topics ----------".format(num_of_topics), logger = logger)
        return labelled_test

    def preprocess_dataset(self, replace_stop_words_list = None, include_words = ['taste', 'flavor', 'amazon', 'price', 'minute', 'time', 'year'],
                            exclude_words = ["not", "no", "least", "less", "last", "serious", "too", "again", "against", "already", "always", "cannot", "few", "must", "only", "though"],
                            root_word_option = 0, remove_stop_words = True, lower_case = True, 
                            word_form = None, ngrams = (1,1), max_doc = 1, min_doc = 1):
        if self.train_dataset is not None:
            self.train_dataset.modify_stop_words_list(replace_stop_words_list, include_words, exclude_words)
            self.train_dataset.create_bow(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)

            self.training_tokens = self.get_input_text(self.train_dataset, root_word_option)
        if self.test_dataset is not None:
            self.test_dataset.modify_stop_words_list(replace_stop_words_list, include_words, exclude_words)
            self.test_dataset.create_bow(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)       

    def display_topics(self, trained_topics, feature_names, num_top_words, train_output_path, training_model):
        topic_key_words = []
        
        for topic in trained_topics:
            topic_key_words.append([feature_names[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]])
        
        if training_model == "LDA":
            topic_vis_path = os.path.join(train_output_path, "topic_vis_{}.html".format(self.num_of_topics))
            panel = pyLDAvis.sklearn.prepare(self.model, self.train_dataset.bow[2], self.train_dataset.bow[0], mds='tsne')
            pyLDAvis.save_html(panel, topic_vis_path)
        return topic_key_words

    def dump_model(self, train_output_path, training_model):
        train_output_path = os.path.join(train_output_path,'{}_model_{}.pk'.format(training_model, self.num_of_topics))
        pickle.dump(self.model, open(train_output_path, 'wb'))

    def set_topic_labels(self, topic_no):
        curr_topic_label = input("Label for Topic {}: ".format(topic_no))
        while len(curr_topic_label) == 0:
            curr_topic_label = input("Please reenter label for topic: ")
        self.topic_label.append(curr_topic_label)

    def get_input_text(self, dataset, root_word_option):
        if root_word_option == 0:
            if remove_stop_words:
                return dataset.tokenized_no_stop_words
            else:
                return dataset.tokenized_words
        
        elif root_word_option == 1:
            return dataset.stem
        
        elif root_word_option == 2:
            return dataset.lemmatize
        
        else:
            raise Exception("invalid root word option")

    def churn_eval_metrics(self, labelled_test, num_top_documents, test_output_path):
        custom_print("-------Evaluating training sample accuracy-------", logger = logger)
        topic_accuracy = []
        sample_labelling = pd.DataFrame()
        for topic in self.topic_label:
            curr_topic = labelled_test.loc[labelled_test['Topic label']== topic, "Text"]
            curr_topic_samples = curr_topic.sample(n=num_top_documents, random_state=4263)
            custom_print("\n--------Allocate next topic------------", logger = logger)
            custom_print("Current topic: {}".format(topic), logger = logger)

            correct_labels = []
            for sample in curr_topic_samples:
                custom_print("\n" + sample, logger = logger)
                curr_label = input("Correct label (1 or 0): ")
                
                while not curr_label.isdigit() or int(curr_label) not in [0,1]:
                    curr_label = input("Please reenter correct label (1 or 0): ")

                curr_label = int(curr_label)
                correct_labels.append(curr_label)

            sample_topic_labels = pd.DataFrame({"Sample text": curr_topic_samples, "Prediction": correct_labels})
            sample_topic_labels.insert(0, 'Topic label', topic)
            
            sample_labelling = pd.concat([sample_labelling, sample_topic_labels])

            accuracy = sum(correct_labels) / num_top_documents
            topic_accuracy.append(accuracy)
            custom_print("{} topic testing accuracy: {}".format(topic, str(accuracy)), logger = logger)
        
        sample_test_output_path = os.path.join(test_output_path, "test_labels.csv")
        sample_labelling.to_csv(sample_test_output_path, index = False)
        average_topic_accuracy = sum(topic_accuracy) / len(topic_accuracy)
        custom_print("-------- Average topic accuracy: {} --------".format(str(average_topic_accuracy)), logger = logger) 

def train_test(train_dataset, train_output_path, training_model, num_of_topics, num_top_words, num_top_documents, replace_stop_words_list, include_words, 
          exclude_words, root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc, test_dataset = None, test_output_path = None):
    trainModel = TopicModel(train_dataset=train_dataset, test_dataset = test_dataset)
    custom_print("------Preprocessing text data--------", logger = logger)
    trainModel.preprocess_dataset(replace_stop_words_list, include_words, exclude_words, root_word_option,
                                 remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)
    trainModel.train(training_model, num_of_topics, num_top_words, num_top_documents, train_output_path)
    trainModel.dump_model(train_output_path, training_model)

    if test_dataset is not None:
        test_labels = trainModel.predict(test_output_path, root_word_option)
        trainModel.churn_eval_metrics(test_labels, num_top_documents, test_output_path)    

def test(test_dataset, pickled_model, pickled_bow, test_output_path, topic_label, replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc):
    #max_doc = 1 and min_doc = 1 so that not all the key words will be filtered off, esp when the input size is small.
    testModel = TopicModel(test_dataset=test_dataset, pickled_model=pickled_model, topic_label = topic_label, pickled_bow= pickled_bow)
    custom_print("-------Preprocessing test data--------", logger = logger)
    testModel.preprocess_dataset(replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc)
    testModel.predict(test_output_path=test_output_path, root_word_option= root_word_option)

if __name__ == "__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'non-bert_topic_modelling_config.yml')
    config_file = parse_config(config_path)
    model_choice = config_file['model_choice']
    training_model = config_file['model'][model_choice]['type_of_model']
    train_file = config_file['model'][model_choice]['data_folder']
    isTrainer = config_file['model'][model_choice]['train']
    isTester = config_file['model'][model_choice]['test']
    num_of_topics = config_file['model'][model_choice]['num_of_topics']
    num_top_words = config_file['model'][model_choice]['num_top_words']
    num_top_documents = config_file['model'][model_choice]['num_top_documents']
    topic_label = config_file['model'][model_choice]['topic_label']
    replace_stop_words_list = config_file['model'][model_choice]['replace_stop_words_list']
    root_word_option = config_file['model'][model_choice]['root_word_option']
    include_words = config_file['model'][model_choice]['include_words']
    exclude_words = config_file['model'][model_choice]['exclude_words']
    remove_stop_words = config_file['model'][model_choice]['remove_stop_words']
    lower_case = config_file['model'][model_choice]['lower_case']
    word_form = config_file['model'][model_choice]['word_form']
    ngrams = (config_file['model'][model_choice]['ngrams_start'], config_file['model'][model_choice]['ngrams_end'])
    max_doc = config_file['model'][model_choice]['max_doc']
    min_doc = config_file['model'][model_choice]['min_doc']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    train_output_path = config_file['model'][model_choice]['train_output_path']
    if train_output_path is not None:
        train_output_path = os.path.join(curr_dir, train_output_path)
        if not os.path.exists(train_output_path):
            os.makedirs(train_output_path)
    test_output_path = config_file['model'][model_choice]['test_output_path']
    if test_output_path is not None:
        test_output_path = os.path.join(curr_dir, test_output_path)
        if not os.path.exists(test_output_path):
            os.makedirs(test_output_path)
    logging_path = config_file['model'][model_choice]['log_path']
    if not os.path.exists("/".join(logging_path.split("/")[:-1])):
        os.makedirs("/".join(logging_path.split("/")[:-1]))
    logging_path = os.path.join(curr_dir,config_file['model'][model_choice]['log_path'])
    pickled_model = config_file['model'][model_choice]['pickled_model']
    if pickled_model is not None:
        pickled_model = os.path.join(curr_dir, pickled_model)
    pickled_bow = config_file['model'][model_choice]['pickled_bow']
    if pickled_bow is not None:
        pickled_bow = os.path.join(curr_dir, pickled_bow)

    data_df = pd.read_csv(os.path.join(home_folder,train_file))
    logger = open(logging_path, 'w')
    custom_print("Train dataset loaded",logger = logger)
    seed_everything()
    custom_print('---------------------------------\n',logger = logger)

    if isTrainer:
        if isTester:
            train_dataset, test_dataset = create_datasets(data_df)
        else:
            train_dataset = Dataset(data_df)
            test_dataset = None
        train_test(train_dataset, train_output_path, training_model, num_of_topics, num_top_words, num_top_documents, replace_stop_words_list, include_words, 
          exclude_words, root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc, test_dataset, test_output_path)
    
    elif isTester:
        test_dataset = Dataset(data_df)
        test(test_dataset, pickled_model, pickled_bow, test_output_path, topic_label, replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc)
        custom_print('Testing complete!',logger = logger)
    logger.close()




