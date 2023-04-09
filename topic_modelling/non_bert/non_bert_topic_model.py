import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
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
    """
    Non-bert based Model for topic modelling
    Users should initiate the preprocess_dataset method first before using train and/or predict functions to generate necessary files
    """
    def __init__(self, train_dataset = None, test_dataset = None, pickled_model = None, topic_label = None, pickled_vectorizer = None, feature_engineer = "bow", custom_print = True):
        """
        Constructs all the necessary attributes for the TopicModel Object.
        For model training only, users should specify train_dataset.
        For model prediction only, users should specify test_dataset, pickled_model, topic_label and pickled_vectorizer.
        For model training and prediction, users should specify train_dataset, test_dataset.
        feature_engineer and custom_print are optional attributes to be specified when necessary.

        Parameters
        ----------
        train_dataset: Dataset class
            Store dataset for model to be trained on. May be None if pickled_model is not None to train on test_dataset.
        test_dataset: Dataset class
            Store dataset for model to be predicted on. May be None if the class is only used to train a model without predicting on test data.
        model: Fitted estimator
            Store model to be trained on train_dataset and/or predicted on test_dataset. Initialised as None when trained_dataset is not None.
            Else, initialise this attribute with the file path of the pickled model.
        topic_label: list of str
            Store the trained topics manually generated by inspecting topic key words and sampled documents in each topic. Length of list = number of topics
            Initialised as empty string when train_dataset is None. Else, initialise with the file path of topic_key_words csv file to read in topic labels.
        training_vectorizer: Fitted vectorizer
            Store the tfidf/bow vectorizer that was fitted to the train dataset.
            Initialised as None when train_dataset is None. Else, initialise with the file path of the pickled vectorizer.
        feature_engineer: str
            Specify the feature engineering to be inputted to the training model.
            Valid inputs: "bow", "tfidf"
        custom_print: bool
            Set to True if logs are to be shown and written out in a log file. Otherwise, set to False.
        
        Attributes
        ----------
        training_tokens: pandas series
            Store the tokens for each text to be used as input into the training model.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.training_tokens = None
        self.model = pickle.load(open(pickled_model, 'rb')) if pickled_model is not None else None
        self.topic_label = pd.read_csv(topic_label).iloc[:,0].tolist() if topic_label is not None else []
        self.training_vectorizer = pickle.load(open(pickled_vectorizer, 'rb')) if pickled_vectorizer is not None else None
        self.feature_engineer = feature_engineer
        self.custom_print = custom_print

    def train(self, training_model, num_of_topics, num_top_words, num_top_documents, train_output_path):
        """
        Mainly to generate the fitted vectorizer, trained model, topic key words and topic labels for each text in trained_dataset.
        Fitted vectorizer and trained model are stored in pickled file.
        Topic key words are stored in a csv file with the first column as the Topic label and the subsequent columns as 
        top key words / phrases for each topic. Each row represents one topic.
        Topic labels will have columns "Text" and "Topic label".
        Sample training doc that are used for generating topic labels are also stored as csv file with columns "Text" and "Topic label"

        Parameters
        ----------
        training_model: str
            Specify the type of model that will be used to train topic modelling.
            valid inputs: "NMF", "LDA"
        num_of_topics: int
            Specify the number of topics to be generated by the training model.
        num_top_words: int
            Specify the number of highest ranking words for each topic to manually input its corresponding topic label.
        num_top_documents: int
            Specify the number of random documents sampled from each topic to manually input its corresponding topic label.
        train_output_path: str
            Specify the directory for generated files to be saved to.
        """
        self.topic_label = []
        if self.feature_engineer == "bow":
            vectorizer = self.train_dataset.bow
        
        elif self.feature_engineer == "tfidf":
            vectorizer = self.train_dataset.tfidf

        else:
            raise Exception('invalid feature engineer input')
        
        if self.custom_print:
            custom_print("------ Training model --------\n", logger = logger)

        if training_model == "LDA":
            training = LatentDirichletAllocation(n_components= num_of_topics, doc_topic_prior=0.5, topic_word_prior=0.5, random_state = 4263)

        elif training_model == "NMF":
            training = NMF(n_components = num_of_topics, init = 'nndsvd', random_state = 4263, solver = 'cd')   

        else:
            raise Exception("invalid model input!")
        
        
        fitted_model = training.fit(vectorizer[2])
        trained_model = fitted_model.transform(vectorizer[2])

        self.model = fitted_model
        self.training_vectorizer = vectorizer[1]

        vectorizer_name = "training_{}_vectorizer_{}.pk".format(self.feature_engineer, num_of_topics)
        self.dump_model(self.training_vectorizer, train_output_path, vectorizer_name)

        doc_topic_labels = pd.DataFrame(trained_model).idxmax(axis = 1)
        num_of_doc_per_topic = doc_topic_labels.value_counts().sort_index()

        if self.custom_print:
            custom_print("------ Distribution for number of documents in each topic --------\n", logger = logger)
            for i in range(len(num_of_doc_per_topic)):
                custom_print("Topic {}: {}".format(i, num_of_doc_per_topic[i]),logger = logger)

            custom_print("------{} documents trained------".format(str(num_of_doc_per_topic.sum())), logger = logger)

        topic_key_words = self.display_topics(training.components_, vectorizer[0].get_feature_names_out(), num_top_words, train_output_path, training_model)
        
        labelled_train_topics = pd.DataFrame({"Text": self.train_dataset.text, "Tokens": self.training_tokens, "Topic no": doc_topic_labels})

        if self.custom_print:
            custom_print("------ Generating key words and sample documents for each topic ------------\n", logger = logger)
        
        sample_doc_labels = pd.DataFrame()
        for i in range(num_of_topics):
            curr_topic_key_words = ", ".join(topic_key_words[i])

            curr_topic_doc = labelled_train_topics.loc[labelled_train_topics['Topic no'] == i, :]
            topic_samples = curr_topic_doc.sample(n = num_top_documents, random_state=4263)

            if self.custom_print:
                custom_print("\nTopic {}".format(i), logger = logger)
                custom_print(curr_topic_key_words, logger = logger)
            
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

        model_name = "training_{}_model_{}.pk".format(training_model, num_of_topics)
        self.dump_model(self.model, train_output_path, model_name)

        if self.custom_print:
            custom_print("-------- End of training for {} topics ----------".format(num_of_topics), logger = logger)

    def predict(self, test_output_path, root_word_option, remove_stop_words):
        """
        Generate test labels based on the trained model. Test labels are stored as a csv file with 2 columns "Text" and "Topic Label".

        Parameters
        ----------
        test_output_path: str
            Specify the directory for generated topic labels for each text in test_dataset to be saved to.
        root_word_option: int
            Specify if each word should be stemmed, lemmatized or remain as its original word before feature engineering
            valid inputs: 0 (use original word), 1 (use stemming on the words), 2 (use lemmatization on the words)
        remove_stop_words: bool
            True if stop words should be removed before feature engineering. False if all words are kept for feature engineering.

        Return
        ------
        labelled_test: pandas dataframe with columns Text and Topic label
        """
        test_input = self.get_input_text(self.test_dataset, root_word_option, remove_stop_words)
        test_data_fitted = self.training_vectorizer.transform(test_input.apply(lambda x: " ".join(x)))
        testing_labels = self.model.transform(test_data_fitted)

        assigned_topic = pd.DataFrame(testing_labels).idxmax(axis = 1)
        
        if self.custom_print:
            custom_print("-------- Exporting labelled topics ----------", logger = logger)
        labelled_test = pd.DataFrame({"Text": self.test_dataset.text, "Topic_no": assigned_topic})
        indexed_topic_label = pd.DataFrame({"Topic label":self.topic_label}).reset_index()

        labelled_test = labelled_test.merge(indexed_topic_label, how = "left", left_on = "Topic_no", right_on = "index")
        labelled_test = labelled_test.loc[:, ['Text', 'Topic label']]
        labelled_test_output_path = os.path.join(test_output_path, "full_labelled_test_dataset.csv")
        labelled_test.to_csv(labelled_test_output_path, index = False)
        
        if self.custom_print:
            custom_print("-------- End of predicting for {} topics ----------".format(len(self.topic_label)), logger = logger)
        return labelled_test

    def preprocess_dataset(self, replace_stop_words_list = None, include_words = ['taste', 'flavor', 'amazon', 'price', 'minute', 'time', 'year'],
                            exclude_words = ["not", "no", "least", "less", "last", "serious", "too", "again", "against", "already", "always", "cannot", "few", "must", "only", "though"],
                            root_word_option = 0, remove_stop_words = True, lower_case = True, 
                            word_form = None, ngrams = (1,1), max_doc = 1, min_doc = 1):
        """
        Preprocess and generate feature engineering for the training model

        Parameters
        ----------
        replace_stop_words_list: list of str
            Specify own list of stop words instead of using the default stop words list in Dataset class.
        include_words: list of str
            Specify additional words to the original stop words list.
            Default are just some words that can be considered to be added into the stop words list.
        exclude_words: list of str
            Specify words that should not be included in the stop words list.
            Default are just some words that can be considered to be removed from stop words list.
        root_word_option: int
            Specify if each word should be stemmed, lemmatized or remain as its original word before feature engineering.
            valid inputs: 0 (use original word), 1 (use stemming on the words), 2 (use lemmatization on the words)
        remove_stop_words: bool
            True if stop words should be removed before feature engineering. False if all words are kept for feature engineering.
        lower_case: bool
            True if all the words should be lowercased, False otherwise.
        word_form: list of str
            Specific parts of sentence to be included for topic model training and prediction.
            valid word forms: "verb", "noun", "adjective", "preposition", "adverb"
        ngrams: tuple of int
            Specify if the user wants unigram, bigram, trigrams or even mixture.
            (1,1) means only unigram, (1,2) means unigram and bigram
        max_doc: int
            in the range [0,1): if token appear in more than max_doc % of documents, the word is not considered in the bag of words.
            for any integer n >= 1: the token can only appear in at most n documents.
            If min_doc > 1, max_doc must be < 1.
            If preprocessing small test dataset, max_doc should be 1 to prevent filter until no words left in pool of words.
        min_doc: int
            in the range [0,1): if token appear in less than min_doc % of documents, the word is not considered in the bag of words
            for any integer n >= 1: the token must appear in at least n documents
            If preprocessing small test dataset, min_doc should be 1 to prevent filter until no words left in pool of words.
        """
        if self.train_dataset is not None:
            self.train_dataset.modify_stop_words_list(replace_stop_words_list, include_words, exclude_words)

            if self.feature_engineer == "bow":
                self.train_dataset.create_bow(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)
            
            elif self.feature_engineer == "tfidf":
                self.train_dataset.create_tfidf(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)

            self.training_tokens = self.get_input_text(self.train_dataset, root_word_option, remove_stop_words)
        if self.test_dataset is not None:
            self.test_dataset.modify_stop_words_list(replace_stop_words_list, include_words, exclude_words)

            if self.feature_engineer == "bow":
                self.test_dataset.create_bow(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)       
            
            elif self.feature_engineer == "tfidf":
                self.test_dataset.create_tfidf(root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)       

    def display_topics(self, trained_topics, feature_names, num_top_words, train_output_path, training_model):
        """
        Output the top key words / phrases for text to be allocated to each topics
        If training_model == "LDA", a visualisation of the semantic distance between each topic will be stored in the specified directory.
        Return topic key words in the form of list of list. Each element in the outer list represents one topic and each element in the
        nested list represents topic key word for that topic.

        Parameters
        ----------
        trained_topics: components_ attribute of the trained LDA or NMF model
        feature_names: .get_features_out() attribute of trained bow / tfidf vectorizer
        num_top_words: int
            Number of top topic key words to be displayed for topic labelling.
        train_output_path: str
            Path to store the LDA topic visualisation when applicable.
        training_model: str
            Indicate whether the NMF or LDA model is used to generate the topics.
            valid inputs: "LDA", "NMF"
        """
        topic_key_words = []
        
        for topic in trained_topics:
            topic_key_words.append([feature_names[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]])
        
        if training_model == "LDA":
            topic_vis_path = os.path.join(train_output_path, "topic_vis_{}.html".format(len(trained_topics)))
            panel = pyLDAvis.sklearn.prepare(self.model, self.training_vectorizer[2], self.train_dataset.bow[0], mds='tsne')            
            pyLDAvis.save_html(panel, topic_vis_path)
        return topic_key_words

    def dump_model(self, model, train_output_path, model_name):
        """
        Store the specified model in pickle file format in the specified directory

        Parameters
        ----------
        model: fitted estimator or vectorizer to be stored as pickle file
        train_output_path: str
            Directory for the pickled model and topic visualisation to be stored at.
        model_name: str
            Name of the pickle file
        """
        train_output_path = os.path.join(train_output_path, model_name)
        pickle.dump(model, open(train_output_path, 'wb'))

    def set_topic_labels(self, topic_no):
        """
        Take in manual input of Topic label and update the topic label attribute in the class

        Parameters
        ----------
        topic_no: int
            Topic number from the training model.
        """
        curr_topic_label = input("Label for Topic {}: ".format(topic_no))

        while len(curr_topic_label) == 0:
            curr_topic_label = input("Please reenter label for topic: ")
        self.topic_label.append(curr_topic_label)

    def get_input_text(self, dataset, root_word_option, remove_stop_words):
        """
        Output the tokens to be trained / predicted on

        Parameters
        ----------
        dataset: Dataset class
            Dataset that will be trained or predicted on.
        root_word_option: int
            Specify if each word should be stemmed, lemmatized or remain as its original word before feature engineering.
            valid inputs: 0 (use original word), 1 (use stemming on the words), 2 (use lemmatization on the words)
        remove_stop_words: bool
            True if stop words should be removed before feature engineering. False if all words are kept for feature engineering.
        """
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
        """
        Manual inspection to check if the texts are labelled to the right topic.
        Filter a sample of texts from test_dataset to generate a csv file (test_sample_labels) with manual inputs 1 or 0 
        to show if the texts are correctly labelled.
        Consist of 3 columns, "Topic label", "Sample text", "Prediction" (1 if right prediction, 0 if wrong prediction).

        Parameters
        ----------
        labelled_test: pandas dataframe
            List of text from test_dataset together with their corresponding topic labels
        num_top_documents: int
            Specify number of documents to be sampled from each topic to determine if the documents are allocated to the right topic label.
        test_output_path: str
            Directory for the sampled topic labels and manual inspection to be stored at.
        """

        if self.custom_print:
            custom_print("-------Evaluating training sample accuracy-------", logger = logger)
        topic_accuracy = []
        sample_labelling = pd.DataFrame()
        for topic in self.topic_label:
            curr_topic = labelled_test.loc[labelled_test['Topic label']== topic, "Text"]
            curr_topic_samples = curr_topic.sample(n=num_top_documents, random_state=4263)

            if self.custom_print:
                custom_print("\n--------Allocate next topic------------", logger = logger)
                custom_print("Current topic: {}".format(topic), logger = logger)

            correct_labels = []
            for sample in curr_topic_samples:

                if self.custom_print:
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
            if self.custom_print:
                custom_print("{} topic testing accuracy: {}".format(topic, str(accuracy)), logger = logger)
        
        sample_test_output_path = os.path.join(test_output_path, "test_sample_labels.csv")
        sample_labelling.to_csv(sample_test_output_path, index = False)
        average_topic_accuracy = sum(topic_accuracy) / len(topic_accuracy)

        if self.custom_print:
            custom_print("-------- Average topic accuracy: {} --------".format(str(average_topic_accuracy)), logger = logger) 

def train_test(train_dataset, feature_engineer, train_output_path, training_model, num_of_topics, num_top_words, num_top_documents, replace_stop_words_list, include_words, 
          exclude_words, root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc, test_dataset = None, test_output_path = None):
    """
    Generate the training model, training vectorizer and predict on the test dataset if necessary.
    """
    trainModel = TopicModel(train_dataset=train_dataset, test_dataset = test_dataset, feature_engineer=feature_engineer)
    trainModel.preprocess_dataset(replace_stop_words_list, include_words, exclude_words, root_word_option,
                                 remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)
    trainModel.train(training_model, num_of_topics, num_top_words, num_top_documents, train_output_path)

    if test_dataset is not None:
        test_labels = trainModel.predict(test_output_path, root_word_option, remove_stop_words)
        trainModel.churn_eval_metrics(test_labels, num_top_documents, test_output_path)    

def test(test_dataset, feature_engineer, pickled_model, pickled_vectorizer, test_output_path, topic_label, replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc):
    """
    Predict on the test dataset based on the pickled model and vectorizer.
    """
    testModel = TopicModel(test_dataset=test_dataset, pickled_model=pickled_model, topic_label = topic_label, pickled_vectorizer= pickled_vectorizer, feature_engineer=feature_engineer)
    testModel.preprocess_dataset(replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
             word_form, ngrams, max_doc, min_doc)
    testModel.predict(test_output_path=test_output_path, root_word_option= root_word_option, remove_stop_words = remove_stop_words)

if __name__ == "__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'non_bert_topic_modelling_config.yml')
    config_file = parse_config(config_path)
    model_choice = config_file['model_choice']
    training_model = config_file['model'][model_choice]['type_of_model']
    feature_engineer = config_file['model'][model_choice]['feature_engineer']
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
    pickled_vectorizer = config_file['model'][model_choice]['pickled_vectorizer']
    if pickled_vectorizer is not None:
        pickled_vectorizer = os.path.join(curr_dir, pickled_vectorizer)

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
        custom_print("------Preprocessing text data--------", logger = logger)
        train_test(train_dataset, feature_engineer, train_output_path, training_model, num_of_topics, num_top_words, num_top_documents, 
                   replace_stop_words_list, include_words, exclude_words, root_word_option, remove_stop_words, lower_case, word_form, 
                   ngrams, max_doc, min_doc, test_dataset, test_output_path)
    
    elif isTester:
        test_dataset = Dataset(data_df)
        custom_print("------Preprocessing text data--------", logger = logger)
        test(test_dataset, feature_engineer, pickled_model, pickled_vectorizer, test_output_path, topic_label, replace_stop_words_list, include_words, 
             exclude_words, root_word_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)
        custom_print('Testing complete!',logger = logger)
    logger.close()




