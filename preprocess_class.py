
# pip install gensim

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from re import sub
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer #sklearn have larger list

nltk.download('punkt')
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')


"""
This dataset class includes all the necessary preprocessing and feature extraction 
to be experimented for the different non-Bert models to predict sentiment analysis 
and topic modelling.
Users should call create_* functions directly to get their ideal preprocessed data.
Functions without "create" as the start are only helper functions for the main create_* functions.
"""
class Dataset:
    def __init__(self, dataset):
        """
        All the attributes will be stored in the form of dataframe
        """
        self.sentiments = dataset['Sentiment']
        self.date = pd.to_datetime(dataset['Time'])
        self.text = dataset['Text'].apply(lambda x: sub("<[^>]+>", " ", x).strip())
        self.root_words_options = [None, "stem", "lemmatize"]
        self.tokenized_words = None
        self.tokenized_no_stop_words = None
        self.tokenized_sentence = None
        self.stem = None
        self.lemmatize = None
        self.stop_words_list = list(ENGLISH_STOP_WORDS).copy()
        self.stop_words_list.remove("no")
        self.stop_words_list.remove("not")
        self.bow = None #overwritten if want to include bigrams or trigrams
        self.tfidf = None
        self.doc2vec = None
        self.word2vec = None

    def create_bow(self, root_words_option = 0, remove_stop_words = True, lower_case = True, ngrams = (1,1), max_doc = 1, min_doc = 1):
        """
        This is to create the bag of words based on the ngrams specified
        root_words: select a method to preprocess the words, namely stem and lemmatize
        stop_words: specify if stop words should be removed before using other preprocessing methods
                    like lemmatizing and stemming
        ngrams:     specify if the user wants unigram, bigram, trigrams or even mixture
                    (1,1) means only unigram, (1,2) means unigram and bigram
        min_doc:    usually in the range [0,1]. if word/phrase appear in less than min_doc % of documents, 
                    the word is not considered in the bag of words
        max_doc:    usually in the range [0,1]. if word/phrase appear in more than max_doc % of documents,
                    the word is not considered in the bag of words.
        To get list of bag of words, use self.bow[0].get_feature_names_out()
        To get the array version to input into models, use self.bow[1].toarray()
        
        Code to create DataFrame with Tokens as columns and Documents as rows:
        pd.DataFrame(data=self.bow[1].toarray(), columns = self.bow[0].get_feature_names_out())
        """
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case)

        vectorizer = CountVectorizer(lowercase=lower_case, ngram_range = ngrams, max_df = max_doc, min_df = min_doc) 
        bow_matrix = vectorizer.fit_transform(final_text.apply(lambda x: " ".join(x)))
        self.bow = [vectorizer, bow_matrix]

    def create_tfidf(self, root_words_option = 0, remove_stop_words = True, lower_case = True, ngrams = (1,1), max_doc = 1, min_doc = 1):
        """
        Possible feature extraction to be included in modelling for Sentiment analysis and Topic modelling
        root_words_option: input 0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        stop_words: specify if stop words should be removed before using other preprocessing methods
                    like lemmatizing and stemming
        ngrams:     specify if the user wants unigram, bigram, trigrams or even mixture
                    (1,1) means only unigram, (1,2) means unigram and bigram
        min_doc:    in the range [0,1): if token appear in less than min_doc % of documents, 
                    the word is not considered in the bag of words
                    for any integer n >= 1: the token must appear in at least n documents
        max_doc:    in the range [0,1): if token appear in more than max_doc % of documents,
                    the word is not considered in the bag of words.
                    for any integer n >= 1: the token can only appear in at most n documents.
        To get list of tfidf, use self.tfidf[0].get_feature_names_out()
        To get the array version to input into models, use self.tfidf[1].toarray()
        
        Code to create DataFrame with Tokens as columns and Documents as rows:
        pd.DataFrame(data=self.tfidf[1].toarray(), columns = self.tfidf[0].get_feature_names_out())
        """
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case)
        
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range = ngrams, max_df = max_doc, min_df = min_doc) #upper case words throughout the feedback may mean the customer is angry, hence negative
        tfidf_output = vectorizer.fit_transform(final_text.apply(lambda x: " ".join(x)))
        self.tfidf = [vectorizer, tfidf_output]

    def create_doc2vec(self, root_words_option = 0, remove_stop_words = True, lower_case = True):
        """
        Create vector representation of each text. Can use machine learning to see how to group
        the texts under the same topic
        root_words_option:  input 0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        stop_words:         specify if stop words should be removed before using other preprocessing methods
                            like lemmatizing and stemming

        """
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case)

        tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_text)]
        model = Doc2Vec(vector_size=100, window=5, workers=4, min_count=1, epochs=100)
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

        # Get the document vectors
        doc_vectors = []
        for i in range(len(tagged_docs)):
            doc_vector = model.infer_vector(tagged_docs[i].words)
            doc_vectors.append(doc_vector)
        self.doc2vec = doc_vectors  

    def create_word2vec(self, root_words_option = 0, remove_stop_words = True, lower_case = True, ngrams = (1,1), max_doc = 1, min_doc = 1):
        """
        Create vector representation of each word. Each of these words can be taken in as input
        for sentiment analysis and topic modelling
        root_words_option:  input 0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        stop_words:         specify if stop words should be removed before using other preprocessing methods
                            like lemmatizing and stemming
        min_doc:            in the range [0,1): if token appear in less than min_doc % of documents, 
                            the word is not considered in the bag of words
                            for any integer n >= 1: the token must appear in at least n documents
        max_doc:            in the range [0,1): if token appear in more than max_doc % of documents,
                            the word is not considered in the bag of words.
                            for any integer n >= 1: the token can only appear in at most n documents.
        """
        
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case)
        
        model = Word2Vec(final_text, vector_size=100, window=5, min_count=1, workers=4, sg=0, epochs=100)

        self.create_bow(root_words_option, remove_stop_words, lower_case, ngrams, max_doc, min_doc)
        tokenized_words = self.bow[0].get_feature_names_out()

        word2vec_mapping = {}
        for word in tokenized_words:
            if word in model.wv.key_to_index:
                word2vec_mapping[word] = model.wv.key_to_index[word]
        
        self.word2vec = word2vec_mapping

    def input_text(self, root_word_option, remove_stop_words, lower_case):
        """
        Helper function to generate the required tokenised words to input for feature extraction / preprocessing
        """
        if type(root_word_option) != int or root_word_option > 2 or root_word_option < 0:
            raise Exception("invalid root word option")
        
        elif self.root_words_options[root_word_option] is None:
            if remove_stop_words:
                self.removing_stop_words(lower_case)
                return self.tokenized_no_stop_words

            else:
                self.word_tokenizer(lower_case)
                return self.tokenized_words
        
        elif self.root_words_options[root_word_option] == "stem":
            self.stemming(remove_stop_words, lower_case)
            return self.stem
        
        elif self.root_words_options[root_word_option] == "lemmatize":
            self.lemmatization(remove_stop_words, lower_case)
            return self.lemmatize
    
    def word_tokenizer(self, lower_case = True):
        """
        Split each text into individual word token.
        Tokens such as numbers and punctuations are not considered as words, hence removed.
        """

        if lower_case:
            self.tokenized_words = self.text.apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha()])
        
        else:
            self.tokenized_words = self.text.apply(lambda x: [word for word in word_tokenize(x) if word.isalpha()])

    def sentence_tokenizer(self):
        """
        Split each text into individual sentence
        """
        self.tokenized_sentence =  self.text.apply(sent_tokenize)

    def removing_stop_words(self, lower_case = True):
        """
        Check through the lower casing of tokenized words to see if they exist in the list of stop words
        """
        self.word_tokenizer(lower_case)

        self.tokenized_no_stop_words = self.tokenized_words.apply(lambda x: [word for word in x if word.lower() not in self.stop_words_list])
 

    def stemming(self, remove_stop_words = True, lower_case = True):
        """
        One of the possible preprocessing steps to be used for feature extractions.
        May specify whether to remove stop words before carrying out stemming.
        """
        words_to_stem = self.input_text(0, remove_stop_words, lower_case)

        ps = PorterStemmer()
        self.stem = words_to_stem.apply(lambda x: [ps.stem(word, to_lowercase = lower_case) for word in x])
    
    def lemmatization(self, remove_stop_words = True, lower_case = True):
        """
        One of the possible preprocessing steps to be used for feature extractions.
        May specify whether to remove stop words before carrying out stemming.
        """
        words_to_lemmatize = self.input_text(0, remove_stop_words, lower_case)
        
        wordnet_lemmatizer = WordNetLemmatizer()
        self.lemmatize = words_to_lemmatize.apply(lambda x: [wordnet_lemmatizer.lemmatize(word) for word in x])
    
        

def create_datasets(df):
    """
    Use this function to create datasets so that all models are using the same training and testing dataset
    """

    train, test = train_test_split(df, test_size = 0.2, random_state = 4263, stratify = df['Sentiment'])

    train_dataset = Dataset(train)
    test_dataset = Dataset(test)

    return [train_dataset, test_dataset]


# Starting line import and split train and test data
# df = pd.read_csv('reviews.csv')
# train_data, test_data = create_datasets(df)