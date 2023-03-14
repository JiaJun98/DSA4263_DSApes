
# pip install gensim

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from re import sub
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet #trying different stopword list from packages nltk has a small list of stopwords
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
and topic modelling
"""
class Dataset:
    def __init__(self, dataset):
        """
        All the attributes will be stored in the form of dataframe
        """
        self.sentiments = dataset['Sentiment']
        self.date = pd.to_datetime(dataset['Time'])
        self.text = dataset['Text'].apply(lambda x: sub("<[^>]+>", " ", x).lower().strip())
        self.tokenized_words = None
        self.tokenized_no_stop_words = None
        self.tokenized_sentence = None
        self.stem = None
        self.lemmatize = None
        self.stop_words_list = list(ENGLISH_STOP_WORDS).copy()
        self.stop_words_list.remove("no")
        self.stop_words_list.remove("not")
        self.bow = None #overwrite if want to include bigrams or trigrams
        self.tfidf = None
        self.doc2vec = None
        self.word2vec = None
    
    def word_tokenizer(self):
        """
        Split each text into individual word token.
        Tokens such as numbers and punctuations are not considered as words, hence removed.
        """
        self.tokenized_words = self.text.apply(lambda x: [word for word in word_tokenize(x) if word.isalpha()])

    def sentence_tokenizer(self):
        """
        Split each text into individual sentence
        """
        self.tokenized_sentence =  self.text.apply(sent_tokenize)

    def stemming(self, remove_stop_words = True):
        """
        One of the possible preprocessing steps to be used for feature extractions.
        May specify whether to remove stop words before carrying out stemming
        """
        if remove_stop_words:
            if self.tokenized_no_stop_words is None:
                self.remove_stop_words()
            words_to_stem = self.tokenized_no_stop_words
        
        else:
            if self.tokenized_words is None:
                self.word_tokenizer()
            words_to_stem = self.tokenized_words

        ps = PorterStemmer()
        self.stem = words_to_stem.apply(lambda x: [ps.stem(word) for word in x])
    
    def lemmatization(self, remove_stop_words = True):
        """
        One of the possible preprocessing steps to be used for feature extractions.
        May specify whether to remove stop words before carrying out stemming.
        """
        if remove_stop_words:
            if self.tokenized_no_stop_words is None:
                self.remove_stop_words()
            words_to_lemmatize = self.tokenized_no_stop_words
        
        else:
            words_to_lemmatize = self.tokenized_words

        wordnet_lemmatizer = WordNetLemmatizer()
        self.lemmatize = words_to_lemmatize.apply(lambda x: [wordnet_lemmatizer.lemmatize(word) for word in x])
    
    def remove_stop_words(self):
        """
        Check through the lower casing of tokenized words to see if they exist in the list of stop words
        """
        if self.tokenized_words is None:
            self.word_tokenizer()
        self.tokenized_no_stop_words = self.tokenized_words.apply(lambda x: [word for word in x if word.lower() not in self.stop_words_list])

    def create_bow(self, root_words = None, stop_words = False, ngrams = (1,1), max_doc = 0.95, min_doc = 0.05):
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
        """
        if root_words == "stem":
            self.stemming(remove_stop_words = stop_words)
            final_text = self.stem
        
        elif root_words == "lemmatize":
            self.lemmatization(remove_stop_words=stop_words)
            final_text = self.lemmatize

        elif root_words is None:
            if stop_words:
                if self.tokenized_no_stop_words is None:
                    self.remove_stop_words()
                final_text = self.tokenized_no_stop_words
            else:
                if self.tokenized_words is None:
                    self.word_tokenizer()
                final_text = self.tokenized_words
        else:
            return "invalid root word"
        vectorizer = CountVectorizer(lowercase=False, ngram_range = ngrams, min_df = min_doc, max_df = max_doc) #upper case words throughout the feedback may mean the customer is angry, hence negative
        bow_output = vectorizer.fit_transform(final_text.apply(lambda x: " ".join(x)))
        self.bow = [vectorizer, bow_output]

    def create_tfidf(self, root_words = None, stop_words = False, ngrams = (1,1), max_doc = 0.95, min_doc = 0.05):
        """
        Possible feature extraction to be included in modelling for Sentiment analysis and Topic modelling
        root_words: select a method to preprocess the words, namely stem and lemmatize
        stop_words: specify if stop words should be removed before using other preprocessing methods
                    like lemmatizing and stemming
        ngrams:     specify if the user wants unigram, bigram, trigrams or even mixture
                    (1,1) means only unigram, (1,2) means unigram and bigram
        min_doc:    usually in the range [0,1]. if word/phrase appear in less than min_doc % of documents, 
                    the word is not considered in the bag of words
        max_doc:    usually in the range [0,1]. if word/phrase appear in more than max_doc % of documents,
                    the word is not considered in the bag of words.
        To get list of tfidf, use self.tfidf[0].get_feature_names_out()
        To get the array version to input into models, use self.tfidf[1].toarray()
        """
        if root_words == "stem":
            self.stemming(remove_stop_words = stop_words)
            final_text = self.stem
        
        elif root_words == "lemmatize":
            self.lemmatization(remove_stop_words=stop_words)
            final_text = self.lemmatize

        elif root_words is None:
            if stop_words:
                if self.tokenized_no_stop_words is None:
                    self.remove_stop_words()
                final_text = self.tokenized_no_stop_words
            else:
                if self.tokenized_words is None:
                    self.word_tokenizer()
                final_text = self.tokenized_words
        
        else:
            return "Invalid root word"
        
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range = ngrams, min_df = min_doc, max_df = max_doc) #upper case words throughout the feedback may mean the customer is angry, hence negative
        tfidf_output = vectorizer.fit_transform(final_text.apply(lambda x: " ".join(x)))
        self.tfidf = [vectorizer, tfidf_output]

    def create_doc2vec(self, root_words = None, stop_words = False):
        """
        Create vector representation of each text. Can use machine learning to see how to group
        the texts under the same topic
        root_words: select a method to preprocess the words, namely stem and lemmatize
        stop_words: specify if stop words should be removed before using other preprocessing methods
                    like lemmatizing and stemming

        """
        if root_words == "stem":
            self.stemming(remove_stop_words = stop_words)
            final_text = self.stem
        
        elif root_words == "lemmatize":
            self.lemmatization(remove_stop_words=stop_words)
            final_text = self.lemmatize

        elif root_words is None:
            if stop_words:
                if self.tokenized_no_stop_words is None:
                    self.remove_stop_words()
                final_text = self.tokenized_no_stop_words
            else:
                if self.tokenized_words is None:
                    self.word_tokenizer()
                final_text = self.tokenized_words
        
        else:
            return "invalid root word"
                

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

    def create_word2vec(self, root_words = None, stop_words = False):
        """
        Create vector representation of each word. Each of these words can be taken in as input
        for sentiment analysis and topic modelling
        root_words: select a method to preprocess the words, namely stem and lemmatize
        stop_words: specify if stop words should be removed before using other preprocessing methods
                    like lemmatizing and stemming
        """
        if root_words == "stem":
            self.stemming(remove_stop_words = stop_words)
            final_text = self.stem
        
        elif root_words == "lemmatize":
            self.lemmatization(remove_stop_words=stop_words)
            final_text = self.lemmatize

        elif root_words is None:
            if stop_words:
                if self.tokenized_no_stop_words is None:
                    self.remove_stop_words()
                final_text = self.tokenized_no_stop_words
            else:
                if self.tokenized_words is None:
                    self.word_tokenizer()
                final_text = self.tokenized_words
        
        else:
            return "invalid root word"
        model = Word2Vec(final_text, vector_size=100, window=5, min_count=1, workers=4, sg=0, epochs=100)

        self.create_bow(root_words, stop_words)
        vectorizer = self.bow[0]

        word2vec_mapping = {}
        for word in vectorizer.get_feature_names_out():
            if word in model.wv.key_to_index:
                word2vec_mapping[word] = model.wv.key_to_index[word]
        
        self.word2vec = word2vec_mapping
    

        

#use this function to create datasets so that all models are using the same training and testing dataset
def create_datasets(df):

    train, test = train_test_split(df, test_size = 0.2, random_state = 4263, stratify = df['Sentiment'])

    train_dataset = Dataset(train)
    test_dataset = Dataset(test)

    return [train_dataset, test_dataset]


# Starting line import and split train and test data
# df = pd.read_csv('reviews.csv')
# train_data, test_data = create_datasets(df)