
# pip install gensim

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from re import sub
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
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
        self.stop_words_list = list(ENGLISH_STOP_WORDS.copy())
        self.bow = None #overwritten if want to include bigrams or trigrams
        self.tfidf = None
        self.doc2vec = None
        self.word2vec = None

    def create_bow(self, root_words_option = 0, remove_stop_words = True, lower_case = True, word_form = None, ngrams = (1,1), max_doc = 1, min_doc = 1):
        """
        Update self.bow with [vectorizer, bow_matrix]. vectorizer is the model to generate the bow output 
        while bow_matrix is the resultant of applying bow on the texts
        @param  root_words_option:  0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        @param  remove_stop_words:  True if stop words should not be included, False otherwise
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param  word_form:  To specify in a list format if only words of certain word forms are included for model building
                            Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        @param  ngrams: specify if the user wants unigram, bigram, trigrams or even mixture
                        (1,1) means only unigram, (1,2) means unigram and bigram
        @param  max_doc:    in the range [0,1): if token appear in more than max_doc % of documents,
                            the word is not considered in the bag of words.
                            for any integer n >= 1: the token can only appear in at most n documents.
        @param  min_doc:    in the range [0,1): if token appear in less than min_doc % of documents, 
                            the word is not considered in the bag of words
                            for any integer n >= 1: the token must appear in at least n documents
                            
        To get list of bag of words, use self.bow[0].get_feature_names_out()
        To get the array version to input into models, use self.bow[1].toarray()
        
        Code to create DataFrame with Tokens as columns and Documents as rows:
        pd.DataFrame(data=self.bow[1].toarray(), columns = self.bow[0].get_feature_names_out())
        """
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case, word_form)

        vectorizer = CountVectorizer(lowercase=lower_case, ngram_range = ngrams, max_df = max_doc, min_df = min_doc) 
        bow_matrix = vectorizer.fit_transform(final_text.apply(lambda x: " ".join(x)))
        self.bow = [vectorizer, bow_matrix]

    def create_tfidf(self, root_words_option = 0, remove_stop_words = True, lower_case = True, word_form = None, ngrams = (1,1), max_doc = 1, min_doc = 1):
        """
        Update self.tfidf with [vectorizer, tfidf_matrix]. vectorizer is the model to generate the tfidf output 
        while tfidf_matrix is the resultant of applying tfidf on the texts
        @param  root_words_option: 0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        @param  remove_stop_words:  True if stop words should not be included, False otherwise
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param  word_form:  To specify in a list format if only words of certain word forms are included for model building
                            Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        @param  ngrams: specify if the user wants unigram, bigram, trigrams or even mixture
                        (1,1) means only unigram, (1,2) means unigram and bigram
        @param  max_doc:    in the range [0,1): if token appear in more than max_doc % of documents,
                            the word is not considered in the bag of words.
                            for any integer n >= 1: the token can only appear in at most n documents.
        @param  min_doc:    in the range [0,1): if token appear in less than min_doc % of documents, 
                            the word is not considered in the bag of words
                            for any integer n >= 1: the token must appear in at least n documents

        To get list of tfidf, use self.tfidf[0].get_feature_names_out()
        To get the array version to input into models, use self.tfidf[1].toarray()
        
        Code to create DataFrame with Tokens as columns and Documents as rows:
        pd.DataFrame(data=self.tfidf[1].toarray(), columns = self.tfidf[0].get_feature_names_out())
        """
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case, word_form)
        
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range = ngrams, max_df = max_doc, min_df = min_doc) #upper case words throughout the feedback may mean the customer is angry, hence negative
        tfidf_output = vectorizer.fit_transform(final_text.apply(lambda x: " ".join(x)))
        self.tfidf = [vectorizer, tfidf_output]

    def create_doc2vec(self, root_words_option = 0, remove_stop_words = True, lower_case = True, word_form = None):
        """
        Update self.doc2vec to store vector representation of each text. Can use machine learning to see how to group
        the texts under the same topic
        @param  root_words_option:  0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        @param  remove_stop_words:  True if stop words should not be included, False otherwise
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param  word_form:  To specify in a list format if only words of certain word forms are included for model building
                            Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        @param  ngrams: specify if the user wants unigram, bigram, trigrams or even mixture
                        (1,1) means only unigram, (1,2) means unigram and bigram
        @param  max_doc:    in the range [0,1): if token appear in more than max_doc % of documents,
                            the word is not considered in the bag of words.
                            for any integer n >= 1: the token can only appear in at most n documents.
        @param  min_doc:    in the range [0,1): if token appear in less than min_doc % of documents, 
                            the word is not considered in the bag of words
                            for any integer n >= 1: the token must appear in at least n documents

        """
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case, word_form)

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

    def create_word2vec(self, root_words_option = 0, remove_stop_words = True, lower_case = True, word_form = None, ngrams = (1,1), max_doc = 1, min_doc = 1):
        """
        Update self.word2vec with a dictionary, with key as the word token and value as the corresponding number to the word
        @param  root_words_option:  0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        @param  remove_stop_words:  True if stop words should not be included, False otherwise
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param  word_form:  To specify in a list format if only words of certain word forms are included for model building
                            Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        @param  ngrams: specify if the user wants unigram, bigram, trigrams or even mixture
                        (1,1) means only unigram, (1,2) means unigram and bigram
        @param  max_doc:    in the range [0,1): if token appear in more than max_doc % of documents,
                            the word is not considered in the bag of words.
                            for any integer n >= 1: the token can only appear in at most n documents.
        @param  min_doc:    in the range [0,1): if token appear in less than min_doc % of documents, 
                            the word is not considered in the bag of words
                            for any integer n >= 1: the token must appear in at least n documents
        """
        
        final_text = self.input_text(root_words_option, remove_stop_words, lower_case, word_form)
        
        model = Word2Vec(final_text, vector_size=100, window=5, min_count=1, workers=4, sg=0, epochs=100)

        self.create_bow(root_words_option, remove_stop_words, lower_case, word_form, ngrams, max_doc, min_doc)
        tokenized_words = self.bow[0].get_feature_names_out()

        word2vec_mapping = {}
        for word in tokenized_words:
            if word in model.wv.key_to_index:
                word2vec_mapping[word] = model.wv.key_to_index[word]
        
        self.word2vec = word2vec_mapping

    def input_text(self, root_word_option = 0, remove_stop_words = True, lower_case = True, word_form = None):
        """
        Helper function to generate the required tokenised words to input for feature extraction / preprocessing
        @param  root_word_option:   0 - None, 1 - stem, 2 - lemmatize (based on self.root_words_options)
        @param  remove_stop_words:  True if stop words should not be included, False otherwise
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param  word_form:  To specify in a list format if only words of certain word forms are included for model building
                            Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        """
        if type(root_word_option) != int or root_word_option > 2 or root_word_option < 0:
            raise Exception("invalid root word option")
        
        elif self.root_words_options[root_word_option] is None:
            if remove_stop_words:
                self.removing_stop_words(lower_case, word_form)
                return self.tokenized_no_stop_words

            else:
                self.word_tokenizer(lower_case, word_form)
                return self.tokenized_words
        
        elif self.root_words_options[root_word_option] == "stem":
            self.stemming(remove_stop_words, lower_case, word_form)
            return self.stem
        
        elif self.root_words_options[root_word_option] == "lemmatize":
            self.lemmatization(remove_stop_words, lower_case, word_form)
            return self.lemmatize

    def modify_stop_words_list(self, replace_stop_words_list = None,
                               include_words = ['taste', 'flavor', 'amazon', 'price', 'minute', 'time', 'year'], 
                               exclude_words = ["not", "no", "least", "less", "last", "serious", "too", "again", "against", "already", "always", "cannot", "few", "must", "only", "though"]):
        """
        Function to modify the stop_words_list according to model requirements
        If expecting to use pos_tag to filter the tokens, users may prefer to specify your own stop words list instead of ENGLISH_STOP_WORDS list to prevent important words to be filtered out.
        @param  replace_stop_words_list: If user does not want to use ENGLISH_STOP_WORDS list, user may like to specify the list here
        @param  include_words: Usually used to add on to default stop words list (when replace_stop_words_list is None). Specify the list of additional words to be included if necessary. 
                               If no words is to be specified in this parameter, specify this parameter as []
        @param  exclude_words: Usually used to remove from default stop words list (when replace_stop_words_list is None). Specify the list of additional words to be included if necessary.
                               If no words is the be specified in this parameter, specify this parameter as []
        """
        if replace_stop_words_list is not None:
            self.stop_words_list = replace_stop_words_list
        
        for word in exclude_words:
            self.stop_words_list.remove(word)

        self.stop_words_list.extend(include_words)        
    
    def word_tokenizer(self, lower_case = True, word_form = None):
        """
        Update self.tokenized_words attribute with lists of tokenized words for each text. 
        Tokens such as numbers and punctuations are not considered as words, hence removed.
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param word_form: To specify in a list format if only words of certain word forms are included for model building
                           Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        """
        def extract_relevant_tokens(text, lower_case):
            """
            Convert "n't" token to "not" and include characters that include tokens that only contain character base on lower case criteria
            """
            output = []

            tokens = word_tokenize(text)
            for token in tokens:
                if token == "n't":
                    output.append("not")
                
                elif token.isalpha():
                    if lower_case:
                        output.append(token.lower())
                    
                    else:
                        output.append(token)
            return output
        
        def extract_relevant_word_forms(tokens, word_form):
            """
            Assign pos_tag to each word in the sentence, then filter the tokens based on required word_form
            """
            df = pd.DataFrame(pos_tag(tokens))
            word_forms_tag = {"noun": "N", "verb": "V", "adjective": "JJ", "preposition": "IN", "adverb": "RB"}

            tagged_word_form = []

            for tag in word_form:
                if tag not in word_forms_tag:
                    raise Exception("Specified word form is not found in the word_forms tagging list!")
                tagged_word_form.append(word_forms_tag[tag.lower()])

            return df.loc[df[1].str.startswith(tuple(tagged_word_form)), 0].tolist()

        self.tokenized_words = self.text.apply(lambda x: extract_relevant_tokens(x,lower_case))

        if word_form is not None:
            self.tokenized_words = self.tokenized_words.apply(lambda x:extract_relevant_word_forms(x, word_form))

    def sentence_tokenizer(self):
        """
        Update self.tokenized_sentence after splitting each text into individual sentence
        """
        self.tokenized_sentence =  self.text.apply(sent_tokenize)

    def removing_stop_words(self, lower_case = True, word_form = None):
        """
        Update self.tokenized_no_stop_words to remove any tokens in self.tokenized_words that exist in the list of stop words
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param word_form: To specify in a list format if only words of certain word forms are included for model building
                           Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        """
        self.word_tokenizer(lower_case, word_form)

        self.tokenized_no_stop_words = self.tokenized_words.apply(lambda x: [word for word in x if word.lower() not in self.stop_words_list])
 

    def stemming(self, remove_stop_words = True, lower_case = True, word_form = None):
        """
        Update self.stem after stemming all the tokens as specified
        @param  remove_stop_words:  True if stop words should not be included, False otherwise
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param word_form: To specify in a list format if only words of certain word forms are included for model building
                           Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        """
        words_to_stem = self.input_text(0, remove_stop_words, lower_case, word_form)

        ps = PorterStemmer()
        self.stem = words_to_stem.apply(lambda x: [ps.stem(word, to_lowercase = lower_case) for word in x])
    
    def lemmatization(self, remove_stop_words = True, lower_case = True, word_form = None):
        """
        Update self.lemmatize after lemmatizing all the tokens as specified
        @param  remove_stop_words:  True if stop words should not be included, False otherwise
        @param  lower_case: True if should apply lower case to all words before creating BOW, False otherwise
        @param word_form: To specify in a list format if only words of certain word forms are included for model building
                           Word forms include ["noun", "adverb", "adjective", "preposition", "verb"]
        """
        words_to_lemmatize = self.input_text(0, remove_stop_words, lower_case, word_form)
        
        wordnet_lemmatizer = WordNetLemmatizer()
        self.lemmatize = words_to_lemmatize.apply(lambda x: [wordnet_lemmatizer.lemmatize(word) for word in x])
    
        

def create_datasets(df):
    """
    Create standardised training and testing dataset after splitting the raw dataset based on Sentiment proportion
    @param  df: Raw dataframe with columns: Sentiments, Time, Text
    @return train and test dataset after converting these dataframe into Dataset Class
    """

    train, test = train_test_split(df, test_size = 0.2, random_state = 4263, stratify = df['Sentiment'])

    train.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    train_dataset = Dataset(train)
    test_dataset = Dataset(test)

    return [train_dataset, test_dataset]
