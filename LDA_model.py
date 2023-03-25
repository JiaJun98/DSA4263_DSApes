import pandas as pd
import numpy as np
from preprocess_class import *
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from nltk.tag import pos_tag
 

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    topic_key_words = []
    topic_doc = []
    
    for topic_idx, topic in enumerate(H):
        # print ("Topic %d:" % (topic_idx))
        # print (" ".join([feature_names[i]
        #                 for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_key_words.append([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        curr_topic_doc = []
        for doc_index in top_doc_indices:
            curr_topic_doc.append(documents[doc_index])
        
        topic_doc.append(curr_topic_doc)
    
    return (topic_key_words, topic_doc)


def get_Cv(topic_words, texts):
    dictionary = corpora.Dictionary(texts)
    coherence_model = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='u_mass')
    coherence = coherence_model.get_coherence()
    return coherence

# temporary function to generate tokenized list with only nouns
def helper_function(x):
    df = pd.DataFrame(pos_tag(word_tokenize(x)))
    nouns = df.loc[df[1].str.startswith("N"), 0].tolist()
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_nouns = [wordnet_lemmatizer.lemmatize(word) for word in nouns]

    stop_words_list = list(ENGLISH_STOP_WORDS.copy())
    stop_words_list.extend(['taste', 'flavor', 'amazon', 'price', 'minute', 'time', 'year'])

    output = [word for word in lemmatized_nouns if word.lower() not in stop_words_list]
    return output

df = pd.read_csv("reviews.csv")
train_data, test_data = create_datasets(df)

final_text = train_data.text.apply(helper_function)

vectorizer = CountVectorizer(lowercase=True, ngram_range = (1,1), max_df = 0.95, min_df = 5) 
bow_matrix = vectorizer.fit_transform(final_text.apply(lambda x: " ".join(x)))

coherence_values = []
for i in range(2,31,2):
    print(i)
    lda = LatentDirichletAllocation(n_components=i, random_state = 0)
    testing_lda = lda.fit_transform(bow_matrix)
    topic_key_words, topic_doc = display_topics(lda.components_, testing_lda, vectorizer.get_feature_names_out(), train_data.text, 20, 10)
    coherence = get_Cv(topic_key_words, final_text)
    coherence_values.append(coherence)
    

# Show graph
import matplotlib.pyplot as plt
x = range(2, 31, 2)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

#selected 6 topics
lda = LatentDirichletAllocation(n_components=6, random_state = 0)
testing_lda = lda.fit_transform(bow_matrix)
topic_key_words, topic_doc = display_topics(lda.components_, testing_lda, vectorizer.get_feature_names_out(), train_data.text, 20, 10)
coherence = get_Cv(topic_key_words, final_text)
print(coherence)

pd.DataFrame(topic_doc).to_csv("topic_doc_6_topics.csv")
pd.DataFrame(topic_key_words).to_csv("topic_key_words_6_topics.csv")

"""
Things to discuss:
1. Should we get filter out the noun before fitting into the models to get the topics?
2. If 1 is true, then do I need to include pos_tag in the preprocessing_class?
3. If 1 is true, should we use u_mass for coherence score since the rest uses sliding window? 
Sliding window is not applicable for texts with only nouns
4. Do we have to modify the stop_words_list to include ['taste', 'flavor', 'amazon', 'price', 'minute', 'time', 'year'].
These words appear frequently throughout various topics when I try a few different runs of LDA.
"""

