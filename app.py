#!/usr/bin/env python
# coding: utf-8

from flask import Flask, url_for, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import torch
import torch.nn.functional as F
import syspend
import numpy as np
import os
from utility import parse_config
from preprocess_class import Dataset
from sentimental_analysis.bert.train import BertClassifier
from topic_modelling.non_bert.non_bert_topic_model import test
from sentimental_analysis.bert.dataset import data_loader


app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = os.path.join(os.getcwd(), 'flask_app_config.yml')
config_file = parse_config(config_path)
model_name = config_file['flask']['sentimental_analysis']['model_name']
n_classes = int(config_file['flask']['sentimental_analysis']['num_classes'])
max_len = int(config_file['flask']['sentimental_analysis']['max_len'])
sentimental_model_path = config_file['flask']['sentimental_analysis']['model_path']
THRESHOLD = float(config_file['flask']['sentimental_analysis']['THRESHOLD'])

#Setting up BERTClassifier for sentimental analysis
MODEL =  BertClassifier(model_name, n_classes)
checkpoint = torch.load(sentimental_model_path, map_location=device)
MODEL.model.load_state_dict(checkpoint['model_state_dict'])
MODEL.model.to(device)

#Setting up non-bert Topic for Topic modelling
curr_dir = os.getcwd()
topic_modelling_config_path = config_file['flask']['topic_modelling']['config_path']
topic_modelling_config_path = os.path.join(os.getcwd(), topic_modelling_config_path)
topic_modelling_config_file = parse_config(topic_modelling_config_path)
model_choice = topic_modelling_config_file['model_choice']
num_of_topics = topic_modelling_config_file['model'][model_choice]['num_of_topics']
topic_modelling_dir = config_file['flask']['topic_modelling']['topic_modelling_dir']
logging_path = os.path.join(curr_dir,topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['log_path'])
pickled_model = os.path.join(curr_dir,topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['pickled_model'])
pickled_bow = os.path.join(curr_dir, topic_modelling_dir, topic_modelling_config_file['model'][model_choice]['pickled_bow'])
test_output_path = os.path.join(curr_dir,topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['test_output_path'])
num_top_documents = topic_modelling_config_file['model'][model_choice]['num_top_documents']
topic_label = os.path.join(curr_dir,topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['topic_label'])
replace_stop_words_list = topic_modelling_config_file['model'][model_choice]['replace_stop_words_list']
include_words = topic_modelling_config_file['model'][model_choice]['include_words']
exclude_words = topic_modelling_config_file['model'][model_choice]['exclude_words']
root_word_option = topic_modelling_config_file['model'][model_choice]['root_word_option']
remove_stop_words = topic_modelling_config_file['model'][model_choice]['remove_stop_words']
lower_case = topic_modelling_config_file['model'][model_choice]['lower_case']
word_form = topic_modelling_config_file['model'][model_choice]['word_form']
ngrams = (topic_modelling_config_file['model'][model_choice]['ngrams_start'], topic_modelling_config_file['model'][model_choice]['ngrams_end'])
max_doc = topic_modelling_config_file['model'][model_choice]['max_doc']
min_doc = topic_modelling_config_file['model'][model_choice]['min_doc']



#Routes - Creating a simple route
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['GET', 'POST'])
def predict(): #Send data from front-end to backend then to front end
    if request.method == "POST":
        # Get the form data
        texts = request.form["review"]
        test_dataloader = data_loader(model_name, max_len, texts, 1)
        preds,probs = MODEL.predict(test_dataloader, THRESHOLD)
        print(probs)
        # Return the template with the form data
        topics = "Coffee" #Place holder
        return render_template("index.html", texts = texts, preds = preds, topics = topics, probs = probs)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    tweets = []
    preds = []
    #topics = [] Uncomment when done
    if not filename.endswith('.csv'): #Place holder. Havent set the thing yet
        print("ONLY CSV is allowed!")
    df = pd.read_csv(file)
    test_dataset = Dataset(df)
    logger = open(logging_path, 'w')
    labelled_test_df = test(test_dataset, pickled_model, pickled_bow, test_output_path, 
        topic_label, num_top_documents, replace_stop_words_list, 
        include_words, exclude_words, root_word_option, remove_stop_words, lower_case,
                word_form, ngrams, max_doc, min_doc, logger, num_of_topics)
    
    #Place holder for him his reviews_csv have "Text" "Time" slightly hardcoded
    texts =  df['Text'].tolist()[1:]
    time =  df['Time'].tolist()[1:]
    print(labelled_test_df)
    topic_labels = list(labelled_test_df["Topic label"].apply(lambda x: " ".join(list(map(lambda y: y.capitalize(),x.split("_"))))))
    
    X_preprocessed = np.array([text for text in texts])
    test_dataloader = data_loader(model_name, max_len, X_preprocessed, len(texts))
    preds, probs = MODEL.predict(test_dataloader, THRESHOLD)
    print(f"Current predictions: {preds}")
    print(f"Current predictions: {probs}")
    # Do something with the uploaded file
    topics = [ele for ele in topic_labels ]
    return render_template("index.html", texts = texts, preds = preds, topics = topics, probs = probs)


if __name__ == '__main__':
    app.run(debug = True, use_reloader = True)