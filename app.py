"""File to display topic and sentiments of input text on webapp"""
#!/usr/bin/env python
# coding: utf-8

import os
from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
import pandas as pd
import torch
import torch.nn.functional as F
import syspend
import numpy as np
from preprocess_class import Dataset
from sentimental_analysis.bert.train import BertClassifier
from sentimental_analysis.bert.dataset import data_loader
from topic_modelling.non_bert.non_bert_topic_model import test
from utility import parse_config
import plotly.express as px
from plotly.subplots import make_subplots

def visualise(sentiment, time, text, topic_label):
    """
    Plot the recent topic trends and general trends

    Parameters
    ----------
    sentiment : Array
        Predicted Sentiments
    time : Array
        Time of text
    text : Array
        Text to predict
    topic_label : Array
        Labels of each text
    
    Returns
    -------
    String containing the div element of HTML
    """
    df = pd.DataFrame({"Sentiment":sentiment, "Time":time, "Text":text, "Topic_label":topic_label})
    df['Time'] = pd.to_datetime(df['Time'],dayfirst=True)
    df.loc[df['Sentiment'] == 'Positive', 'Sentiment'] = 1
    df.loc[df['Sentiment'] == 'positive', 'Sentiment'] = 1
    df.loc[df['Sentiment'] == 'Negative', 'Sentiment'] = 0
    df.loc[df['Sentiment'] == 'negative', 'Sentiment'] = 0
    print(df)
    group_df_1mo = df.groupby(['Topic_label',
                            pd.Grouper(freq="1M",key='Time')]).agg(Positive=('Sentiment', np.sum),
                                                                      Count = ('Sentiment', len),
                                                                      Avg=('Sentiment',
                                                                           lambda x:x.sum()/len(x)))
    group_df_1mo.reset_index(inplace=True)
    group_df_1mo = group_df_1mo.loc[group_df_1mo['Time'] >='2020-01-01',:]
    group_df_1mo_general = group_df_1mo.groupby('Time').agg(Positive=('Positive', np.sum),
                                                            Count = ('Count', np.sum))
    group_df_1mo_general['Avg'] = group_df_1mo_general['Positive'] / group_df_1mo_general['Count']
    pivot_df_1mo_avg = group_df_1mo.pivot(index='Time',columns='Topic_label', values='Avg')
    group_df =df.groupby(['Topic_label'], as_index=False, group_keys=False)\
        .agg(Positive=('Sentiment', np.sum), Count = ('Sentiment', len),
             Avg=('Sentiment', lambda x: x.sum()/len(x)))

    fig1 = px.line(pivot_df_1mo_avg, x=pivot_df_1mo_avg.index, y =pivot_df_1mo_avg.columns,
                   labels={"value":'Positive Review Percentage', 
                           "Time":'Date', "Topic_label": "Topic label"})

    fig2 = px.line(group_df_1mo_general, x=group_df_1mo_general.index,
                   y ='Avg', hover_data=['Positive'], labels={"Avg":'Positive Review Percentage',
                                                              "Time":'Date'})

    fig3 = px.bar(group_df, x='Topic_label', y ='Count', hover_data=['Avg'],
                labels={"Count":'Number of reviews', "Time":'Date', "Topic_label": "Topic label",
                        "Avg":"Positive Review Percentage"})
    figures = [
                fig1,
                fig2,
                fig3
        ]
    fig = make_subplots(rows=len(figures), cols=1,
            subplot_titles=('Change in positive Review Percentage from 2020 onwards',
            'Change in positive Review Percentage from 2020 onwards (All topics)',
            'Total reviews per topic')) 

    for i, figure in enumerate(figures):
        for trace in range(len(figure["data"])):
            fig.add_trace(figure["data"][trace], row=i+1, col=1)
    fig.update_layout(margin =dict(l=10,r=10,t=30,b=30), height = 800, width=1100)

    return fig.to_html(full_html = False)


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
feature_engineer = os.path.join(curr_dir,
topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['feature_engineer'])
logging_path = os.path.join(curr_dir,
topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['log_path'])
pickled_model = os.path.join(curr_dir,topic_modelling_dir,
topic_modelling_config_file['model'][model_choice]['pickled_model'])
pickled_vectorizer = os.path.join(curr_dir, 
topic_modelling_dir, topic_modelling_config_file['model'][model_choice]['pickled_vectorizer'])
test_output_path = os.path.join(curr_dir,
topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['test_output_path'])
num_top_documents = topic_modelling_config_file['model'][model_choice]['num_top_documents']
topic_label = os.path.join(curr_dir,
topic_modelling_dir,topic_modelling_config_file['model'][model_choice]['topic_label'])
replace_stop_words_list = topic_modelling_config_file['model'][model_choice]['replace_stop_words_list']
include_words = topic_modelling_config_file['model'][model_choice]['include_words']
exclude_words = topic_modelling_config_file['model'][model_choice]['exclude_words']
root_word_option = topic_modelling_config_file['model'][model_choice]['root_word_option']
remove_stop_words = topic_modelling_config_file['model'][model_choice]['remove_stop_words']
lower_case = topic_modelling_config_file['model'][model_choice]['lower_case']
word_form = topic_modelling_config_file['model'][model_choice]['word_form']
ngrams = (topic_modelling_config_file['model'][model_choice]['ngrams_start'], 
topic_modelling_config_file['model'][model_choice]['ngrams_end'])
max_doc = topic_modelling_config_file['model'][model_choice]['max_doc']
min_doc = topic_modelling_config_file['model'][model_choice]['min_doc']

# Setting up csv for initial historic trend plot
historic_data_path = config_file['flask']['historic_trend']['data_path']
df = pd.read_csv(historic_data_path)
df['Time'] = pd.to_datetime(df['Time'],dayfirst=True)
plot_html = visualise(df['Sentiment'], df['Time'],df['Text'],df['Topic_label'])


#Routes - Creating a simple route
@app.route('/')
def index():
    """
    Return the home page
    """
    return render_template("index.html", plot_html = plot_html)

@app.route('/predict', methods = ['GET', 'POST'])
def predict(): #Send data from front-end to backend then to front end
    """
    Return the Text and its predicted sentiment and topic for manual input
    """
    if request.method == "POST":
        text = request.form["review"]
        text_no_space = text.strip()
        if not text or not text_no_space:
            return render_template("index.html", js_script = "empty_text")
        test_dataloader = data_loader(model_name, max_len, text, 1)
        pred,prob = MODEL.predict(test_dataloader, THRESHOLD)
        one_line_df = pd.DataFrame({"Text": [text]})
        one_text_dataset = Dataset(one_line_df)
        logger = open(logging_path, 'w', encoding='utf-8')
        labelled_test_df =test(one_text_dataset, feature_engineer, replace_stop_words_list,
                               include_words, exclude_words, root_word_option, remove_stop_words,
                               lower_case, word_form, ngrams, max_doc, min_doc, test_output_path,
                               pickled_model, pickled_vectorizer, topic_label)
        topic = list(labelled_test_df["Topic_label"].apply(
            lambda x: " ".join(list(map(lambda y: y.capitalize(),x.split("_"))))))[0]
        logger.close()
        return render_template("index.html",
                                texts = [],
                                preds = [],
                                topics = [],
                                probs = [],
                                text =  text,
                                pred = pred[0],
                                prob = prob[0],
                                topic = topic,
                                plot_html = plot_html)

@app.route('/upload', methods=['POST'])
def upload():
    """
    Return a predictions for uploaded csv file with 3 columns: Text, Sentiments, Topic.
    Updates the sentiment and topic plots
    """
    file = request.files['file']
    filename = secure_filename(file.filename)
    preds = []
    if not filename.endswith('.csv'):
        return render_template("index.html", js_script = "wrong_file_type")
    df = pd.read_csv(file)
    date_time_col = df['Time'].apply(lambda x: x.split("/")[1])
    unique_months = list(set(list(date_time_col)))
    no_unique_months = len(unique_months)
    test_dataset = Dataset(df)
    logger = open(logging_path, 'w', encoding='utf-8')
    labelled_test_df =test(test_dataset, feature_engineer, replace_stop_words_list, include_words,
         exclude_words, root_word_option, remove_stop_words, lower_case, word_form,
         ngrams, max_doc, min_doc, test_output_path, pickled_model,
         pickled_vectorizer, topic_label)
    texts =  df['Text'].tolist()
    time =  df['Time'].tolist()
    topic_labels = list(labelled_test_df["Topic_label"].apply(
        lambda x: " ".join(list(map(lambda y: y.capitalize(),x.split("_"))))))
    topics = [ele for ele in topic_labels]
    x_preprocessed = np.array([text for text in texts])
    test_dataloader = data_loader(model_name, max_len, x_preprocessed, len(texts))
    preds, probs = MODEL.predict(test_dataloader, THRESHOLD)

    # Columns needed: Sentiment, Time, Text, Topic Label
    plot_html = visualise(preds, time, texts, topic_labels)
    logger.close()

    # Checking if request is sent outside of the website
    if 'local' in request.files:
        data_df = pd.DataFrame({"Text" : texts, "Time" : time,
                                "predicted_sentiment_probability" : probs,
                                "predicted_sentiment" : preds})
        response = make_response(data_df.to_csv())
        response.headers['Content-Disposition'] = 'attachment; filename=result.csv'
        response.headers['Content-type'] = 'text/csv'
        print('Finished making response')
        return response
    
    if no_unique_months == 1:
        return render_template("index.html", texts = texts, preds = preds, topics = topics,
                               probs = probs, plot_html = plot_html, js_script = "only_one_month")
    return render_template("index.html", texts = texts, preds = preds, topics = topics,
                           probs = probs, plot_html = plot_html)

if __name__ == '__main__':
    app.run(debug = True, use_reloader = True)
