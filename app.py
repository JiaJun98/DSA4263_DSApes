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
from sentimental_analysis.bert.train import BertClassifier
#from sentimental_analysis.non_bert.train import predict_data
from sentimental_analysis.bert.dataset import data_loader


app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = os.path.join(os.getcwd(), 'flask_app_config.yml')
config_file = parse_config(config_path)
model_type = config_file['flask']['model_type']
model_name = config_file['flask']['model_name']
n_classes = int(config_file['flask']['num_classes'])
max_len = int(config_file['flask']['max_len'])
model_path = config_file['flask']['model_path']
THRESHOLD = float(config_file['flask']['THRESHOLD'])
MODEL =  BertClassifier(model_name, n_classes)
checkpoint = torch.load(model_path, map_location=device)
MODEL.model.load_state_dict(checkpoint['model_state_dict'])
MODEL.model.to(device)

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
    df = pd.read_csv(file, header=None)
    #Place holder for him his reviews_csv have "Text" "Time" slightly hardcoded
    texts =  df[0].tolist()[1:]
    #time =  df[1].tolist()[1:] Real code DONT remove
    predicted_topics = df[1].tolist()[1:]
    X_preprocessed = np.array([text for text in texts])
    test_dataloader = data_loader(model_name, max_len, X_preprocessed, len(texts))
    preds, probs = MODEL.predict(test_dataloader, THRESHOLD)
    print(f"Current predictions: {preds}")
    print(f"Current predictions: {probs}")
    # Do something with the uploaded file
    topics = [ele for ele in predicted_topics ]
    return render_template("index.html", texts = texts, preds = preds, topics = topics, probs = probs)


if __name__ == '__main__':
    app.run(debug = True, use_reloader = True)