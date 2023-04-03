#!/usr/bin/env python
# coding: utf-8

from flask import Flask, url_for, render_template, request
import torch
import torch.nn.functional as F
import syspend
import numpy as np
import os
from utility import parse_config
from sentimental_analysis.bert.train import BertClassifier
from sentimental_analysis.non_bert.train import predict_data
from sentimental_analysis.bert.dataset import single_data_loader


app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = os.path.join(os.getcwd(), 'flask_app_config.yml')
config_file = parse_config(config_path)
model_type = config_file['flask']['model_type']
model_name = config_file['flask']['model_name']
n_classes = int(config_file['flask']['num_classes'])
max_len = int(config_file['flask']['max_len'])
model_path = config_file['flask']['model_path']
THRESHOLD = int(config_file['flask']['THRESHOLD'])
MODEL =  BertClassifier(model_name, n_classes) #Add non-bert also
MODEL.model.to(device)

#Routes - Creating a simple route
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods = ['GET', 'POST'])
def predict(): #Send data from front-end to backend then to front end
    if request.method == "POST":
        # Get the form data
        tweet = request.form["tweet"]
        #[[0.3976914  0.60230863]] Means positive
        THRESHOLD = 0.8
        test_dataloader = single_data_loader(model_name, max_len, tweet)
        preds = MODEL.predict(test_dataloader, THRESHOLD)
        # Return the template with the form data
        return render_template("index.html", tweet = tweet, preds = preds)


if __name__ == '__main__':
    app.run(debug = True, use_reloader = True)




