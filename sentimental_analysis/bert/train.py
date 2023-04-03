#!/usr/bin/env python
# coding: utf-8

import os
import re
import string
import syspend
import random
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split,KFold

import matplotlib.pyplot as plt
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import sys
sys.path.append("../..")
from utility import parse_config, seed_everything, custom_print
from preprocess_class import create_datasets
from sentimental_analysis.bert.dataset import full_bert_data_loader,preprocessing_for_bert, create_data_loader, full_create_data_loader

from model_base_class import BaseModel
#TODO: Finish BERT class framework(with Trainer Arguments so can customise other BERT models(BERT small, medium large or OTHERS))
#TODO: predictions
#TODO: See results or plot graph when adjusting threshold


#Creating BERTClassifier class


# In[3]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred= predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# In[4]:


#Trainer arguments

# Define the training loop
def train(model_name, train_dataset, eval_dataset):
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir= model_path, #Model predictions and checkpoints
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=0.01,
        adam_epsilon = 1e-8, #Default
        logging_dir=logging_path, #Tensorboard logs
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        evaluation_strategy="epoch", #No Default
        logging_strategy = "epoch",
        save_strategy = "epoch"
    )

    # Define the optimizer and scheduler
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    #num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    #lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #optimizers=optimizer,#Default AdamW
        compute_metrics=compute_metrics
    )

    # Train the model
    training = trainer.train()

    return trainer

#Trainer utility functions- Removing soon
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred= predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

###### Driver class
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'bert_sentiment_config.yml')
    config_file = parse_config(config_path)
    model_name = config_file['model']['model_name']
    n_classes = int(config_file['model']['n_classes'])
    max_len = int(config_file['model']['max_len'])
    batch_size = int(config_file['model']['batch_size'])
    epochs = int(config_file['model']['epochs'])
    learning_rate = float(config_file['model']['learning_rate'])
    epsilon = float(config_file['model']['epsilon'])
    train_val_percentage = float(config_file['model']['train_val_percentage'])
    #test_size_percentage = float(config_file['model']['test_size_percentage'])
    train_on_full_data = eval(str(config_file['model']['train_on_full_data']))
    train_file = config_file['model']['data_folder']
    isTrainer = config_file['model']['trainer']
    noOfKFolds = config_file['model']['noOfKFolds']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    model_path = os.path.join(curr_dir, config_file['model']['model_path'])
    logging_path = os.path.join(curr_dir,config_file['model']['log_path'])
    plot_path =  os.path.join(curr_dir,config_file['model']['plot_path'])
    data_df = pd.read_csv(os.path.join(home_folder,train_file))
    logger = open(os.path.join(curr_dir, logging_path), 'w')
    custom_print(f'Device availiable: {device}', logger = logger)
    train_df, test_df = train_test_split(data_df, test_size = 0.2, random_state = 4263) #Trainer Arguments
    train_dataset, val_dataset = full_bert_data_loader(model_name,max_len, batch_size, True, train_df) #Trainer Arguments
    custom_print("Train_val dataset loaded",logger = logger)
    custom_print('Training model',logger = logger)
    seed_everything()
    custom_print('---------------------------------\n',logger = logger)

    custom_print("Hyperparameters:",logger = logger)
    custom_print(f"model name: {model_name}",logger = logger)
    custom_print(f"Number of epochs: {epochs}",logger = logger)
    custom_print(f"number of classes: {n_classes}",logger = logger)
    custom_print(f"max length: {max_len}",logger = logger)
    custom_print(f"batch size: {batch_size}",logger = logger)
    custom_print(f"learning rate: {learning_rate}",logger = logger)
    
    trainer = train(model_name, train_dataset, val_dataset)
    custom_print('Training complete!')
    
    custom_print('Showing Training and Evaluation metrics....')
    #https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
    for obj in trainer.state.log_history:
        for key,value in obj.items():
            custom_print(f'{key}: {value}')
    logger.close()

# %%
