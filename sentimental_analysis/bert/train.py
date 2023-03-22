#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import re
import string
import random
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
import syspend

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from utility import parse_config, seed_everything, custom_print
from preprocess_class import create_datasets
from dataset import full_bert_data_loader,preprocessing_for_bert, create_data_loader, full_create_data_loader


# ##### DONE: Finish BERT class framework(with Trainer Arguments so can customise other BERT models(BERT small, medium large or OTHERS))
# ##### DONE: Add Bert Dataset class (Inherit or create yourself)
# ##### DONE: Run 1 iterations of BERT model and add the metrics measuring(This week lecture) in utility.py
# ##### DONE: Add the config variables from yml
# ##### PENDING: Remove some other data(like links)
# ##### PENDING: predictions
# ##### PENDING: See results or plot graph when adjusting threshold

# In[14]:


#Creating BERTClassifier class


# In[12]:


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


# In[10]:


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


# #### Driver class

# In[13]:


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
    train_file = config_file['model']['data_folder']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    model_path = os.path.join(curr_dir, config_file['model']['model_path'])
    logging_path = os.path.join(curr_dir,config_file['model']['log_path'])
        
    data_df = pd.read_csv(os.path.join(home_folder,train_file))
    logger = open(os.path.join(curr_dir, logging_path), 'w')
    custom_print(f'Device availiable: {device}', logger = logger)
    train_df, test_df = train_test_split(data_df, test_size = 0.2, random_state = 4263) #You are using slighly different for BERT
    train_dataset, val_dataset = full_bert_data_loader(model_name,max_len, batch_size, True, train_df)
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
    custom_print('Training complete!',logger = logger)
    
    custom_print('Showing Training and Evaluation metrics....',logger = logger)
    #https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
    for obj in trainer.state.log_history:
        for key,value in obj.items():
            custom_print(f'{key}: {value}')
    logger.close()
    


# In[17]:


get_ipython().system('jupyter nbconvert --to script train.ipynb')

