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

from utility import parse_config, seed_everything, custom_print,churn_eval_metrics, plot_roc_curve, plot_pr_curve
from preprocess_class import create_datasets
from sentimental_analysis.bert.dataset import full_bert_data_loader,preprocessing_for_bert, create_data_loader, full_create_data_loader

from model_base_class import BaseModel

class BertClassifier(BaseModel): 
    """
    Bert-base Model for sentimental analysis

    ...

    Attributes
    ----------
    model_name : str
        BERT model type
    num_classes : int
        number of output class
    freeze_bert : bool
         Set `False` to fine-tune the BERT model

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, model_name, num_classes,freeze_bert=False):
        """
        Constructs all the necessary attributes for the BertClassifier object.

        Parameters
        ----------
            model_name : str
                BERT model type
            num_classes : int
                number of output class
            freeze_bert : bool
                Set `False` to fine-tune the BERT model
        """
        #super(BertClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels = num_classes)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        # Freeze the BERT model
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        
        Parameters
        ----------
            input_ids : torch.Tensor
                an input tensor with shape (batch_size,max_length)
            attention_mask : torch.Tensor
                a tensor that hold attention mask information with shape (batch_size, max_length)
            logits : torch.Tensor
                an output tensor with shape (batch_size,num_labels)
        """
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        #TODO: Change SequenceClassifierOutput to Tensor; Some issues for forward
        return outputs


    def train(self,learning_rate, epsilon,train_dataloader, plot_path, val_dataloader = None, epochs = 2, evaluation=False, logger = None):
        """Train the BertClassifier model.
         Parameters
        ----------
            learning_rate : float
                learning rate for BERTClassifier Model
            epsilon : float
                epsilon value for BERTClassifier Model
            train_dataloader : torch.utils.data.DataLoader
                DataLoader containing training dataset
            val_dataloader : torch.utils.data.DataLoader, optional
                DataLoader containing validation dataset (default is None)
            epochs : int
                number of epochs to train the model (default is 2)
            evaluation : bool
                Set `True` to evaluate the model after each epoch (default is False)
            logger : _io.TextIOWrapper
                logger file to write the training process (default is None)
        """
        # TODO: Create the differrent optimisers
        optimizer = AdamW(self.model.parameters(),
                        lr=learning_rate,    
                        eps=epsilon    
                        )
        # Total number of training steps
        total_steps = len(train_dataloader) * epochs
        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)
        
        loss_fn = nn.CrossEntropyLoss()
        # Start training loop
        custom_print("Start training...\n",logger = logger)
        start_time = time.time()
        for epoch_i in tqdm(range(epochs)):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            custom_print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}",logger = logger)
            custom_print("-"*70,logger = logger)

            losses = []
            correct_predictions = 0
            all_logits = []
            all_labels = []
            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(tqdm(train_dataloader, 'Training')):
                batch_counts +=1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.model(input_ids = b_input_ids,attention_mask = b_attn_mask)
                all_logits.append(logits[0])
                all_labels.append(b_labels)
                _, preds = torch.max(logits[0], dim=1)
                loss = loss_fn(logits[0], b_labels)
                
                # accumulate loss
                correct_predictions += torch.sum(preds == b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()
                losses.append(loss.item())

                # compute gradient
                loss.backward()
                # clip gradient to prevent vanishing/exploding gradients. facilitates better training
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # gradient descent
                optimizer.step()
                # decrease learning rate
                scheduler.step()
                # reset gradient computations
                optimizer.zero_grad()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch
                    # Print training results
                    custom_print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}",logger = logger)

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            probs = F.softmax(all_logits, dim=1).detach().cpu().numpy()
            threshold = 0.5
            y_preds = np.where(probs[:, 1] > threshold, 1, 0)
            all_labels = [tensor.cpu().tolist() for tensor in all_labels]
            y_preds = y_preds.tolist()
            custom_print(f"Training metrics\n", logger = logger)
            churn_eval_metrics(all_labels, y_preds, logger)

            custom_print("-"*70,logger = logger)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                
                custom_print(f"Validation metrics\n", logger = logger)
                val_loss, val_accuracy = self.evaluate(val_dataloader, logger = logger)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                custom_print(f"\n{'Epoch':^7} | {'Batch':^7} | {'Avg Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}",logger = logger)
                custom_print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}",logger = logger)
                custom_print("-"*70, logger = logger)
            custom_print("\n", logger = logger)

        hours, seconds = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(seconds, 60)
        custom_print(f"Current training time:",logger = logger)
        custom_print("{:02d}:{:02d}:{:06.2f}".format(int(hours), int(minutes), seconds), logger = logger)
        return time_elapsed
        #custom_print("Training complete!",logger = logger)


    def evaluate(self, dataloader, test = False, plotting_dir = None, logger = None):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
         Parameters
        ----------
            model : transformers.models.bert.modeling_bert.BertForSequenceClassification
                BERT-based model
            dataloader : torch.utils.data.DataLoader
                DataLoader containing validation or test dataset
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        loss_fn = nn.CrossEntropyLoss()
        self.model.eval()

        # Tracking variables
        total_loss = []
        total_accuracy = []
       
        all_logits = []
        all_labels = []

        # For each batch in our validation/holdout set...
        for batch in tqdm(dataloader, 'Validation' if not test else 'Testing'):
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            all_logits.append(logits[0])
            all_labels.append(b_labels)

            # Compute loss
            loss = loss_fn(logits[0], b_labels)
            total_loss.append(loss.item())

            # Get the predictions
            _, preds = torch.max(logits[0], dim=1)
            #preds = torch.argmax(logits[0], dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            total_accuracy.append(accuracy)
        
        #Finding validation metrics    
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        #print(all_logits)
        probs = F.softmax(all_logits, dim=1).detach().cpu().numpy()
        #print(probs)
        threshold = 0.5
        y_preds = np.where(probs[:, 1] > threshold, 1, 0)
        #print(y_preds)
        all_labels = [tensor.cpu().tolist() for tensor in all_labels]
        y_preds = y_preds.tolist()
        churn_eval_metrics(all_labels, y_preds, logger)
        #print(probs[:, 1].tolist())
        
        if test:
            #plotting_dir = "plots/roberta_large_sentimental_non_trainer"
            plot_roc_curve(probs[:, 1].tolist(),all_labels,plotting_dir)
            plot_pr_curve(probs[:, 1].tolist(),all_labels,plotting_dir)
        
        # Compute the average accuracy and loss over the validation set.
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)

        return mean_loss, mean_accuracy

    def predict(self, single_dataloader,threshold):
        self.model.eval()
        all_logits = []
        for batch in single_dataloader:
            b_input_ids, b_attn_mask = tuple(t for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        probs = F.softmax(logits[0], dim=1).cpu().numpy() 
        print(probs)
        preds = np.where(probs[:, 0] > threshold, "Positive", "Negative") #SWITCH later
        return preds, probs[:, 0] 

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
    train_on_full_data = eval(str(config_file['model']['train_on_full_data']))
    train_file = config_file['model']['data_folder']
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
    
    custom_print('\nLoading data.....',logger = logger)
    sentimental_classifier = BertClassifier(model_name, n_classes)
    sentimental_classifier.model.to(device)
    custom_print('Model initialised!', logger = logger)
    
    if not train_on_full_data:
        totalTime = 0
        kf = KFold(n_splits=noOfKFolds, random_state=4263, shuffle=True)
        train, test = train_test_split(data_df, train_size = train_val_percentage, random_state = 4263)
        test_dataloader = create_data_loader(model_name, batch_size,max_len, test, predict_only=False)
        custom_print(f"\nTest size: {len(test)}",logger = logger)
        custom_print('Test data loaded!', logger = logger)
        for fold, (train_index, val_index) in enumerate(kf.split(train)):
            custom_print(f"\n Current folds cross validation: {fold}", logger = logger)
            train_df = train.iloc[train_index]
            val_df = train.iloc[val_index]
            train_dataloader = create_data_loader(model_name, batch_size,max_len, train_df)
            custom_print(f"\nTrain size: {len(train)}",logger = logger)
            custom_print('Train data loaded!', logger = logger)
            val_dataloader = create_data_loader(model_name, batch_size,max_len, val_df, predict_only=False)
            custom_print(f"\nVal size: {len(val_df)}",logger = logger)
            custom_print('Val data loaded!', logger = logger)
            totalTime += sentimental_classifier.train(learning_rate, epsilon,train_dataloader,plot_path, val_dataloader = val_dataloader,epochs =1, evaluation=True, logger = logger)
            custom_print("Training complete!",logger = logger)
        custom_print(f"\nTesting (Holdout) metrics", logger = logger)
        sentimental_classifier.evaluate(test_dataloader,test = True, plotting_dir = plot_path, logger = logger)
        hours, seconds = divmod(totalTime, 3600)
        minutes, seconds = divmod(seconds, 60)
        custom_print(f"Total Training Time:",logger = logger)
        custom_print("{:02d}:{:02d}:{:06.2f}".format(int(hours), int(minutes), seconds), logger = logger)
        print(f"Testing data: {test}")    
    else:
        full_train_data = full_create_data_loader(model_name, batch_size,max_len, data_df)
        custom_print('Full Data loaded!',logger = logger)
        sentimental_classifier.train(learning_rate, epsilon,full_train_data, plot_path, epochs =1, logger = logger)
    custom_print('Saving model ...', logger = logger)
    torch.save({'model_state_dict':sentimental_classifier.model.state_dict()}, model_path)
    logger.close()

