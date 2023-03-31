#!/usr/bin/env python
# coding: utf-8

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

from utility import parse_config, seed_everything, custom_print,churn_eval_metrics
from preprocess_class import create_datasets
from dataset import full_bert_data_loader,preprocessing_for_bert, create_data_loader, full_create_data_loader

from model_base_class import BaseModel
#TODO: Finish BERT class framework(with Trainer Arguments so can customise other BERT models(BERT small, medium large or OTHERS))
#TODO: predictions
#TODO: See results or plot graph when adjusting threshold

# In[14]:

import torch.nn.functional as F

#Creating BERTClassifier class
class BertClassifier(BaseModel): #BaseModel,
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


    def train(self,learning_rate, epsilon,train_dataloader, val_dataloader = None, epochs = 2, evaluation=False, logger = None):
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
            churn_eval_metrics(all_labels, y_preds, logger)

            custom_print("-"*70,logger = logger)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.evaluate(self.model, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                custom_print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}",logger = logger)
                custom_print("-"*70, logger = logger)
            custom_print("\n", logger = logger)

        hours, seconds = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(seconds, 60)
        custom_print(f"Total Training Time:",logger = logger)
        custom_print("{:02d}:{:02d}:{:06.2f}".format(int(hours), int(minutes), seconds), logger = logger)
        custom_print("Training complete!",logger = logger)


    def evaluate(self,model, val_dataloader):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
         Parameters
        ----------
            model : transformers.models.bert.modeling_bert.BertForSequenceClassification
                BERT-based model
            val_dataloader : torch.utils.data.DataLoader
                DataLoader containing validation dataset
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        loss_fn = nn.CrossEntropyLoss()
        self.model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        all_logits = []
        all_labels = []

        # For each batch in our validation set...
        for batch in tqdm(val_dataloader, 'Validation'):
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits[0])
            all_labels.append(b_labels)

            # Compute loss
            loss = loss_fn(logits[0], b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            _, preds = torch.max(logits[0], dim=1)
            #preds = torch.argmax(logits[0], dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
        
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


        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    def predict(self):
        pass

#Trainer arguments - Removing soon
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
    train_on_full_data = eval(str(config_file['model']['train_on_full_data']))
    train_file = config_file['model']['data_folder']
    isTrainer = config_file['model']['trainer']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    model_path = os.path.join(curr_dir, config_file['model']['model_path'])
    logging_path = os.path.join(curr_dir,config_file['model']['log_path'])
        
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
    
    if isTrainer:
        trainer = train(model_name, train_dataset, val_dataset)
        custom_print('Training complete!',logger = logger)
        custom_print('Showing Training and Evaluation metrics....',logger = logger)
        #https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
        for obj in trainer.state.log_history:
            for key,value in obj.items():
                custom_print(f'{key}: {value}', logger = logger)
    else:
        custom_print('Loading data.....',logger = logger)
        sentimental_classifier = BertClassifier(model_name, n_classes)
        sentimental_classifier.model.to(device)
        custom_print('Model initialised!', logger = logger)
        if not train_on_full_data:
            train, test = train_test_split(data_df, test_size = 0.2, random_state = 4263)
            custom_print(f"dev size: {len(test)}",logger = logger)
            train_dataloader = create_data_loader(model_name, batch_size,max_len, train)
            custom_print('Train data loaded!', logger = logger)
            val_dataloader = create_data_loader(model_name, batch_size,max_len, test, predict_only=False)
            sentimental_classifier.train(learning_rate, epsilon,train_dataloader, val_dataloader,epochs =1, evaluation=True, logger = logger)
        else:
            full_train_data = full_create_data_loader(model_name, batch_size,max_len, data_df)
            custom_print('Full Data loaded!',logger = logger)
            sentimental_classifier.train(learning_rate, epsilon,full_train_data, epochs =1, logger = logger)
        custom_print('Saving model ...', logger = logger)
        torch.save({'model_state_dict':sentimental_classifier.model.state_dict()}, model_path)
    logger.close()
