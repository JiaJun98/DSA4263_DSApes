#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset,DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import syspend

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocessing_for_bert(data,tokenizer_name,max_len):
    """
    Perform required preprocessing steps for pretrained BERT.

    Parameters
    ----------
        tokenizer_name: str
            Name of tokenizer, usually the name of the model being used
        max_len : int
            Integer maximum length of sentence allowed
    
    Returns
    -------
        input_ids: torch.tensor(List)
            Torch Tensor list representing the token IDs of the sentence. 

        attention_masks: torch.tensor(List)
            A list of Torch tensor of size (max_len,) containing 1s and 0s indicating which tokens should be attended to (1) and which ones should be ignored (0).
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True,     # Return attention mask
            truncation=True
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# In[4]:
def data_loader(tokenizer_name,max_len, sentences, num_setences):
     """Facilitate loading of single data into Transformer model
    Parameters
    ----------
        tokenizer_name: str
            Name of tokenizer, usually the name of the model being used
        max_len : int
            Integer maximum length of sentence allowed
        sentences: List(int)
            List of data samples X
        num_setences: int
            Number of sentences to be loaded into model at a time
    Returns
    -------
        torch.utils.data.DataLoader
            generates data for input into model
    """
     inputs, masks = preprocessing_for_bert(sentences,tokenizer_name,max_len)
     data = TensorDataset(inputs, masks)
     sampler = SequentialSampler(data)
     return DataLoader(data, sampler=sampler, batch_size = num_setences)



def create_data_loader(tokenizer_name, batch_size,max_len, data_df, predict_only=False):
    """Facilitate loading of dataframe into model
    Parameters
    ----------
        tokenizer_name: str
            Name of tokenizer, usually the name of the model being used
        batch_size: int
            Integer batch size of samples loading into model
        max_len : int
            Integer maximum length of sentence allowed
        data_df: pandas DataFrame
            DataFrame containing text and optional labels
        predict_only: bool ,optional
            Boolean to check if the any targets should be used to load the dataset
    Returns
    -------
        torch.utils.data.DataLoader
            generates data for input into model
    """
    data_df["Sentiment"] = data_df["Sentiment"].apply(lambda x: 0 if x == "positive" else 1 )
    X = data_df.Text.values
    y = data_df.Sentiment.values
    X_preprocessed = np.array([text for text in X])
    y_labels = torch.tensor(y)
    inputs, masks = preprocessing_for_bert(X_preprocessed,tokenizer_name,max_len)
    if not predict_only:
        labels = torch.tensor(y_labels)
        data = TensorDataset(inputs, masks, labels)
    else:
        data = TensorDataset(inputs, masks)
    sampler = RandomSampler(data) if not predict_only else SequentialSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)


def full_create_data_loader(tokenizer_name, batch_size,max_len,train_df):
    """Facilitate loading of dataframe into model
    Parameters
    ----------
        tokenizer_name: str
            Name of tokenizer, usually the name of the model being used
        batch_size: int
            Integer batch size of samples loading into model
        max_len : int
            Integer maximum length of sentence allowed
        train_df: pandas DataFrame
            Full training DataFrame containing text and labels
    Returns
    -------
        torch.utils.data.DataLoader
            generates full data for input into model
    """
    train_df["Sentiment"] = train_df["Sentiment"].apply(lambda x: 0 if x == "positive" else 1 )
    X = train_df.Text.values
    y = train_df.Sentiment.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)
    X_train_preprocessed = np.array([text for text in X_train])
    X_val_preprocessed = np.array([text for text in X_val])
    train_inputs, train_masks = preprocessing_for_bert(X_train_preprocessed,tokenizer_name,max_len)
    train_labels = torch.tensor(y_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val_preprocessed,tokenizer_name,max_len)
    val_labels = torch.tensor(y_val)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
    full_train_sampler = RandomSampler(full_train_data)
    return DataLoader(full_train_data, sampler=full_train_sampler, batch_size=batch_size)


