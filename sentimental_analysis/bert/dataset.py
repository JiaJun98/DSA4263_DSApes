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

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def full_bert_data_loader(tokenizer_name,max_len, batch_size, shuffle, data):
    """Facilitate loading of full data; Overloaded function
    """
    #sentences = list(data["text"])
    #targets = list(data["target"])
    sentences = list(data["Text"])
    targets = list(data["Sentiment"].apply(lambda x: 0 if x == "positive" else 1 ))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
    X_train, X_val, y_train, y_val = train_test_split(sentences,targets, test_size = 0.15, random_state = 4263)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_len)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=max_len)
    train_dataset = BERTDataset(X_train_tokenized,y_train)
    val_dataset  = BERTDataset(X_val_tokenized,y_val)
    #train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=shuffle)
    #test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle=shuffle)
    return [train_dataset,val_dataset]

# In[3]:


def preprocessing_for_bert(data,tokenizer_name,max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
    # Create empty lists to store outputs
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
            #return_tensors='pt',           # Return PyTorch tensor
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
     """Facilitate loading of data
     
    @param tokenizer_name: Name of tokenizer, usually the name of the model being used
    @max_len: Integer maximum length of sentence allowed
    @sentences: List of data samples X
    @return: DataLoader object that generates data for input into model
    """
     inputs, masks = preprocessing_for_bert(sentences,tokenizer_name,max_len)
     data = TensorDataset(inputs, masks)
     sampler = SequentialSampler(data)
     return DataLoader(data, sampler=sampler, batch_size = num_setences)



def create_data_loader(tokenizer_name, batch_size,max_len, data_df, predict_only=False):
    """Facilitate loading of data

    @param tokenizer_name: Name of tokenizer, usually the name of the model being used
    @max_len: Integer maximum length of sentence allowed
    @batch_size: Integer batch size of samples loading into model
    @shuffle: Boolean to decide whether to shuffle samples while loading into model
    @sentences: List of data samples X
    @targets: List of target variables, if any y
    @predict_only: Boolean to check if the any targets should be used to load the dataset
    @return: DataLoader object that generates data for input into model
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


# In[5]:


def full_create_data_loader(tokenizer_name, batch_size,max_len,train_df):
    """Facilitate loading of full data; Overloaded function

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


