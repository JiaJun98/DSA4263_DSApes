#!/usr/bin/env python
# coding: utf-8

import os
import re
import string
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml

import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc, f1_score, confusion_matrix

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def parse_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

def custom_print(*msg, logger):
    """Prints a message and uses a global variable, logger, to save the message
    :param msg: can be a list of words or a word
    :returns: nothing
    """
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def churn_eval_metrics(Y_pred, Y_test, logger):
    model_acc = accuracy_score(Y_test, Y_pred)
    model_prec = precision_score(Y_test, Y_pred)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label = 1)
    model_auc = auc(fpr,tpr)

    model_f1score = f1_score(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/(tp+fn)

    custom_print("model_accuracy:", model_acc, logger = logger)
    custom_print("model_precision:",model_prec, logger = logger)
    custom_print("model_auc:", model_auc, logger = logger)
    custom_print("model_f1score:",model_f1score, logger = logger)
    custom_print("sensitivity:", sensitivity, logger = logger)
    custom_print("specificity:", specificity, logger = logger)