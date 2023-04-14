# DSA4263 Voice of Customers Project (Group DSApes)
This repository contains our solution on analysing the Voice of Customer (VOC) for the module DSA4263: Sense-making Case Analysis: Business and Commerce in NUS.
# Project description
Given a dataset containing the customer's feedback, we were tasked with finding:
1. Sentiment analysis (Is the feedback positive or negative)
2. Topic modelling (What topic is the feedback on)
Afterwards, using our trained machine learning model, we visualise the change in trends sentiment for the topics.
# About the Repository
The repository contains 2 main subfolders, sentiment analysis and topic modelling. And within both folders, there is a BERT and non-BERT related model solution. The main page contains files related to our preprocessing, docker and flask solutions.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements

```bash
pip install -r requirments.txt
```

# User Guide
With the root directory as the working directory:

```bash
docker build -t <image_name> . 
docker run -d -p 5000:5000 <image_name>
```
The website can be accessed by localhost:5000.

# Credits
Chan Zhen Hao, Benny

Choong Meng Zhun

Goh Jia Jun

Lim Zi Hong

Liu Ting Yen
