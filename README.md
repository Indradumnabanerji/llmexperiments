# LLM_Experiments
# Author: Indra Banerjee


Input Dataset:  SQUAD Dataset 

Data Split: Divide your dataset into training and testing subsets. The training set will be used to train the model, while the testing set will be used to evaluate its performance.

Model Selection: google/flan-t5-small

Model Training: Train the selected model on the training data. 

Model Evaluation: Evaluate with Rouge and Bleu scores.


Limitations: 
Smaller dataset for Campaign Recommender model as the model creates number of records (x50) embedding dimension. 

Usage:

    Squad dataset to be dowloaded in local
    
    brew install wget
    
    mkdir -p local_cache
    
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O train.json
    
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O dev.json

Run all the scripts:

    python3 invoker.py

Scripts: 

    invoker.py --> all functions invocation
    
    Promptcode.py --> Main code
    
    TaskSN.ipynb --> Notebook
