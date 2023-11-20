import json
import os
import time
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import evaluate
import pandas as pd
import numpy as np
import pandas as pd
import tiktoken
import seaborn as sns
from tenacity import retry, wait_exponential
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
import evaluate
from Promptcode import PromptMachine
warnings.filterwarnings('ignore')
tqdm.pandas()

def log(text):
    print(text)

def main():

    # Summary generation
    huggingface_dataset_name = "knkarthick/dialogsum"
    model_name='google/flan-t5-small'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(huggingface_dataset_name)

    # In a production stage variables such as index will be supplied as config param
    index = 200
    PromptMachine.summary_generator(index, dataset, tokenizer, original_model)

    # Question answer model
    query = " Who is the best footballer of all times"
    output = PromptMachine.question_answer(query, tokenizer, original_model)
    query = "What is a good way to procrastinate"
    output = PromptMachine.question_answer(query,tokenizer, original_model)
    dash_line = '-'.join('' for x in range(100))
    log(dash_line)
    log(dash_line)
    log(f'Generated Answer:\n{output}')

    # Print the model parameters
    log(dash_line)
    log(dash_line)
    log(f'Model Parameters')
    log(PromptMachine.print_number_of_trainable_model_parameters(original_model))
    log(dash_line)
    log(f'Model Details')
    log(original_model)

    # Last layer manipulation
    # dim is the last layer number of nodes you want to feed in
    dim = 512
    PromptMachine.final_layer_zero_setting(original_model, dim)
    final_layer = original_model.decoder.final_layer_norm.weight
    print(final_layer)

    # squad dataset to be dowloaded in local
    # brew install wget
    # mkdir -p local_cache
    # wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O train.json
    # wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O dev.json

    # Prepare a subsample of train and validation df
    train_df = PromptMachine.json_to_dataframe_with_titles(json.load(open('train.json')))
    val_df = PromptMachine.json_to_dataframe_with_titles(json.load(open('dev.json')))
    df = PromptMachine.get_diverse_sample(val_df, sample_size=100, random_state=42)
    log(dash_line)
    log(f'Full train and validation dataset')
    log (df.head(5))

    # Read in the google flan t5 small model again for question answering with context(RAG)
    model_name='google/flan-t5-small'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("squad")
    df.to_json("100_val.json", orient="records", lines=True)
    df = pd.read_json("100_val.json", orient="records", lines=True)
    log(dash_line)
    log(f'Smaller validation sample dataset')
    log (df.head(5))

    # Sample output to show the model is responding with context
    log(dash_line)
    for i in range(3):
        df["answers"][i] = PromptMachine.question_answer2(df["question"][i], df["context"][i], tokenizer, original_model)
    
    # Inject a context and overwrite the question and observe
    df["question"][1] = "Who is Amitabh Bachchan?"
    df["context"][1] = "Amitabh Bachchan, born in 1942, is an Indian film producer, television host, occasional playback singer and former politician, and actor who works in Hindi cinema. In a film career spanning over five decades, he has starred in more than 200 films. Bachchan is widely regarded as one of the most successful and influential actors in the history of Indian cinema"
    log('New context injection')
    log(dash_line)
    df["answers"][1] = PromptMachine.question_answer2(df["question"][1], df["context"][1], tokenizer, original_model)

    # While the idea to take the length of prediction in the reference text is not actually correct, this is just for demonstration purposes how the evaluation module will work
    # # More logical rules that keep the prediction and reference lengths the same for BLEU and ROUGE matrices will need to be implemented.
    # ECSM implementation (BERTScore) can also be done

    for i in range(2):
        df["answers"][i] = PromptMachine.question_answer2(df["question"][i], df["context"][i], tokenizer, original_model)
        predictions = df["answers"][i]
        references = df["context"][i][0:len(predictions)]
        PromptMachine.Evaluatemet(predictions, references)


if __name__ == "__main__":
    """
    This Performs a training for the Dataframe and Save the Models files in Object Store
    """
    try:
        log(f"application: invoker.py, arguments")
        main()
    except Exception as e:
        log.exception(e)
        raise