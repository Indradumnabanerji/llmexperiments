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
warnings.filterwarnings('ignore')
tqdm.pandas()

def log(text):
    print(text)

class PromptMachine:
    #Optional Config parameters can be placed here
    def __init__(self, parameter):
        self.parameter = None
    
    def summary_generator(index, dataset, tokenizer, original_model):
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        prompt = f"""
        Summarize the following conversation.
        {dialogue}
        Summary:
        """
        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
        original_model.generate(
            inputs["input_ids"], 
            max_new_tokens=200,
            )[0], 
            skip_special_tokens=True
            )
        dash_line = '-'.join('' for x in range(100))
        log(dash_line)
        log(f'INPUT PROMPT:\n{prompt}')
        log(dash_line)
        log(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
        log(dash_line)
        log(f'MODEL GENERATION - ZERO SHOT:\n{output}')
    
    def question_answer(query, tokenizer, original_model):
        prompt = f"""
        Answer the query in brief and a catchy manner in about 100 characters
        {query}
        """
        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
        original_model.generate(
            inputs["input_ids"], 
            max_new_tokens=200,
            )[0], 
            skip_special_tokens=True
            )
        return output
    
    def final_layer_zero_setting(original_model, dim):
        # change the dimensions of final layer and set it to zero or anything else
        original_model.decoder.final_layer_norm.weight=torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
        return original_model.decoder.final_layer_norm.weight
    
    def json_to_dataframe_with_titles(json_data):
        qas = []
        context = []
        is_impossible = []
        answers = []
        titles = []

        for article in json_data['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    qas.append(qa['question'].strip())
                    context.append(paragraph['context'])
                    is_impossible.append(qa['is_impossible'])
                    
                    ans_list = []
                    for ans in qa['answers']:
                        ans_list.append(ans['text'])
                    answers.append(ans_list)
                    titles.append(title)

        df = pd.DataFrame({'title': titles, 'question': qas, 'context': context, 'is_impossible': is_impossible, 'answers': answers})
        return df
    
    def get_diverse_sample(df, sample_size=100, random_state=42):

        sample_df = df.groupby(['title', 'is_impossible']).apply(lambda x: x.sample(min(len(x), max(1, sample_size // 50)), random_state=random_state)).reset_index(drop=True)
        if len(sample_df) < sample_size:
            remaining_sample_size = sample_size - len(sample_df)
            remaining_df = df.drop(sample_df.index).sample(remaining_sample_size, random_state=random_state)
            sample_df = pd.concat([sample_df, remaining_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        return sample_df.sample(min(sample_size, len(sample_df)), random_state=random_state).reset_index(drop=True)
    
    def get_prompt(row):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
        Question: {row.question}\n\n
        Context: {row.context}\n\n
        Answer:\n""",
            },
        ]

    

    def dataframe_to_jsonl(df):
        def create_jsonl_entry(row):
            answer = row["answers"][0] if row["answers"] else "I don't know"
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
                Question: {row.question}\n\n
                Context: {row.context}\n\n
                Answer:\n""",
                },
                {"role": "assistant", "content": answer},
            ]
            return json.dumps({"messages": messages})


    def question_answer2(query, context, tokenizer, original_model):
        prompt = f"""
        Answer the query in brief and a catchy manner in about 100 characters
        {query}
        Use the additional information given in the context below to come up an answer
        {context}
        """
        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
        original_model.generate(
            inputs["input_ids"], 
            max_new_tokens=200,
            )[0], 
            skip_special_tokens=True
            )
        log(f'Generation with Context added:\n{output}')
        return str(output)
        
    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    
    def Evaluatemet(predictions, references):
        rouge = evaluate.load('rouge')
        results_rouge = rouge.compute(predictions=predictions,references=references)

        bleu = evaluate.load("bleu")
        results_bleuscore = bleu.compute(predictions=predictions,references=references)

        dash_line = '-'.join('' for x in range(100))
        log(dash_line)
        log(results_rouge)
        log(dash_line)
        log(results_bleuscore)


# if __name__ == "__main__":
#     huggingface_dataset_name = "knkarthick/dialogsum"
#     model_name='google/flan-t5-base'
#     original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     dataset = load_dataset(huggingface_dataset_name)

    
    

#     index = 200
#     PromptMachine.summary_generator(index)
    
#     #Print number of trainable parameters
#     #log(original_model)
#     #log(PromptMachine.print_number_of_trainable_model_parameters(original_model))

#     # Set layer norm weights for decoder
#     #original_model.decoder.final_layer_norm.weight.data = 0 
#     # original_model["decoder.final_layer_norm.weight.data"] = 0
#     #for k, v in original_model.state_dict().items():
#     #    print(k, v.shape)

#     query = "What is a good way to procrastinate"
#     PromptMachine.question_answer(query)
#     # original_model.decoder.final_layer_norm.weight=torch.nn.Parameter(torch.ones(768, dtype=float))
#     # print(original_model.decoder.final_layer_norm.weight)
#     # PromptMachine.question_answer(query)
#     original_model.decoder.final_layer_norm.weight=torch.nn.Parameter(torch.zeros(768, dtype=torch.bfloat16))
#     print(original_model.decoder.final_layer_norm.weight)
#     PromptMachine.question_answer(query)

    

    

    

    

    


