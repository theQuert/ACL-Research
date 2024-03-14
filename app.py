# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import nltk
import random, time
import datetime
# nltk.download("stopwords")
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.metrics import classification_report
import transformers
from transformers import BartForSequenceClassification, AdamW, BartTokenizer, get_linear_schedule_with_warmup, pipeline, set_seed
from transformers import pipeline, set_seed, BartTokenizer
from datasets import load_dataset, load_metric
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from datasets import Dataset, load_metric
import datasets
import gradio as gr
import pyperclip
# from openai import OpenAI
import openai
import google.generativeai as genai
import requests

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
# from vicuna_generate import *
# from convert_article import *

# Data preprocessing

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).

# Create the learning rate scheduler.

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def decode(paragraphs_needed):
    # model_ckpt = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained("theQuert/NetKUp-tokenzier")
    # pipe = pipeline("summarization", model="bart-decoder",tokenizer=tokenizer)
    pipe = pipeline("summarization", model="hyesunyun/update-summarization-bart-large-longformer",tokenizer=tokenizer)
    contexts = [str(pipe(paragraph)) for paragraph in paragraphs_needed]
    return contexts

def fetch_top_repos(keywords):
    url = "https://api.github.com/search/repositories"
    qu = f"{keywords}:python"
    query = {
        "q": qu,
        "sort": "stars",
        "order": "desc"
    }
    response = requests.get(url, params=query)
    data = response.json()
    top_repos = [item['name'] for item in data['items'][:5]]
    return '\n'.join(top_repos)

def split_article(article, trigger):
    if len(article.split("\\n ")) > len(article.split("\\\\c\\\\c")):
        paragraphs = article.split("\\n ")
    else:
        paragraphs = article.split("\\\\c\\\\c")
    pars = [str(par) + " -- " + str(trigger) for par in paragraphs]
    # pd.DataFrame({"paragraphs": paragraphs}).to_csv("./util/experiments/check_par.csv")
    format_pars = [par for par in paragraphs]
    formatted_input = "\n".join(format_pars)
    return pars, formatted_input 

def config():
    load_dotenv()

def evaluation(input_ref, gen_sols):
    gpt = f"""
As a paper reviewer of NLP top-conference, you have to generate the comments and scores given the machine-generated contents and human-generated contents. To generate the comments, you have to consider the USABILITY, RATIONALITY and INNOVATION.
### USABILITY:
### RATIONALITY:
### INNOVATION:
### Scores:
"""
    completion = openai.chat.completions.create(model = "gpt-4",
    messages = [
        {"role": "user", "content": gpt}
    ])
    response = completion.choices[0].message.content
    return str(response)
    



def call_gpt(input_abs, input_work):
    paragraph = input_abs
    trigger = input_work
    
    # tokenizer = BartTokenizer.from_pretrained("theQuert/NetKUp-tokenzier")
    # with open("/home/imlab/Quert/NetKUp/txt/abstract.txt", "r") as f: 
            # abstract = f.read()
    # with open("/home/imlab/Quert/NetKUp/txt/work.txt", "r") as f: 
            # related_work = f.read()
    # with open("/home/imlab/Quert/NetKUp/txt/topic.txt", "r") as f: 
            # topic = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/topic_1.txt", "r") as f: 
            topic_1 = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/abstract_1.txt", "r") as f: 
            abstract_1 = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/related_work_1.txt", "r") as f: 
            related_work_1 = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/method_1.txt", "r") as f: 
            method_1 = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/topic_2.txt", "r") as f: 
            topic_2 = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/abstract_2.txt", "r") as f: 
            abstract_2 = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/related_work_2.txt", "r") as f: 
            related_work_2 = f.read()
    with open("/home/imlab/Quert/NetKUp/txt/method_2.txt", "r") as f: 
            method_2 = f.read()
    abstract = paragraph
    related_work = trigger


    inputs_for_gpt = f"""
As a researcher of a computer science department major in natural language processing, please first return the top-3 best methods or ideas given ABSTRACT, RELATED WORK and TOPIC. Also, check if the generated methods are valid. If not valid, return another top-3 methods until match the innovation, usability and being reasonable. 
Lastly, return the latent and valid methods in given ABSTRACT, RELATED WORK and TOPIC.
\n
There are two examples below, and you have to learn the pattern of problem solving and return the latent METHODS. ### Let's think step by step:
### ABSTRACT: \n
{abstract_1}
### RELATED WORK: \n
{related_work_1}
### Let's think step by step:
### METHOD: \n
{method_1}
###### Below is the second example:
### ABSTRACT: \n
{abstract_2}
### RELATED WORK: \n
{related_work_2}
### Let's think step by step:
### METHOD: \n
{method_2}
\n
######
Now, provide reasonable solutions as METHOD to solve latent disadvantages in given ABSTRACT and RELATED WORK. Also, predict if the generated methods really solve the laten disadvantages found in given ABSTRACT and RELATED WORK. \n 
### ABSTRACT: \n
{abstract}
### RELATED WORK: \n
{related_work}
### Let's think step by step:
METHOD: \n
""" 

    methods_inputs = f"""
As a researcher of a computer science department major in natural language processing, please first return the top-3 best methods or ideas given ABSTRACT, RELATED WORK and TOPIC. Then check if the generated methods are valid. If not valid, return another top-3 methods until match. 
Lastly, return the latent and valid methods in given ABSTRACT, RELATED WORK and TOPIC.
\n
There are two examples below, and you have to learn the pattern of problem solving and return the latent METHODS. ### Let's think step by step:
### ABSTRACT: \n
{abstract_1}
### RELATED WORK: \n
{related_work_1}
### Let's think step by step:
### METHOD: \n
{method_1}
###### Below is the second example:
### ABSTRACT: \n
{abstract_2}
### RELATED WORK: \n
{related_work_2}
### Let's think step by step:
### METHOD: \n
{method_2}
\n
######
Now, you have to understand and think step-by-step according to given ABSTRACT and RELATED WORK. Scoring the generated methods, and mark the scores at the end of each method. Regenerate the methods which you think it is not good enough until it match your understanding. Any latent solutions are accepteable to be the candidate solutions, you have to choose top-5 of them to solve the disadvantages you find from given ABSTRACT and RELATED WORK. Also, scoring the generated methods, and mark the scores at the end of each method. Re-generate if your think it is not good enough, and predict if the generated methods really solve the latent disadvantages found in given ABSTRACT and RELATED WORK. \n 
### ABSTRACT: \n
{abstract}
### RELATED WORK: \n
{related_work}
METHOD: \n
PREDICTED RESULT: \n
""" 
    eval_inputs = f"""
As a researcher of a computer science department PHD in natural language processing and deep learning, please first return the top-3 best methods or ideas given ABSTRACT, RELATED WORK and TOPIC. Then check if the generated methods are valid. If not valid, return another top-3 methods until match. Lastly, return the latent and valid methods in given ABSTRACT, RELATED WORK and TOPIC.
\n
There are two examples below, and you have to learn the pattern of problem solving and return the latent METHODS. ### Let's think step by step:
### ABSTRACT: \n
{abstract_1}
### RELATED WORK: \n
{related_work_1}
### Let's think step by step:
### METHOD: \n
{method_1}
###### Below is the second example:
### ABSTRACT: \n
{abstract_2}
### RELATED WORK: \n
{related_work_2}
### Let's think step by step:
### METHOD: \n
{method_2}
\n
######
As an NLP and Deep Learning developer, now you have to scope the draft of implementation with Python Code. Provide 5 reasonable methods to solve latent disadvantages which is suitable for generating Python Codes by given ABSTRACT and RELATED WORK. Also, scoring the generated methods, and mark the scores at the end of each method. Re-generate if your think it is not good enough, and predict if the generated methods really solve the latent disadvantages found in given ABSTRACT and RELATED WORK. \n 
PREDICTED RESULT: \n
""" 
    completion = openai.chat.completions.create(model = "gpt-4",
    messages = [
        {"role": "user", "content": inputs_for_gpt}
    ])
    response = completion.choices[0].message.content
    # if "<"+response.split("<")[-1].strip() == "<"+paragraph.split("<")[-1].strip(): response = response 
    # else: response = response + " <"+paragraph.split("<")[-1].strip()
    completion = openai.chat.completions.create(model = "gpt-4",
    messages = [
        {"role": "user", "content": methods_inputs}
    ])
    methods = completion.choices[0].message.content
    completion = openai.chat.completions.create(model = "gpt-4",
    messages = [
        {"role": "user", "content": eval_inputs}
    ])
    self_eval = completion.choices[0].message.content
    ask_repo = f"""
According to the following recommended solutions to solve research problems, please recommend the top-5 related keywords aims to search respositories on GitHub. Return only the fine-grained related keywords.
```
{methods}
```
KEYWORDS:
    """

    ask_for_code = f"""
According to the following recommended solutions to solve research problems, please generate python code including the preprocessing, list the steps in more details with the builds of data preprocessing, neural network building and specify the steps in texts, then return the detailed implementable code with Pytorch and Pytorch-Lightning. If some of the steps cannot be generated in python code with pytorch implementation, please list the the code in general. 
Solutions:
```
{methods}
```
Return the python code:
### CODE:

    """ 
    completion = openai.chat.completions.create(model = "gpt-4",
    messages = [
        {"role": "user", "content": ask_repo}
    ])
    keywords = completion.choices[0].message.content
    # keywords = keywords.split("\n")
    # rec_repo = fetch_top_repos(keywords[0])
    rec_repo = keywords

    completion = openai.chat.completions.create(model = "gpt-4",
    messages = [
        {"role": "user", "content": ask_for_code}
    ])
    gen_code = completion.choices[0].message.content

    return str(response), str(methods), str(rec_repo), str(gen_code)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

    
def main(input_article, input_trigger, input_ref):
    input_abs = input_article
    input_work = input_trigger

    paths = [".util/experiments/input_paragraphs.csv",
             "./util/experiments/formatted_input.txt",
             "./util/experiments/updated_article.txt",
             "./util/experiments/paragraphs_needed.txt",
             "./util/experiments/updated_paragraphs.txt",
             "./util/experiments/paragraphs_with_prompts.csv",
             "./util/experiments/classification.csv",
             "./util/experiments/paragraphs_needed.csv",
             "./util/experiments/par_with_class.csv",
             ]

    for path in paths: 
        try:
            if os.path.isfile(path): os.remove(path)
        except: continue 

    modified = "TRUE"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_sols, methods, rec_repo, gen_code = call_gpt(input_abs, input_work)
    if input_ref:
        eval_ref = evaluation(input_ref, gen_sols)
    else: eval_ref = "None"


    """
    # Predictions
    predictions = []
    for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            with torch.no_grad():
                output= model(b_input_ids,
                              attention_mask=b_input_mask)
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                predictions.extend(list(pred_flat))

    # Write predictions for each paragraph
    df_output = pd.DataFrame({"target": predictions}).to_csv('./util/experiments/classification.csv', index=False)
    if len(data_test)==1: predictions[0] = 1

    # extract ids for update-needed paragraphs (extract the idx with predicted target == 1)
    pos_ids = [idx for idx in range(len(predictions)) if predictions[idx]==1]
    neg_ids = [idx for idx in range(len(predictions)) if predictions[idx]==0]

    # feed the positive paragraphs to decoder
    paragraphs_needed = [data_test[idx] for idx in pos_ids]
    paragraphs_needed = [par.split(" -- ")[0].replace("[ADD]", "") for par in paragraphs_needed]
    pd.DataFrame({"paragraph": paragraphs_needed}).to_csv("./util/experiments/paragraphs_needed.csv", index=False)
    paragraphs_needed_str = "\n\n".join(paragraphs_needed)
    # paragraphs_needed_str = paragraphs_needed_str.replace("Updated Paragraph:\n", "")
    with open("./util/experiments/paragraphs_needed.txt", "w") as f:
        f.write(paragraphs_needed_str)

    # updated_paragraphs = decode(input_paragraph, input_trigger)
    # updated_paragraphs = call_vicuna(paragraphs_needed, input_trigger)
    config()
    updated_paragraphs = [call_gpt(paragraph, input_trigger) for paragraph in paragraphs_needed]
    updated_paragraphs_str = "\n\n".join(updated_paragraphs)
    updated_paragraphs_str = updated_paragraphs_str.replace("Updated Paragraph:\n", "")
    with open("./util/experiments/updated_paragraphs.txt", "w") as f:
        f.write(updated_paragraphs_str)

    # merge updated paragraphs with non-updated paragraphs
    paragraphs_merged = data_test.copy()
    paragraphs_merged = [str(par).split(" -- ")[0] for par in paragraphs_merged]
    paragraphs_old = paragraphs_merged.copy()
    for idx in range(len(pos_ids)):
        paragraphs_merged[pos_ids[idx]] = updated_paragraphs[idx]
    global_pos = [pos_ids[idx] for idx in range(len(pos_ids))]

    updated_color_sents = []
    old_color_sents = []
    for idx in range(len(paragraphs_merged)):
        if idx not in global_pos:
            # color_sents[paragraphs_merged]="white"
            tup = (paragraphs_merged[idx]+"\n", "Unchanged")
            updated_color_sents.append(tup)
        else: 
            tup = (paragraphs_merged[idx]+"\n", "Updated")
            updated_color_sents.append(tup)
    for idx in range(len(paragraphs_old)):
        if idx not in global_pos:
            tup = (paragraphs_old[idx]+"\n", "Unchanged")
            old_color_sents.append(tup)
        else:
            tup = (paragraphs_old[idx]+"\n", "Modified")
            old_color_sents.append(tup)

            
    sep = "\n"
    # paragarphs_merged = ["".join(par.split(" -- ")[:-1]) for par in paragraphs_merged]
    updated_article = str(sep.join(paragraphs_merged))
    updated_article = updated_article.replace("[{'summary_text': '", "").replace("'}]", "").strip()
    class_res = pd.read_csv("./util/experiments/classification.csv")
    if class_res.target.values.all() == 0: modified="False"

    if len(data_test)==1: 
        modified="TRUE"
        updated_article = call_gpt(input_article, input_trigger)
    with open("./util/experiments/updated_article.txt", "w") as f:
        f.write(updated_article)

    # combine the predictions and paragraphs into csv format file
    merged_par_pred_df = pd.DataFrame({"paragraphs": data_test, "predictions": predictions}).to_csv("./util/experiments/par_with_class.csv")
    # modified_in_all = str(len(paragraphs_needed)) + " / " + str(len(data_test))
    # return formatted_input, updated_article
    # return formatted_input, updated_color_sents
    return old_color_sents, updated_color_sents
    """
    return gen_sols, methods, rec_repo, gen_code, eval_ref

def copy_to_clipboard(t):
    with open("./util/experiments/updated_article.txt", "r") as f:
        t = f.read()
        pyperclip.copy(t)

def compare_versions():
    with open("./util/experiments/paragraphs_needed.txt", "r") as f:
        old = f.read()
        old = old.replace("[ADD]", "")
    with open("./util/experiments/updated_paragraphs.txt", "r") as f:
        new = f.read()
        new = new.replace("[ADD]", "")
    return old, new



with open("../NetKUp/txt/abstract_0.txt", "r") as f:
    exin_0 = f.read()
with open("../NetKUp/txt/work_0.txt", "r") as f:
    trigger_0 = f.read()
with open("../NetKUp/txt/abstract_1.txt", "r") as f:
    exin_1 = f.read()
with open("../NetKUp/txt/work_1.txt", "r") as f:
    trigger_1 = f.read()
with open("../NetKUp/txt/abstract_2.txt", "r") as f:
    exin_2 = f.read()
with open("../NetKUp/txt/work_2.txt", "r") as f:
    trigger_2 = f.read()
with open("../NetKUp/txt/abstract_3.txt", "r") as f:
    exin_3 = f.read()
with open("../NetKUp/txt/work_3.txt", "r") as f:
    trigger_3 = f.read()
with open("../NetKUp/txt/abstract_4.txt", "r") as f:
    exin_4 = f.read()
with open("../NetKUp/txt/work_4.txt", "r") as f:
    trigger_4 = f.read()
with open("../NetKUp/txt/abstract_5.txt", "r") as f:
    exin_5 = f.read()
with open("../NetKUp/txt/work_5.txt", "r") as f:
    trigger_5 = f.read()
with open("../NetKUp/txt/abstract_6.txt", "r") as f:
    exin_6 = f.read()
with open("../NetKUp/txt/work_6.txt", "r") as f:
    trigger_6 = f.read()
with open("../NetKUp/txt/abstract_7.txt", "r") as f:
    exin_7 = f.read()
with open("../NetKUp/txt/work_7.txt", "r") as f:
    trigger_7 = f.read()
with open("../NetKUp/txt/abstract_8.txt", "r") as f:
    exin_8 = f.read()
with open("../NetKUp/txt/work_8.txt", "r") as f:
    trigger_8 = f.read()
with open("../NetKUp/txt/abstract_9.txt", "r") as f:
    exin_9 = f.read()
with open("../NetKUp/txt/work_9.txt", "r") as f:
    trigger_9 = f.read()
with open("../NetKUp/txt/ref_0.txt", "r") as f:
    ref_0 = f.read()
with open("../NetKUp/txt/ref_1.txt", "r") as f:
    ref_1 = f.read()
with open("../NetKUp/txt/ref_2.txt", "r") as f:
    ref_2 = f.read()
with open("../NetKUp/txt/ref_3.txt", "r") as f:
    ref_3 = f.read()
with open("../NetKUp/txt/ref_4.txt", "r") as f:
    ref_4 = f.read()
with open("../NetKUp/txt/ref_5.txt", "r") as f:
    ref_5 = f.read()
with open("../NetKUp/txt/ref_6.txt", "r") as f:
    ref_6 = f.read()
with open("../NetKUp/txt/ref_7.txt", "r") as f:
    ref_7 = f.read()
with open("../NetKUp/txt/ref_8.txt", "r") as f:
    ref_8 = f.read()
with open("../NetKUp/txt/ref_9.txt", "r") as f:
    ref_9 = f.read()

with gr.Blocks(title="Research Methods Geneartion") as demo:
    gr.HTML("""<div style="text-align: center; max-width: 700px; margin: 0 auto;">
            <div
            style="
                display: inline-flex;
                align-items: center;
                gap: 0.8rem;
                font-size: 1.75rem;
            "
            >
            <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
                Research Methods Generation 
            </h1>
            </div>"""
        )
    with gr.Tab("Generation"):
        gr.Markdown("## All-In-One tool for Research Methods Generation, Detailed Evaluation, and Code Generation")
        gr.Markdown("### Input examples could be found in the subsequent tab.")
        gr.Markdown("#### Return: Recommended Solutions, Searched Keywords, and Code Generation.")
        gr.Markdown("#### If the input tokens are too long, it would take more time to process...")
        # input_key = gr.Textbox(label="OpenAI API Key", lines=1, placeholder="sh-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        input_1 = gr.Textbox(label="Abstract w/ Introduction", lines=5, placeholder="Input the contexts...")
        input_2 = gr.Textbox(label="Related Work", lines=5, placeholder="Input the contexts...") 
        input_3 = gr.Textbox(label="References", lines=5, placeholder="Optional") 
        with gr.Row():
            gr.ClearButton([input_1, input_2, input_3])
            btn = gr.Button(value="Submit")
        with gr.Row():
            output_0 = gr.Textbox(label="Top-5 latent solutions:", show_copy_button=True, show_label=True, interactive=False)

        """
        with gr.Row():
            # output_1 = gr.HighlightedText(label="Inputs")
            # output_2 = gr.HighlightedText(label="Generation")
            output = gr.Textbox(label="Generation", show_copy_button=True)
        """

        with gr.Row():
            output = gr.Textbox(label="Self-Evaluation with predicted results.", show_copy_button=True, show_label=True)
        with gr.Row():
            output_eval = gr.Textbox(label="Detailed Evaluation (If refererences are given...)", show_copy_button=True, show_label=True)
        with gr.Row():
            rec_repo = gr.Textbox(label="Keywords", show_copy_button=True, show_label=True)
        with gr.Row():
            gen_code = gr.Code(language="python", lines=5, label="Code")

        # with gr.Row():
        #     output_2 = gr.Textbox(label="Self-Eval", show_copy_button=True)
        if input_3 is None: input_3 = "None"
        btn.click(fn=main, inputs=[input_1, input_2, input_3], outputs=[output, output_0, rec_repo, gen_code, output_eval])
        # btn_copy = gr.Button(value="Copy Generated Methods to Clipboard")
        # btn_copy.click(fn=copy_to_clipboard, inputs=[output_0], outputs=[])

        com_1_value, com_2_value = "Pleas finish article updating, then click the button above", "Pls finish article updating, then click the button above."
   #  with gr.Tab("Compare between versions"):
   #      btn_com = gr.Button(value="Differences Highlighting")
   #      with gr.Row():
   #          com_1 = gr.Textbox(label="Non-update Paragraphs", value=com_1_value, lines=15)
   #          com_2 = gr.Textbox(label="Updated Paragraphs", value=com_2_value, lines=15)
   #      btn_com.click(fn=compare_versions, inputs=[], outputs=[com_1, com_2])
    with gr.Tab("Examples"):
        gr.Markdown("## Examples")
        gr.Markdown("### Examples are listed below; contents would be auto-filled when clicked.")
        gr.Examples(
            examples=[[exin_0, trigger_0, ref_0], [exin_1, trigger_1, ref_1], [exin_2, trigger_2, ref_2], [exin_3, trigger_3, ref_3], [exin_4, trigger_4, ref_4], [exin_5, trigger_5, ref_5], [exin_6, trigger_6, ref_6], [exin_7, trigger_7, ref_7], [exin_8, trigger_8, ref_8], [exin_9, trigger_9, ref_9]],
            fn=main,
            inputs=[input_1, input_2, input_3],
            outputs = [output],
            # outputs=[output_1, output_2],
            # cache_examples=True,
            # run_on_click=True,
                ),
    # with gr.Tab("Human Evaluation"):
    #     gr.Markdown("## Human Evaluation")
        # gr.Markdown("### Examples are listed below; contents would be auto-filled when clicked.")
        # gr.Examples(
        #     examples=[[exin_0, trigger_0, ref_0], [exin_1, trigger_1, ref_1], [exin_2, trigger_2, ref_2], [exin_3, trigger_3, ref_3], [exin_4, trigger_4, ref_4], [exin_5, trigger_5, ref_5], [exin_6, trigger_6, ref_6], [exin_7, trigger_7, ref_7], [exin_8, trigger_8, ref_8], [exin_9, trigger_9, ref_9]],
        #     fn=main,
        #     inputs=[input_1, input_2, input_3],
        #     outputs = [output],
        #     # outputs=[output_1, output_2],
        #     # cache_examples=True,
        #     # run_on_click=True,
        #         ),
        
    """
    gr.HTML("""
            <div align="center">
                <p>
                Demo by 🤗 <a href="https://github.com/thequert" target="_blank"><b>Yu-Ting Lee</b></a>
                </p>
            </div>
            <div align="center">
                <p>
                Supported by <a href="https://iis.sinica.edu.tw/en/index.html"><b>Institute of Information Science, Academia Sinica</b></a>
                </p>
            </div>
        """
        )
    """

demo.launch(server_name="0.0.0.0", server_port=10010)

