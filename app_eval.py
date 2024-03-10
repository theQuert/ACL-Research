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
import openai
import requests
import copy

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


def evaluation(input_ref, gen_sols):
    gpt = f"""
As a paper reviewer of NLP top-conference, you have to generate the comments and scores given the machine-generated contents and human-generated contents. To generate the comments, you have to consider the USABILITY, RATIONALITY and INNOVATION.
### Machine-genearted conrtents: {gen_sols} \n
### Human-generated contents: {input_ref} \n
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output, output_0, rec_repo, gen_code = call_gpt(input_abs, input_work)
    if input_ref:
        output_eval = evaluation(input_ref, output)
    else: output_eval = "None"
    return output_0, output, output_eval, rec_repo, gen_code

def generate_unique_filename():
    base_name = "/home/imlab/Quert/NetKUp/labeling_out/"
    number = 0
    while True:
        file_path = os.path.join(base_name, f"{base_name}saved_content_{number}.txt")
        if not os.path.exists(file_path):
            return file_path
        number += 1 

cnt = 0
# This function will be used to save the edits to a text file
def save_edits(input_1, input_2, input_3, output_0_original, output_original, output_eval_original, rec_repo_original,
               output_0, output, output_eval, rec_repo):
    global cnt

    # file_path = f"/home/imlab/Quert/NetKUp/labeling_out/saved_content_{cnt}.txt"
    file_path = generate_unique_filename()
    cnt += 1

    with open(file_path, "w") as file:
        file.write(f"Input Article:\n {input_1}\n")
        file.write(f"Input Trigger:\n {input_2}\n")
        file.write(f"Input References:\n {input_3}\n\n")

        contents = {
            "Top-5 latent solutions": (output_0_original, output_0),
            "Self-Evaluation": (output_original, output),
            "Detailed Evaluation": (output_eval_original, output_eval),
            "Keywords": (rec_repo_original, rec_repo),
            # "Code": (gen_code_original, gen_code),
        }

        def is_content_equal(original, edited):
            return original.strip().lower() == edited.strip().lower()

        for key, (original, edited) in contents.items():
            if is_content_equal(original, edited):
                file.write(f"{key}:\n {original}\n\n=====================================\n\n")
            else:
                file.write(f"{key}:\n {edited}\n\n=====================================\n\n")
    gr.Info("Saved")
    # return



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

def get_text_from_textbox(input_text):
    return input_text


for i in range(30):
    file_path = f"../NetKUp/txt/abstract_{i}.txt"
    with open(file_path, "r") as f:
        globals()[f"exin_{i}"] = f.read()
for i in range(30):
    file_path = f"../NetKUp/txt/work_{i}.txt"
    with open(file_path, "r") as f:
        globals()[f"trigger_{i}"] = f.read()
for i in range(30):
    file_path = f"../NetKUp/txt/ref_{i}.txt"
    with open(file_path, "r") as f:
        globals()[f"ref_{i}"] = f.read()




with gr.Blocks(title="Research Methods Geneartion (Human-Evaluation Mode)") as demo:
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
                Research Methods Generation (Human-Evaluation Mode)
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
            clear_button = gr.Button(value="Clear")
            submit_button = gr.Button(value="Submit")

        # output_0_original_str, output_original_str, output_eval_original_str, rec_repo_original_str, gen_code_original_py = main(input_1, input_2, input_3)

        # output_0_original = gr.Textbox(visible=False, value=output_0_original_str)
        # output_original = gr.Textbox(visible=False, value=output_original_str)
        # output_eval_original = gr.Textbox(visible=False, value=output_eval_original_str)
        # rec_repo_original = gr.Textbox(visible=False, value=rec_repo_original_str)
        # gen_code_original = gr.Code(language="python", visible=False, value=gen_code_original)
    

        # output_0 = gr.Textbox(label="Top-5 latent solutions:", show_copy_button=True, show_label=True, interactive=True, value=output_0_original_str)
        # output = gr.Textbox(label="Self-Evaluation with predicted results.", show_copy_button=True, show_label=True, interactive=True, value=output_original_str)
        # output_eval = gr.Textbox(label="Detailed Evaluation (If references are given...)", show_copy_button=True, show_label=True, interactive=True, value=output_eval_original_str)
        # rec_repo = gr.Textbox(label="Keywords", show_copy_button=True, show_label=True, interactive=True, value=rec_repo_original_str)
        # gen_code = gr.Code(language="python", lines=5, label="Code", interactive=False, value=gen_code_original_py)

        output_0_original = gr.Textbox(label="Top-5 latent solutions:", show_copy_button=True, show_label=True, interactive=True)
        output_original = gr.Textbox(label="Self-Evaluation with predicted results.", show_copy_button=True, show_label=True, interactive=True)
        output_eval_original = gr.Textbox(label="Detailed Evaluation (If references are given...)", show_copy_button=True, show_label=True, interactive=True)
        rec_repo_original = gr.Textbox(label="Keywords", show_copy_button=True, show_label=True, interactive=True)
        gen_code_original = gr.Code(language="python", lines=5, label="Code", show_label=True, interactive=False)

        output_0 = output_0_original
        output = output_original
        output_eval = output_eval_original
        rec_repo = rec_repo_original
        gen_code = gen_code_original


        # output_0 = gr.Textbox(visible=False)
        # output = gr.Textbox(visible=False)
        # output_eval = gr.Textbox(visible=False)
        # rec_repo = gr.Textbox(visible=False)
        save_button = gr.Button(value="Save Edits (save if modified else same)")

        with gr.Row():
            clear_button.click(inputs=[input_1, input_2, input_3, output_0_original, output_original, output_eval_original, rec_repo_original, gen_code_original],
                                fn=lambda *args: [""]*8)
            # clear_button.click(inputs=[input_1, input_2, input_3, output_0, output, output_eval, rec_repo, gen_code],
                                # fn=lambda *args: [""]*8)
            submit_button.click(fn=main, inputs=[input_1, input_2, input_3], outputs=[output_0_original, output_original, output_eval_original, rec_repo_original, gen_code_original])
            save_button.click(
                fn=save_edits,
                inputs=[input_1, input_2, input_3, output_0_original, output_original, output_eval_original, rec_repo_original, output_0, output, output_eval, rec_repo]
            )


        com_1_value, com_2_value = "Please finish article updating, then click the button above", "Please finish article updating, then click the button above."
    with gr.Tab("Examples"):
        gr.Markdown("## Examples")
        gr.Markdown("### Examples are listed below; contents would be auto-filled when clicked.")
        gr.Examples(
            examples=[[exin_0, trigger_0, ref_0], [exin_1, trigger_1, ref_1], [exin_2, trigger_2, ref_2], [exin_3, trigger_3, ref_3], [exin_4, trigger_4, ref_4], [exin_5, trigger_5, ref_5], [exin_6, trigger_6, ref_6], [exin_7, trigger_7, ref_7], [exin_8, trigger_8, ref_8], [exin_9, trigger_9, ref_9], [exin_10, trigger_10, ref_10], [exin_11, trigger_11, ref_11], [exin_12, trigger_12, ref_12], [exin_13, trigger_13, ref_13], [exin_14, trigger_14, ref_14], [exin_15, trigger_15, ref_15], [exin_16, trigger_16, ref_16], [exin_17, trigger_17, ref_17], [exin_18, trigger_18, ref_18], [exin_19, trigger_19, ref_19], [exin_20, trigger_20, ref_20], [exin_21, trigger_21, ref_21], [exin_22, trigger_22, ref_22], [exin_23, trigger_23, ref_23], [exin_24, trigger_24, ref_24], [exin_25, trigger_25, ref_25], [exin_26, trigger_26, ref_26], [exin_27, trigger_27, ref_27], [exin_28, trigger_28, ref_28], [exin_29, trigger_29, ref_29]],
            fn=main,
            inputs=[input_1, input_2, input_3],
            outputs = [output_0_original, output_original, output_eval_original, rec_repo_original, gen_code_original],
            examples_per_page=6
                ),

    # with gr.Tab("Human Evaluation"):
    #     gr.Markdown("## Human Evaluation")
        
    gr.HTML("""
            <div align="center">
                <p>
                Demo by 🤗 
                </p>
            </div>
            <div align="center">
                <p><b>
                Language and Knowledge Technologies Lab,
                </b></p>
                <p>
                <a href="https://iis.sinica.edu.tw/en/index.html"><b>Institute of Information Science, Academia Sinica, Taipei, Taiwan.</b></a>
                </p>
            </div>
        """
        )

demo.launch(server_name="0.0.0.0", server_port=10020)

