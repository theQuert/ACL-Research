from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")
import pandas as pd

pipe = pipeline("text-classification", model="typeof/distilbert_base_uncased_csabstruct")

data_df = pd.read_csv("./data/abstract_w_all.csv")
citing_abs = data_df["citing_abstracts"].tolist()
cited_abs = data_df["cited_abstracts"].tolist()
citation_contexts = data_df["citation_contexts"].tolist()


store_dict_cited = {"BACKGROUND": [], "OBJECTIVE": [], "METHOD": [], "RESULT": [], "OTHER": []}

def citing_class(abstract):
    store_dict_citing = {"BACKGROUND": [], "OBJECTIVE": [], "METHOD": [], "RESULT": [], "OTHER": []}
    sentences = sent_tokenize(abstract)
    res = [pipe(sent)[0]["label"] for sent in sentences]
    for idx in range(len(res)):
        store_dict_citing[res[idx]].append(sentences[idx])
    background_methods = store_dict_citing["BACKGROUND"] + store_dict_citing["METHOD"]
    return background_methods 

def cited_class(abstract):
    store_dict_cited = {"BACKGROUND": [], "OBJECTIVE": [], "METHOD": [], "RESULT": [], "OTHER": []}
    sentences = sent_tokenize(abstract)
    res = [pipe(sent)[0]["label"] for sent in sentences]
    for idx in range(len(res)):
        store_dict_cited[res[idx]].append(sentences[idx])
    background = store_dict_cited["BACKGROUND"]
    objs = store_dict_cited["OBJECTIVE"]
    methods = store_dict_cited["METHOD"]
    return background, objs, methods

def call_gpt(prompt):
    openai.api_key = "sk-"
    inputs_for_gpt = " ".join(prompt.split()[:15000])

    completion = openai.ChatCompletion.create(
         model = "gpt-3.5-turbo-1106",
         messages = [
             {"role": "user", "content": inputs_for_gpt}
         ]
     )
    response = completion.choices[0].message.content
    return str(response)

# define prompts (organize the background, objs, known info to form inputs)

# prepare methods for validation
# call gpt (zeroshot) to form predicted methods iteratively
hyp_methods, ref_methods_lst = [], []
# for idx in range(len(data_df)):
for idx in range(0, 5):
    background_methods = citing_class(citing_abs[idx])
    cited_bg, objective, ref_methods = cited_class(cited_abs[idx])
    background_merged = background_methods + cited_bg[idx]
    citation_text = sent_tokenize(citation_contexts[idx])
    background_all = background_merged + citation_text
    background_text = " ".join(background_all)
    objs = " ".join(objective)
    prompt = f"""
    I have gathered information regarding the existing BACKGROUND and OBJECTIVE of current scholarly research. Please formulate research methodology that effectively bridges the gap between this background knowledge and the stated objectives.
    ### BACKGROUND
    {background_text}
    ### OBJECTIVE
    {objs}
    """
    hyp_methods.append(call_gpt(prompt).strip())
    ref_methods_text = " ".join(ref_methods)
    ref_methods_lst.append(ref_method_text)


pd.DataFrame({"hyps": hyp_methods}).to_csv("./hyps_methods.csv")
pd.DataFrame({"refs": ref_methods_lst}).to_csv("./refs_methods.csv")
    
