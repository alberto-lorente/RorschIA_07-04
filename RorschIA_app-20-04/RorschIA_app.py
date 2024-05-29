import streamlit as st
import pandas as pd
import deepl
from langdetect import detect
import pickle
import torch
import spacy
from itertools import compress
from transformers import BertModel, BertTokenizer
from functions import raw_text_response_eval
from functions import evaluate_text

nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classifications = ["individual_determinants", "macro_determinants", "individual_contents","macro_contents"]

# May need to change the info_dict to put the right path for the models

st.title("RorschIA")
text_entered = st.text_input("Paste the text of your protocol or sentence :)")
process_button = st.button("Process")

if process_button == True:
    try:
        # will go through if it is a full report 
        evaluation = raw_text_response_eval(text_entered)
        csv = evaluation.to_csv()
        st.download_button(label='Download CSV', data=csv, file_name='RorschIA_results.csv', mime='text/csv')
        
    except:
        # will trigger if the response is a plain sentence
        lang = detect(text_entered)
        if lang == "fr":
            # st.write("French")
            with open("RorschIA_app/DEEPL_API_KEY.txt", "r") as f:
                API_KEY = f.read()
                API_KEY = API_KEY.strip("\n")
            translator = deepl.Translator(API_KEY)
            result = translator.translate_text(text_entered, target_lang="EN-US", preserve_formatting=True)
            response = result.text
        else: # language will be english
            response = text_entered
            
        st.write("*Response sentence*: {} ".format(response))
        
        for model in classifications:
            classification = evaluate_text(model, response)
            # print(classification)
            st.write("{} : **{}** ".format(classification, classification))


