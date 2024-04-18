import pandas as pd
import numpy as np

import re
import json
import spacy


import streamlit as st
import deepl

import spacy
import pickle
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_sm')

def get_responses(raw_text):
  
  """This function takes plain text, parses the in-text markers to extract the text corresponding to each Rorschach figure,
  runs the responses through the Deepl API to translate them and returns a dictionary with the responses for each figure properly organized.
  """

  dict_numbers = {"I": 1, "II": 2,
                  "III": 3, "IV": 4,
                  "V": 5, "VI": 6,
                  "VII": 7, "VIII": 8,
                  "IX": 9, "X": 10}

  regex_response_number = '[A-Z]+/'                                # since the responses alway start with the number + \

  list_number_figure = re.findall(regex_response_number, raw_text)  # finds the regex for the number of responses

  text_i = re.split(regex_response_number, raw_text)

  # regex_choix_pos = r'Choix +(.*?)Choix -'

  # regex_choix_neg = r'Choix -(.*?)Rq :'

  # regex_rq = r'Rq :(.*?)\n'

  try: 
    
    text_choix_pos = re.findall(r'Choix \+ :(.*?)\nCh.?', raw_text)[0].strip()

    text_choix_neg = re.findall(r'Choix \- :(.*?)\n', raw_text)[0].strip()

    text_rq = re.findall(r'Rq :(.*?)\n', raw_text)[0].strip()

    additional_info = {"Choix_pos": text_choix_pos, "Choix_neg": text_choix_neg, "Rq": text_rq}
    
  except:
      
      additional_info = {"Choix_pos": "nada", "Choix_neg": "nada", "Rq": "nada"}
      
    #   print("Additional info not found")


  list_responses = []

  for i in range(len(list_number_figure)):

    if list_number_figure[i] not in text_i[i]:

      j = i + 1 # there is a \n\n string at index 0 so the text actually starts at index 1
      # print("not found")

    else:

      j = i

    dict_responses = {}

    n_response = list_number_figure[i][:-1]

    number = dict_numbers[n_response]

    text_i = re.split(regex_response_number, raw_text)

    full_text = text_i[j].strip()

    regex_line_break = '\n++'

    list_sentences_raw = text_i[j].split(".")

    text_i[j] = re.sub(regex_line_break, " ", text_i[j]) \
                  .replace("@", "") \
                  .replace("^", "") \
                  .replace("V", "") \
                  .replace(">", "") \
                  .replace("  ", " ") \
                  .strip() \

    # DEEPL API TRANSLATION

    # add language detector is it english or french? - if 

    with open("DEEPL_API_KEY.txt", "r") as f:
      API_KEY = f.read()
    
    # print("API Key Found!")


    translator = deepl.Translator(API_KEY)
    result = translator.translate_text(text_i[j], target_lang="EN-US", preserve_formatting=True)

    text_i[j] = result.text

    dict_responses["figure_number"] = number

    dict_responses["raw_response"] = full_text

    dict_responses["clean_response"] = text_i[j]

    list_sentences_clean = text_i[j].split(".")

    clean_sentences = []

    i = 0

    special_markers_list = ["@", "^", "V ", "<", ">",]

    # structuring the sentences inside the while loop

    while i < len(list_sentences_clean):

      if "Choix" in list_sentences_clean[i]:  # if we reach choix, there are no more actual responses by the patient, those are just comments by the psychologist
        break

      elif "Choice" in list_sentences_clean[i]:  # if we reach choix, there are no more actual responses by the patient, those are just comments by the psychologist
        break

      j = i + 1

      # cleaning the sentences and adding the . back at the end of the sentence

      dict_sentence_info = {}

      # use the full_text of the text here to parse, the text[j is clean
      item = list_sentences_clean[i].strip()

      item_2 = list_sentences_raw[i].strip()

      if item != "":                      # split leaves an empty string at the end of the list and by adding a . , we get a "." item at the end of the list
        item = item + "."
        dict_sentence_info["response_{}".format(j)] = item
        clean_sentences.append(dict_sentence_info)


        for marker in special_markers_list:
          if marker in item_2:
            dict_sentence_info["special_marker"] = marker

      j = j + 1

      i = i + 1

    dict_responses["sentences"] = clean_sentences
        # in case we need to double-check

    list_responses.append(dict_responses)

  list_responses.append(additional_info)

  return list_responses

def get_list_figure_responses(og_dict):

    responses_list = []

    for figure in og_dict[:-1]:

        figure_sents = figure["sentences"]

        for item in figure_sents:

            item.pop("special_marker", None)
        responses_list.append(figure_sents)

    return responses_list

def clean_dict(responses_fig_list):

    '''merges the dictionary for each the responses of a figure into one response
    to go from a list of dicts [{response: blablabla},  {response: blablabla}, {response: blablabla}]
    to just one dict{response_1: , response_2: , response_3: }

    '''

    new_dict = {}
    for dictionary in responses_fig_list:
            for k , v in dictionary.items():
                # print(k,v)
                new_dict[k]= v
    return new_dict

def transform_dictionary_to_figure_list(og_dictionary):
    '''this function transforms the first dictionary
    into individual response dictionaries for each figure and returns it in list form.
    '''

    responses_list = get_list_figure_responses(og_dictionary)

    clean_fig_1 = clean_dict(responses_list[0])
    clean_fig_2 = clean_dict(responses_list[1])
    clean_fig_3 = clean_dict(responses_list[2])
    clean_fig_4 = clean_dict(responses_list[3])
    clean_fig_5 = clean_dict(responses_list[4])
    clean_fig_6 = clean_dict(responses_list[5])
    clean_fig_7 = clean_dict(responses_list[6])
    clean_fig_8 = clean_dict(responses_list[7])
    clean_fig_9 = clean_dict(responses_list[8])
    clean_fig_10 = clean_dict(responses_list[9])

    list_dicts = [clean_fig_1, clean_fig_2, clean_fig_3, clean_fig_4, clean_fig_5, clean_fig_6, clean_fig_7, clean_fig_8, clean_fig_9, clean_fig_10]

    return list_dicts

def get_noun_phrase(doc):
    """Takes a NLP spaCy doc as an input and outputs the noun phrase subtree at the highest depth within the syntax tree
    """
    # print(doc)
    list_dependencies = ["nsubj", "pobj", "attr", "agent", "csubj", "csubjpass", "npmod", "oprd", "pobj", "dative", "appos", "ROOT"]
    pos = ["NOUN", "PROPN", "NUM"]
    
    trans_dictionary = {"[": "", 
                    "]": "", 
                    ",": "",
                    ".": "",
                    "  ": "",
                    " '": "'",
                    ".":",",
                    " )": ")",
                    "( ": "("}
    
    for token in doc:
        # print(token)


        if token.dep_ in list_dependencies and token.pos_ in pos and token.pos_ != "PRON":  # remove personal pronouns?
            ancestors = [t for t in token.ancestors if t.dep_ in list_dependencies and t.pos_ != "PRON" and t.dep_ !="ROOT"]
            
            # in the above code, verify that the "ROOT" dependency acts the way we intend
            
            # print(token,"ancestors:", ancestors)
            
            len_ancestors = len(ancestors)
            # print(ancestors, len_ancestors)
            if len_ancestors > 0 :
                continue
                
            elif len_ancestors == 0 :
                
                og_ancestor = token

                final_children = [t for t in og_ancestor.children]
                
                for child in final_children:
                    grand_children = [t for t in child.children] # this way we get all the tree under the og ancestor
                    for item in grand_children:
                        if item != None:
                            final_children.append(item)
                            
                final_children = final_children + [og_ancestor]  
                
                sorted_sentence = sorted(final_children) # if needed, we can access the token's index with token.i
                response = str(sorted_sentence).replace("[", "").replace("]", "").replace(",", "").replace("  ", " ").replace(".", "").replace(" )", ")").replace("( ", "(").strip(" ") #to clean the response
                # print(response)
                return response
                # sometimes the response will be empty
    return "no meaningful NP found"

def get_np(sentence):
    
    doc = nlp(sentence)

    coordination = [(token, token.dep_, token.i) for token in doc if token.dep_ == "cc"]
    
    if len(coordination) > 0 :  # slice the sentence by a token and add another response to the list
        
        # print(coordination)
        list_indexes = []
        
        for tuple in coordination:  # if there are more than one instances of coordination we'll need access to the indexes of every cc to know where to cut the sentences
            index_cut = tuple[2]
            list_indexes.append(index_cut)
            
        new_sentences = []
        
        for i in range(len(list_indexes)):
            
            first_new_sentence = doc[:list_indexes[i]].text
            try:
                second_new_sentence = doc[list_indexes[i]+1:list_indexes[i+1]].text # will execute if there are more than one "cc"s
            except:
                second_new_sentence = doc[list_indexes[i]+1:].text
            
            new_sentences.append(first_new_sentence)
            new_sentences.append(second_new_sentence)
            
        # print(new_sentences)
        
        for i in range(len(new_sentences)):

            doc = nlp(new_sentences[i])
            # print(doc)
            response_chopped_sentence = get_noun_phrase(doc)
            
            # print(response_chopped_sentence)
            
            new_sentences[i] = response_chopped_sentence
        
        coordination = True
        # print(new_sentences)
        return new_sentences, coordination

    elif len(coordination) == 0:
        # print(doc)
        response = get_noun_phrase(doc)
        
        coordination = False
        
        return response, coordination

def preprocess_text_for_transformer(text):
    
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    x_array = embeddings_model.encode(text, convert_to_numpy=True)
    
    x_centroid = np.mean(x_array)
    X_transformers = x_centroid.reshape(-1,1)

    
    return X_transformers[0]

def evaluate_one_vs_rest_transformer(path, text):
    
    pipeline = pickle.load(open(path, "rb"))
    
    if "content" in path:
        # print("content found")
        possible_outcomes = ['(A)', '(Ad)', '(H)', '(Hd)', 'A', 'Abs', 'Ad', 'Alim', 'Anat', 'Art',
       'Bot', 'Elem', 'Frag', 'Ge', 'H', 'Hd', 'Id', 'Nat', 'Obj', 'Pays', 'Radio', 'Sc', 'Sex', 'Sg', 'Vet']
        
    elif "determinant" in path:
        # print("determinant found")
        possible_outcomes = ['C', 'C\'', 'C\'F', 'CF', 'E', 'EF', 'F', 'FC', 'FC\'', 'FE', 'K', 'kan']

    text_transformed = preprocess_text_for_transformer(text)
    
    prediction = pipeline.predict([text_transformed])
    probabilities = pipeline.predict_proba([text_transformed]) # sometimes no prediction is given back so we can take the outcome with the highest P instead

    # print("prediction:", prediction)
    # print("probabilities:", probabilities)
    
    list_predictions = prediction.tolist()
    list_predictions = [x for sublist in list_predictions for x in sublist] # avoid lists with sublists

    
    if len(list_predictions) != len(possible_outcomes): # sanity check
        print(prediction)
        print( len(list_predictions)  )
        print(possible_outcomes)
        print( len(possible_outcomes)  )
        print("Error encountered in the predictions")
        
    results = ([possible_outcomes[i] for i in range(len(list_predictions)) if list_predictions[i] == 1]) 

    if results == []:
        # print("No result")
        i = probabilities.argmax(1).item()
        # print(ix)
        final_results = possible_outcomes[i]
    
    else:
        final_results = str(results).replace("\'", "").replace("[", "").replace("]", "")
    
    return final_results

# ONLY CHANGE IS IN THIS FX

def evaluation_list_dicts(list_dicts, model_contents=r"sentence_transformer_contents_V23-18-04.sav", model_determinants=r"sentence_transformer_determinants_V23-18-04.sav"):
    """This function runs the evaluation with our first two models. 
    It takes as input the list of dictionary responses, prints the evaluation 
    and returns the content and determinant labels for each response in dictionary form.   
    """
    st.write("Starting evaluation")
    
    # f = open("evaluation_report.txt", "a")
    
    evaluation = []
    i = 1
    for dictionary in list_dicts:
        
        figure_number = "Figure_{}".format(i)
        
        st.write("*Evaluating* *Figure* *{}* ... :male-teacher: ".format(i))
        
        dict_evaluation_per_figure = {}
        
        # print(figure_number, "\n") 
        # st.write(figure_number)
        
        list_evaluation_figure = []
        
        j = 1
        
        for response in dictionary:
            
            sentence = dictionary[response]
            # print(dictionary[response]) #works until here

            # print(response, content, determinant)
            
            dict_eval = {}
            dict_eval["response"] = sentence
            noun_phrases, coordination = get_np(sentence)
            dict_eval["noun_phrase"] = noun_phrases

            # print("Response {}: ".format(j), sentence)
            # st.write("Response {}: {} \n".format(j, sentence))
            
            if noun_phrases == "no meaningful NP found":
                # contents = "None"
                # determinants = "None"
                dict_eval["content"] =  "None"
                dict_eval["determinant"] =  "None"
                dict_eval["coordination"] = coordination
            
            elif coordination == True:
                
                # print("Coordination Found!")
                # st.write("Coordination Found!")
                
                contents = []
                determinants = []
                
                
                for np in noun_phrases:
                    
                    content = evaluate_one_vs_rest_transformer(model_contents, np)
                    contents.append((content, np))
                    
                    determinant = evaluate_one_vs_rest_transformer(model_determinants, np)
                    determinants.append((determinant, np))
                    
                    dict_eval["content"] = contents
                    dict_eval["determinant"] = determinants
                    
                dict_eval["coordination"] = coordination
                    
                # print(response, content)
            elif coordination == False:
                # print(noun_phrases)
                contents = []
                determinants = []
                contents = evaluate_one_vs_rest_transformer(model_contents, noun_phrases)
                determinants = evaluate_one_vs_rest_transformer(model_determinants, noun_phrases)
                
                dict_eval["content"] = contents
                dict_eval["determinant"] = determinants
                dict_eval["coordination"] = coordination
                
            # dict_eval["position"] = None
            # dict_eval["colour"] = None
            
            list_evaluation_figure.append(dict_eval)
            
            # print("\nNoun Phrase(s):", noun_phrases, "\nContent:", contents , "\nDeterminant:", determinants, "\n")
            
            # st.write("Noun Phrase(s):{} \n ".format(str(noun_phrases)))
            # st.write("Content(s):{} \n ".format(str(contents)))
            # st.write("Determinant:{} \n ".format(str(determinants)))
            
            j = 1 + j
            
        dict_evaluation_per_figure[figure_number] = list_evaluation_figure
        
        evaluation.append(dict_evaluation_per_figure)
        
        i = i +1
        
    st.write("Evaluation Finished :exclamation:")
    
    return evaluation

def get_frame(list_dicts):

    final_df = pd.DataFrame()

    for figure_dict in list_dicts:
        # print(figure_dict)
        # print(type(figure_dict))
        for figure in figure_dict:
            response_dict = figure_dict[figure] # list of responses 
            # print(response_dict)
            for individual_response in response_dict:
                # print(individual_response)
                try:
                    del individual_response["coordination"]
                except:
                    coord_status = "deleted"
                individual_response["figure"] = figure
                # if we dont have consistent scalar values, it will crash
                try:
                    response_data = pd.DataFrame(individual_response)
                except:
                    for k, v in individual_response.items():
                        # print(k, v)
                        individual_response[k] = [v]
                    response_data = pd.DataFrame(individual_response)
                final_df = pd.concat([final_df, response_data])
    final_df = final_df.reset_index()
    
    return final_df

def raw_text_response_eval(raw_text):
    
    responses = get_responses(raw_text)

    list_dicts = transform_dictionary_to_figure_list(responses)

    evaluation_dict = evaluation_list_dicts(list_dicts)
    
    results_dataframe = get_frame(evaluation_dict)
    
    # results_dataframe.to_csv("RorschIA_results.csv")
    
    return results_dataframe, evaluation_dict
