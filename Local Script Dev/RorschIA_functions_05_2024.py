import pandas as pd
import numpy as np
import re
import json
import spacy
import deepl
import spacy
import langdetect
import transformers
import torch

from transformers import BertModel, BertTokenizer

import numpy as np 
import shutil
from itertools import compress

# Needed for the NLP models

nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classifications = ["individual_determinants", "macro_determinants", "individual_contents","macro_contents"]


######################---CLASSIFICATION FX

def classification_info(classification):
    
    info_dictionary = {
        "individual_determinants":{"labels":['C', "C'", "C'F", 'CF', "CF'", 'CLOB', 'CLOBF', 
                                             'E', 'EF', 'F', 'FC', "FC'", 'FCLOB', 'FE', 'K', 
                                             'KAN', 'KOB', 'KP'],
                          "path":"best_model_RorschIA_individual_determinants.pt"},
        
        "macro_determinants":{"labels":['color_sum', 'threat_sum', 'fading_sum', 
                                        'form_sum', 'kinesthetics_sum'],
                          "path":"best_model_RorschIA_macro_determinants.pt"},
        
        "individual_contents":{"labels":['(A)', '(AD)', '(H)', '(HD)', 'A', 'ABS', 'AD', 
                                         'ALIM', 'ANAT', 'ARCH', 'ART', 'BOT', 'ELEM', 
                                         'FRAG', 'GÉO', 'H', 'HD', 'MQ', 'NAT', 'OBJ', 
                                         'PAYS', 'RADIO', 'SC', 'SCÈNE', 'SEX', 'SG', 'VÊT'],
                          "path":"best_model_RorschIA_individual_contents.pt"},
        
        "macro_contents":{"labels":['animal_sum', 'human_sum', 'abs_sum', 
                                    'food_sum', 'art_arch_sum', 'nature_sum', 
                                    'fragment_sum', 'geo_sum', 'object_sum', 
                                    'science_sum', 'graphic_sum'],
                          "path":"best_model_RorschIA_macro_contents.pt"}
        }
    labels = info_dictionary[classification]["labels"]
    path = info_dictionary[classification]["path"]
    
    return labels, path

def determine_model_class(labels, path):
    n_labels = len(labels)
    class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            # self.bert_model = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True, problem_type="multi_label_classification")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased", return_dict=True, problem_type="multi_label_classification")

            self.dropout = torch.nn.Dropout(0.3) 
            self.linear = torch.nn.Linear(768, n_labels) # have to change he n of possible labels here

        def forward(self, input_ids, attn_mask, token_type_ids):
            output = self.bert_model(
                input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids
            )

            output_dropout = self.dropout(output.pooler_output)
            # print(output_dropout)
            output = self.linear(output_dropout)
            return output
        
    model = BERTClass()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    return model, device

def preprocess_text(text):

    encodings = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
)
    return encodings

def predict_prob(model, device, encodings):
    model.eval()

    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        input_ids = encodings['input_ids'].to(device, dtype=torch.long)
        attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)

        output = model(input_ids, attention_mask, token_type_ids)

        # prob_outputs = outputs.cpu().detach().numpy().tolist()
        probs = sigmoid(torch.Tensor(output)).cpu().detach().numpy().tolist()

    # final probabilities
    probs_unfolded = [prob for probs_list in probs for prob in probs_list]
    return probs_unfolded

def y_boolean(probabilities, classification):
    probabilities_array = np.array(probabilities)
    y_pred = np.zeros(probabilities_array.shape)
    y_pred[np.where(probabilities_array>=0.5)] = 1
    if "individual" in classification:
        class_type = classification.split("_")
        name_class = "canonical_" + class_type[1]
    else:
        name_class = classification
    if 1 not in y_pred:
        idx = np.argmax(probabilities_array)
        best_pred = probabilities_array[idx]
        y_pred[idx] = 1
        print("Low confidence prediction for {}".format(name_class))
    y_bool = y_pred.astype(bool)
    return y_bool

def bool_to_label(y_bool, labels):
    
    label_output = list(compress(labels, y_bool))
    return label_output

def evaluate_text(classification, text):
    
    labels, path = classification_info(classification)
    model, device = determine_model_class(labels, path)
    encodings = preprocess_text(text)
    probabilities = predict_prob(model, device, encodings)
    y_bool = y_boolean(probabilities, classification)
    label_output = bool_to_label(y_bool, labels)
    
    return label_output

##################################################3------PROCESSING THE PROTOCOL AND APPLYING THE MODELS

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
    language = langdetect.detect(raw_text)
    
    if language == "fr":
      with open(r"DEEPL_KEY\DEEPL_API_KEY.txt", "r") as f:
        API_KEY = f.read()
      # print("API Key Found!")
      translator = deepl.Translator(API_KEY)
      result = translator.translate_text(text_i[j], target_lang="EN-US", preserve_formatting=True)
      text_i[j] = result.text

    dict_responses["figure_number"] = number
    dict_responses["raw_response"] = full_text
    dict_responses["clean_response"] = text_i[j]

    doc = nlp(text_i[j])
    list_sentences_clean = [sent for sent in doc.sents]
    clean_sentences = []

    special_markers_list = ["@", "^", "V ", "<", ">"]
    # structuring the sentences inside the while loop
    i = 0
    while i < len(list_sentences_clean):

      if "Choix" in list_sentences_clean[i]:  # if we reach choix, there are no more actual responses by the patient, those are just comments by the psychologist
        break
      elif "Choice" in list_sentences_clean[i]:  # if we reach choix, there are no more actual responses by the patient, those are just comments by the psychologist
        break

      j = i + 1 # for display purposes

      dict_sentence_info = {}
      dict_sentence_info["response"] = list_sentences_clean[i]
      dict_sentence_info["response_number"] = j
      clean_sentences.append(dict_sentence_info)

      list_special_markers = []
      for marker in special_markers_list:
        if marker in list_sentences_raw[i]:
          list_special_markers.append(marker)
          
      dict_sentence_info["special_marker"] = list_special_markers

      j = j + 1
      i = i + 1

    dict_responses["sentences"] = clean_sentences
    print("Figure", dict_responses["figure_number"])
    for dictionary in dict_responses["sentences"]:
      # print(dictionary)
      list_keys = list(dictionary.keys())
      # print(list_keys)
      for key in list_keys:
        if "response" == key:
          sentence = dictionary[key]
          print("\nResponse {}: ".format(dictionary["response_number"]), str(sentence))
          print("\nSpecial markers {}: ".format(key), dictionary["special_marker"])
          
          # applying the classification function here 
          
          for model in classifications:
              classification = evaluate_text(model, str(sentence))
              # print(classification)
              dictionary[model] = classification

          print("\nCanonical Contents:", dictionary["individual_contents"], "\nMacro Contents:", dictionary["macro_contents"])
          print("Canonical Determinants:", dictionary["individual_determinants"], "\nMacro Determinants:", dictionary["macro_determinants"], "\n")            

    list_responses.append(dict_responses)

  list_responses.append(additional_info)



  return list_responses

###############################---------------FUNCTIONS TO ORGANIZE THE RESULTING DICT INTO A DF

def clean_lists_evaluations(evaluation):
    list_keys = list(evaluation.keys())
    for key in list_keys:
        val = evaluation[key]
        if type(val) != str:
            clean = str(val).replace('\'', "").replace("[", "").replace("]", "")
            evaluation[key] = clean   
    return evaluation

def get_frame(list_dicts):
    """Turn the result dictionary into a DataFrame  
    """

    final_df = pd.DataFrame()

    for figure_dict in list_dicts:
        # print(figure_dict)
        # print(type(figure_dict))
        # print(figure_dict.keys())
        try:
            figure_n = figure_dict["figure_number"]
            list_sentences = figure_dict["sentences"]
        except:
            continue
        for sentence in list_sentences:
            # print(individual_response)
            individual_response = clean_lists_evaluations(sentence)
            individual_response["figure"] = figure_n
            # if we dont have consistent scalar values, it will crash
            response_data = pd.DataFrame(individual_response, index=[0])
            final_df = pd.concat([final_df, response_data])
    final_df = final_df.reset_index()
    
    return final_df

##########################----------------FINAL WORKFLOW

def raw_text_response_eval(raw_text):
    
    responses = get_responses(raw_text)
    
    results_dataframe = get_frame(responses)
    
    results_dataframe.to_csv("RorschIA_results.csv")
    
    return results_dataframe