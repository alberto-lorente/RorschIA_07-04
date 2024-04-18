import streamlit as st
import pandas as pd

from functions import raw_text_response_eval

st.title("RorschIA")

text = st.text_input("Paste the text of your protocol :)")



process_button = st.button("Process")

if process_button == True:
    
    evaluation, list_dicts = raw_text_response_eval(text)
    
    report = st.write("REPORT")
    for figure_dict in list_dicts:
        
        # maybe hacer 3 boxes, 1 para qué figura es/imagen de la figura, otro para la columna y otra para el valor 
        
        # print(figure_dict)
        # print(type(figure_dict))
        for figure in figure_dict:
            response_dict = figure_dict[figure] # list of responses 
            # print(response_dict)
            for individual_response in response_dict:
                
                individual_response["Sentence"] = individual_response.pop("response")
                individual_response["Response"] = individual_response.pop("noun_phrase")
                individual_response["Determinant label"] = individual_response.pop("determinant")
                individual_response["Content label"] = individual_response.pop("content")
            
                for k, v in individual_response.items():
                    
                    if k !="figure":
                        text_report = st.write(k, ":", v)
                    
    # st.write("JSON Results")
    # st.write(list_dicts)
    
    csv = evaluation.to_csv()
    
    st.download_button(label='Download CSV', data=csv, file_name='RorschIA_results.csv', mime='text/csv')





