Prototype for RorschIA 21-04 for French.

Newest update: basic streamlit webpage in https://rorschia.streamlit.app/ , code in the RorschIA_app folder :)

The current functionalities are the following: the script parses the text, it organizes it and translates it. Then it splits the sentences based on punctuation and sentence coordination of noun phrases. It runs each noun phrase through two RF-sentence_transformers models, one for contents and another one for determinants. Finally, it prints a little report in English for each figure and response. For an example of the output you can quickly look at the Prototype notebook or you can check out the streamlit app, https://rorschia.streamlit.app/ .

* The development python notebook is included in case you are interested in having a deeper look under the hood. 

How to use the prototype:

- For convenience, copy-paste the protocol or sentence you want to evaluate into the app textbox. 

- Press the Process button.

- You will se a little report on the app. If you passed an entire protocol, you will be able to download it at the very end of the page.

Some considerations:

- The script relies on certain regular expressions we are using to parse the text. We took those markers from the first brute protocol that was sent (I will include it in within the Dev Notebook folder). Please make sure the protocols have the roman numeral numbers + / before the responses for each figure. The location of how the figure was looked at is encoded with the characters V ^ < > .
 
		For example: 

	"I/ @ ^ Une sorte de coléoptère avec des (...) 
	II/ V Deux personnages en train d’établir un contact ou un combat parce qu’il y a du sang ou une jambe qui a été coupée. (...)" and so on.

- For this prototype the final results are shown in English (since the training data was in English) but the sentence can be easily reversed to French, as they were originally.

- As of now, the script computes Determinants and Contents. The models are still in progress due to the low quality of the data (more discussion about this in the ML Development notebook).
