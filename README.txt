RorschIA Project

Web App is hosted in https://huggingface.co/spaces/alberto-lorente/RorschIA , 
Repo in https://huggingface.co/spaces/alberto-lorente/RorschIA/tree/main :)

Organization of the Repository

The datasets are in the Dataset folders, with one folder dedicated to the exploration, celeaning and storing of each dataset used.

Machine Learning Models contains the development for the both types of models as well as the analysis of their results. The final models themselves are accesible in the Web App repository and in https://huggingface.co/alberto-lorente . 

Finally, the RorschIA_app folder contains the code for the app, minus the models. It is a copy from the Web App repository.

RorschIA APP

The current functionalities are the following: the script parses the text, it organizes it and translates it. Then it splits the sentences based on punctuation and sentence segmentation. It runs each sentence through four bert transformers models, two for contents and another two for determinants. Finally, it prints a little report in English for each figure and response. For an example of the output you can quickly look at the Prototype notebook or you can check out the app in the link above.

* The development python notebook is included in case you are interested in having a deeper look under the hood. 

How to use the prototype app:

- Copy-paste the protocol or sentence you want to evaluate into the app textbox. 

- Press the Process button.

- You will see a little report on the app. If you passed an entire protocol, you will be able to download it at the very end of the page.

Some considerations:

- The script relies on certain regular expressions we are using to parse the text. We took those markers from the example brute protocol. Please make sure the protocols have the roman numeral numbers + / before the responses for each figure. The location of how the figure was looked at is encoded with the characters V ^ < > .
 
		For example: 

	"I/ @ ^ Une sorte de coléoptère avec des (...) 
	II/ V Deux personnages en train d’établir un contact ou un combat parce qu’il y a du sang ou une jambe qui a été coupée. (...)" and so on.

- For this prototype the final results are shown in English (since the training data was in English) but the sentence can be easily reversed to French, as they were originally.
