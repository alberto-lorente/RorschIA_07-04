Basic workflow:

The basic function of the app is to enter a whole protocol in French or a sentence in either French or English, print a little report  and if a protocol was passed, you can download the results in .csv format.

Some notes on the code:

To not make the explanation too long, I am going to assume you have checked out the dev_scrip notebook :)

1. Since we were using regex to parse the entire protocol in the dev version, passing a sentence would have crashed the program. To bypass this little issue, I am just using try and except: if it tries to parse the regex and fails then that means that a sentence was passed. For the possibility of French/English input, I am using langdetect: if the text is in French, it runs it through Deepl. 

2. Every file, models and API key, needed for the streamlit page is in this folder.

