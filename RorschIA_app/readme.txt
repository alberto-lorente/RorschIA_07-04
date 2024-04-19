Basic workflow:

The basic function of the app is to enter a whole protocol in French or a sentence in either French or English, print a little report  and if a protocol was passed, you can download the results in .csv format.

Some notes on the code:

To not make this excedengly long, I am going to assume you have checked out the dev_scrip notebook :)

1. Since we were using regex to parse the entire protocol in the dev version, passing a sentece would have crashed the program. To bypass this little issue, I am just using try and except: if it tries to parse the regex and fails then that means that a sentence was passed. For the posibility of dual language input, I am using langdetect: if the text is in French, it runs it through Deepl. 

2. For the case where we want to evaluate single responses in the webapp, as it was discussed previously, there is a posibility of coordinated responses appearing in a sentence. To remedy that, I am using the get_np function from the dev which returns a tuple with the actual noun_phrase as tuple[0], and whether coordination was found or not as a boolean in tuple[1]. Passing a list to the evaluation_transformer function would fail so the program checks whether tupe[1] is true (there is coordination) or not (there is no coordination) and either runs the evaluation function for the single response (case 2) or does a for loop and evaluates each response (1).

3. Every file, models and API key, needed for the streamlit page is in this folder.

