from Functions.RorschIA_functions_07_04 import raw_text_response_eval as eval

file_path = input("Copy and paste the path of the text doc.\n\n")

with open(file_path, "r") as f:
    text = f.read()

evaluation = eval(text)