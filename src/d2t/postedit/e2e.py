import re

def detokenize(tokens):
        
    tokens = [t for t in tokens if t[0] != "<" and t[-1] != ">"]
    text = " ".join(tokens)
    text = re.sub(r" (\.|,)", r"\1", text) 
    text = re.sub(r" - ", r"-", text) 
    text = re.sub("\u00A3 ", "\u00A3", text) 
    return text


def lexicalize(text, labels):

    text = text.replace("the __NAME__", labels["name"].lower())
    text = text.replace("__NAME__", labels["name"].replace("The ", "").lower())
    
    if "near" in labels and labels["near"] != "N/A":
        text = text.replace("the __NEAR__", labels["near"].lower())
        text = text.replace("__NEAR__", labels["near"].replace("The ", "").lower())

    return text
