import re


def name(utterance, compare=False):
    count = len(list(re.findall("__NAME__", utterance)))
    if count == 1:
        return {"lex_value": "PLACEHOLDER"}
    else:
        return None

def find_drive_range(utterance):
    slot_count = len(list(re.findall(
        "__DRIVERANGE__", utterance)))
    dontcare_count = len(list(re.findall(
        "you do not care about the drive( size)? range|any drive (size )?range", utterance)))
    if slot_count == 1 and dontcare_count == 0:
        return {"lex_value": "PLACEHOLDER"}
    elif slot_count == 0 and dontcare_count == 1:
        return {"no_lex_value": "dontcare"}
    elif slot_count >= 1 or dontcare_count >= 1:
        return -1
    else:
        return None
 
def find_price_range(utterance):
    slot_count = len(list(re.findall(
        "__PRICERANGE__", utterance)))
    dontcare_count = len(list(re.findall(
        "you do not care about the price range|in any price range", utterance)))
    if slot_count == 1 and dontcare_count == 0:
        return {"lex_value": "PLACEHOLDER"}
    elif slot_count == 0 and dontcare_count == 1:
        return {"no_lex_value": "dontcare"}
    elif slot_count >= 1 or dontcare_count >= 1:
        return -1
    else:
        return None
    
def find_weight_range(utterance):
    slot_count = len(list(re.findall(
        "__WEIGHTRANGE__", utterance)))
    dontcare_count = len(list(re.findall(
        "you do not care about the weight range|in any weight range", utterance)))
    if slot_count == 1 and dontcare_count == 0:
        return {"lex_value": "PLACEHOLDER"}
    elif slot_count == 0 and dontcare_count == 1:
        return {"no_lex_value": "dontcare"}
    elif slot_count >= 1 or dontcare_count >= 1:
        return -1
    else:
        return None
    

 
def find_battery_rating(utterance):
    slot_count = len(list(re.findall(
        "__BATTERYRATING__", utterance)))
    dontcare_count = len(list(re.findall(
        "you do not care about the battery rating|any battery rating", utterance)))
    if slot_count == 1 and dontcare_count == 0:
        return {"lex_value": "PLACEHOLDER"}
    elif slot_count == 0 and dontcare_count == 1:
        return {"no_lex_value": "dontcare"}
    elif slot_count >= 1 or dontcare_count >= 1:
        return -1
    else:
        return None
  
def find_family(utterance):
    slot_count = len(list(re.findall(
        "__FAMILY__", utterance)))
    dontcare_count = len(list(re.findall(
        r"(you )?do not care about (the )?product family|in all product family|you are not picky about the (\w+ ){0,5}(family|product)|any laptop|any product family", utterance)))
    if slot_count == 1 and dontcare_count == 0:
        return {"lex_value": "PLACEHOLDER"}
    elif slot_count == 0 and dontcare_count == 1:
        return {"no_lex_value": "dontcare"}
    elif slot_count >= 1 or dontcare_count >= 1:
        return -1
    else:
        return None
 

def find_field(utterance, field):
    if field == "driverange":
        return find_drive_range(utterance)
    elif field == "pricerange":
        return find_price_range(utterance)
    elif field == "weightrange":
        return find_weight_range(utterance)
    elif field == "batteryrating":
        return find_battery_rating(utterance)

    elif field == "family":
        return find_family(utterance)

    count = len(list(re.findall(
        "__{}__".format(field.upper()), utterance)))
    
    if count == 0:
        return None
    if count == 1:
        return {"lex_value": "PLACEHOLDER"}
    else:
        return -1

def is_for_biz(utterance):
    no_count = len(list(re.findall(
        "not (used )?for business", utterance)))
    dontcare_count = len(list(re.findall(
        "(you )?do not care (about )?whether it is (used )?for business", utterance)))
    if dontcare_count > 0:
        utterance = re.sub("(you )?do not care (about )?whether it is (used )?for business", " ", utterance)
    yes_count = len(list(re.findall
        ("(?!you do not care whether it )+(is |can be |are )(used )?for business", utterance)))
    yes_count += len(list(re.findall("available business use laptop|laptop for business computing|for a business computing laptop|^for business computing|is good for business computing", utterance)))
    if no_count == 1 and yes_count == 0 and dontcare_count == 0:
        return {"no_lex_value": "false"}
    elif no_count == 0 and yes_count == 1 and dontcare_count == 0:
        return {"no_lex_value": "true"}
    elif no_count == 0 and yes_count == 0 and dontcare_count == 1:
        return {"no_lex_value": "dontcare"}
    elif no_count >= 1 or yes_count >= 1 or dontcare_count >= 1:
        return -1
    else:
        return None

def no_info(utterance):
    fields = {}
    for term in ["battery", "dimension", "design", "drive", "memory", 
                 "platform", "price",
                 "processor", "utility", "warranty", "weight"]:
        count = len(list(re.findall(term, utterance)))
        if count == 1:
            fields[term] = {"no_lex_value": "none"}
        elif count > 1:
            return None
    return fields 


def classify_da(text):


    # inform_only_match

    if "no other laptop -s except" in text:
        return "inform_only_match"
    if "the only" in text:
        return "inform_only_match"
    if "only one laptop" in text:
        return "inform_only_match"
    if "unfortunately , apart from" in text:
        return "inform_only_match"
    if "sorry there is only this one" in text:
        return "inform_only_match"
    if "no other choice -s" in text:
        return "inform_only_match"
    if "no laptop -s other than" in text:
        return "inform_only_match"
    if "exactly one" in text:
        return "inform_only_match"
    if "we have one laptop" in text:
        return "inform_only_match"
    if "only option" in text:
        return "inform_only_match"
    if "sole product" in text:
        return "inform_only_match"

    # inform no match
    if "other than" not in text:
        if re.search("there are (no|not any) laptop -s", text):
            return "inform_no_match"

    # inform all
    if "__COUNT__" not in text:
        if "all laptop -s" in text:
            return "inform_all"
        if "we have found all" in text:
            return "inform_all"
        if re.search(r"^all (\w+ )*laptop -s", text):
            return "inform_all"
        if "all the laptop -s" in text:
            return "inform_all"
        if "all of these laptop -s" in text:
            return "inform_all"
        if "all of our available" in text:
            return "inform_all"
        if "all __PRICERANGE__ -ly priced laptop" in text:
            return "inform_all"






    # SELECT
    #if len(re.findall("1__", text)) == 1 and len(re.findall("__FIELD__", text)) == 1:
    if "do you care for something" in text:
        return "?select"
    if "sorry would you like something" in text:
        return "?select"
    if "please choose between" in text:
        return "?select"
    if "sorry would you like a something" in text:
        return "?select"
    if "would you like this product" in text:
        return "?select"
    if "are you seeking something used" in text:
        return "?select"
    if "do you select if you prefer" in text:
        return "?select"
    if "sorry , would you like" in text:
        return "?select"
    if "sorry would you like" in text:
        return "?select"
    if "do you not care about the product" in text:
        return "?select"
    if "do you care about" in text:
        return "?select"


    # REQUEST
    if " __VALLIST__" in text:
        return "?request"

    # COMPARE
    if "compare" in text:
        return "?compare"
    if len(re.findall("1__", text)) == 3 and len(re.findall("2__", text)) == 3:
        return "?compare"
    if len(re.findall("1__", text)) == 2 and len(re.findall("2__", text)) == 2:
        return "?compare"
    if "for business computing , do you prefer" in text:
        return "?compare"
    if "which would you prefer" in text:
        return "?compare"
    if "depending on your needs" in text:
        return "?compare"
    if "which would you choose" in text:
        return "?compare"
    if "what is your preference" in text:
        return "?compare"

    # CONFIRM

    if re.search(r"(right|correct)$", text):
        return "?confirm"
    if "confirming" in text:
        return "?confirm"
    if "comfirming" in text:
        return "?confirm"
    if "can we confirm" in text:
        return "?confirm"
    if "please confirm" in text:
        return "?confirm"
    if "can you confirm" in text:
        return "?confirm"
    if "may i confirm" in text:
        return "?confirm"
    if "just to verify" in text:
        return "?confirm"
    if "just to make sure" in text:
        return "?confirm"
    if "is it correct" in text:
        return "?confirm"
    if "is it true" in text:
        return "?confirm"
    if "let me confirm that you are looking" in text:
        return "?confirm"
    if "so are you saying that you" in text:
        return "?confirm"

    #if "let me confirm" in text:
    #    return "?confirm"

    if "to confirm" in text:
        return "?confirm"

    #if "confirm" in text:
    #    return "?confirm"
    #if "so you are looking" in text:
    #    return "?confirm"


    if "__COUNT__" in text:
        return "inform_count" 
    if "for example , something" in text:
        return "suggest"
    if "__VAL3__" in text:
        return "suggest"
    if "__VAL2__" in text:
        if "are you looking for something" in text:
            return "suggest"
        if "if you do not care" in text:
            return "suggest"
        if "should you like something" in text:
            return "suggest"
        if "you could get something" in text:
            return "suggest"

    if "goodbye" in text:
        return "goodbye"
    #if "thank you" in text:
    #    return "goodbye"

    # INFORM NO INFO
    if re.search(r"i do not know anything|no info|i can not inform you|sorry i have no (\w+ ){0,3} info|there is none", text):
        return "inform_no_info"
    if "information" in text:
        if "none" in text:
            return "inform_no_info"
        if "i do not have" in text:
            return "inform_no_info"
        if "unavailable" in text:
            return "inform_no_info"
    if "sorry" in text:
        if "we do not have any info" in text:
            return "inform_no_info"
    if re.search(r"there are no results.*that match(ed)? your query", text):
        return "inform_no_info"
    if "system does not have information" in text:
        return "inform_no_info"
    if re.search(r"i am sorry (but )?(i|we) (have no|do not have) (\w+ )?information", text):
        return "inform_no_info"
    if "sorry the information could not be found" in text:
        return "inform_no_info"
    if "now information" in text:
        return "inform_no_info"
    #if re.search(r"this (\w+ )+is none ", text):
    #    return "inform_no_info"
    #if re.search(r"\w+ none , \w+ none", text):
    ##    return "inform_no_info"

   
    if "recommend" in text:
        return "recommend" 
#    if re.search(r"^(the )?__NAME__ is (a|an) ((nice|great|good|fineexceptional) )?laptop", text):
#        return "recommend"
    #if re.search(r"line is a quality laptop", text):
    #    return "recommend"
    if re.search("__NAME__ is (a|an) (\w+ ){0,4}laptop", text):
        return "inform"
    if re.search(r"__NAME__ is (a |an )(__\w+__ )?laptop", text):
        return "inform"
    if re.search(r"^(the )?(__\w+__ )?__NAME__ laptop (is|has a|runs on)", text):
        return "inform"

    if re.search(r"another laptop (in|from) the", text):
        return "inform"
   
    if len(re.findall("__NAME__", text)) == 1 and len(re.findall(r"__[^N ].+?__", text)) in [1, 2,3]:
        return "inform"
    if len(re.findall("__NAME__", text)) == 1 and "business" in text:
        return "inform"

    return "N/A"

def text_da2mr(text, da):
    if da == "inform":
        return text2mr_inform(text)
    if da == "inform_no_match":
        return text2mr_inform_no_match(text)
    if da == "inform_count":
        return text2mr_inform_count(text)
    if da == "?select":
        return text2mr_select(text)
    else:
        raise Exception()

VALUES = ["battery", "batteryrating", "design", "dimension", "drive",
          "driverange", "family", "memory",
          "platform", "price", "pricerange", "processor", "utility",
          "warranty", "weight", "weightrange", "isforbusinesscomputing"]


def text2mr_select(text, field):

    fields = {}
    if "__VAL1__" in text:
        fields["item1"] = {field: {"no_lex_value": "PLACEHOLDER"}}
    if "__VAL2__" in text:
        fields["item2"] = {field: {"no_lex_value": "PLACEHOLDER"}}

    for m in re.findall("or you do not care", text):
        if "item1" not in fields:
            fields["item1"] = {field: {"no_lex_value": "dontcare"}}
        elif 'item2' not in fields:
            fields["item2"] = {field: {"no_lex_value": "dontcare"}}
        else:
            return None
    for m in re.findall("which is not for business computing", text):
        if "item1" not in fields:
            fields["item1"] = {"isforbusinesscomputing": {"no_lex_value": "false"}}
        elif 'item2' not in fields:
            fields["item2"] = {"isforbusinesscomputing": {"no_lex_value": "false"}}
        else:
            return None

    for m in re.findall("which is for business computing", text):
        if "item1" not in fields:
            fields["item1"] = {"isforbusinesscomputing": {"no_lex_value": "true"}}
        elif 'item2' not in fields:
            fields["item2"] = {"isforbusinesscomputing": {"no_lex_value": "true"}}
        else:
            return None

    if "which is for business computing or which is for business computing" in text:
       fields = {"item1": {"isforbusinesscomputing": {"no_lex_value": "true"}},
                 "item2": {"isforbusinesscomputing": {"no_lex_value": "true"}}}
   
    if "which is or is not for business computing" in text:
       fields = {"item1": {"isforbusinesscomputing": {"no_lex_value": "dontcare"}},
                 "item2": {"isforbusinesscomputing": {"no_lex_value": "false"}}
       } 
    mr = {"da": "?select", "fields": fields}
    return mr

    matches = []
    for slot in re.iterfind(r"__(.*?)__)", text):
        field = slot.groups()[0].lower()
        matches.append(
            (slot.start(), {field: {"lex_value": "PLACEHOLDER"}}))
        
    matches.sort(key=lambda x: x[0])

    mr = {"da": "?select", "fields": {}}
    for i, match in enumerate(matches, 1):
        mr["fields"]["item{}".format(i)] = match[1]

    return mr

def text2mr_inform_count(text):
    fields = {}
    for field in VALUES + ["count"]:
        val = find_field(text, field)
        if val == -1:
            return None
        elif val is not None:
            fields[field] = val
    biz = is_for_biz(text)
    if biz == -1:
        return None
    if biz is not None:
        fields["isforbusinesscomputing"] = biz
    return {"da": "inform_count", "fields": fields}

def text2mr_inform(text):
    fields = {}
    for field in VALUES + ["name"]:
        val = find_field(text, field)
        if val == -1:
            return None
        elif val is not None:
            fields[field] = val
    biz = is_for_biz(text)
    if biz == -1:
        return None
    if biz is not None:
        fields["isforbusinesscomputing"] = biz

    return {"da": "inform", "fields": fields}

def text2mr_inform_no_match(text):
    fields = {}
    for field in VALUES:
        val = find_field(text, field)
        if val == -1:
            return None
        elif val is not None:
            fields[field] = val
    biz = is_for_biz(text)
    if biz == -1:
        return None
    if biz is not None:
        fields["isforbusinesscomputing"] = biz

    return {"da": "inform", "fields": fields}


def text2mr_compare(text):
    def find_business(text):

        matches = []
        
        no_patt = "not (used )?for business"
        for match in re.finditer(no_patt, text):
            matches.append((match.start(), "false"))

        yes_patt = "(?!you do not care whether it )(is |can be |are )(used )?for business"
        for match in re.finditer(yes_patt, text):
            matches.append((match.start(), "true"))

        if len(matches) == 0:
            return None
        if len(matches) != 2:
            return -1
        matches.sort(key=lambda x: x[0])
        
        return matches[0][1], matches[1][1]


    mr = {"da": "?compare", "fields": {"item1": {}, "item2": {}}}

    for slot in re.findall(r"(__.*?__)", text):
        m = re.match(r"__(.*?)(\d)__", slot)
        if not m:
            return None
        slot = m.groups()[0].lower()
        slot_num = m.groups()[1]
        mr["fields"]["item" + slot_num][slot] = {"lex_value": "PLACEHOLDER"}
   
    biz = find_business(text) 
    if biz == -1:
        return None
    if biz is not None:
        mr["fields"]["item1"]["isforbusinesscomputing"] = {
            "no_lex_value": biz[0]}
        mr["fields"]["item2"]["isforbusinesscomputing"] = {
            "no_lex_value": biz[1]}
    len1 = len(mr["fields"]["item1"])
    len2 = len(mr["fields"]["item2"])
    if len1 != len2:
        return None
    if len1 not in [2,3]:
        return None
    for key in mr["fields"]["item1"]:
        if key not in mr["fields"]["item2"]:
            return None

    return mr


