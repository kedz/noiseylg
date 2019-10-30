import re

#('goodbye', 0)
#('?reqmore', 1)
#('?request', 3)
#('?select', 5)
#('suggest', 6)
#('inform_no_info', 29)
#('inform_all', 30)
#('?compare', 76)
#('?confirm', 106)
#('inform_only_match', 141)
#('inform_no_match', 143)
#('inform_count', 860)
#('recommend', 1389)
#('inform', 1432)

FIELDS = [
    "resolution",
    "family",
    "count",
    "price",
    "name",
    "pricerange",
    "screensizerange",
    "screensize",
    "accessories",
    "audio",
    "ecorating",
    "color",
    "powerconsumption",
    "hdmiport",
    "hasusbport",
]

def mr2source_inputs(mr):
    if mr["da"] == "?compare" or mr['da'] == '?select':
        return mr2source_inputs_compare(mr)

    return mr2source_inputs_normal(mr)

def mr2source_inputs_normal(mr):
    
    field_toks = ["DA"]
    value_toks = [mr["da"]] 
    inputs = [mr["da"]]

    for field in FIELDS:
        if field in mr["fields"]:
            if "lex_value" in mr["fields"][field] or mr['fields'][field]["no_lex_value"] in [None, "none"]:
                #val = mr['fields'][field]["lex_value"]
                inputs.append(field)
            else:
                val = mr['fields'][field]["no_lex_value"]
                inputs.append(field + "_" + val)
    return inputs

def mr2source_inputs_compare(mr):
    inputs = [mr["da"]]
    for field in FIELDS:
        if field in mr['fields']['item1']:
            if "lex_value" in mr["fields"]['item1'][field]:
                inputs.append(field) 
            else:
                val = mr['fields']['item1'][field]["no_lex_value"]
                inputs.append(field + "_" + val)
    for field in FIELDS:
        if field in mr['fields']['item2']:
            if "lex_value" in mr["fields"]['item2'][field]:
                inputs.append(field) 
            else:
                val = mr['fields']['item2'][field]["no_lex_value"]
                inputs.append(field + "_" + val)

    return inputs




def extract_mr(string):
    mr = {"fields": {}}

    m = re.search(r"(.*?)\((.*?)\)", string)
    da = m.groups()[0]

    mr["da"] = da

    field_values = m.groups()[1].split(";")
    
    if da == "?select":
        mr["fields"] = {
            "item{}".format(i): get_field_values([field_value])
            for i, field_value in enumerate(field_values, 1)
        }

    elif da == "?compare":
        num_items = len(field_values) // 2
        mr["fields"] = {
            "item1": get_field_values(field_values[:num_items]),
            "item2": get_field_values(field_values[num_items:])
        }
    elif da == "suggest":
        mr["fields"] = {
            "item{}".format(i): get_field_values([field_value])
            for i, field_value in enumerate(field_values, 1)
        }
    else:
        mr["fields"] = get_field_values(field_values)    

    return mr

def get_field_values(field_values):
    fields = {}
    for field_value in field_values:
        if field_value == '':
            continue
        elif "=" in field_value:
            field, value = field_value.split('=')
        else:
            field = field_value
            value = None
        
        if field == "type": 
            continue

        assert field not in fields
        fields[field] = eval("extract_mr_{}".format(field))(value)        
    return fields

def extract_mr_ecorating(value):
    if value == None:
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "lex_value": value,
        "delex_value": "__ECORATING__",
    }

def extract_mr_hasusbport(value):
    return {"no_lex_value": value}


def extract_mr_pricerange(value):
    if value == "dontcare":
        return {"no_lex_value": value}
    if value == None:
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex_value": "__PRICERANGE__",
        "lex_value": value,
    }

def extract_mr_count(value):
    if value == "dontcare":
        return {"no_lex_value": value}
    delex_value = re.sub(r'[\d\.]+', '__COUNT__', value)
    lex = re.search(r'([\d\.]+)', value).groups()[0]
    return {
        "delex": "__COUNT__",
        "delex_value": delex_value,
        "lex": lex,
        "lex_value": value,
    }

def extract_mr_screensizerange(value):
    if value is None:
        return {"no_lex_value": None}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__SCREENSIZERANGE__",
        "delex_value": "__SCREENSIZERANGE__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_name(value):
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__NAME__",
        "lex": value,
        "lex_value": value,
        "delex_value": "__NAME__",
    }

def extract_mr_price(value):
    if value == "none":
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'[\d\.]+', '__PRICE__', value)
        lex = re.search(r'([\d\.]+) ', value).groups()[0]
        return {
            "delex": "__PRICE__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_powerconsumption(value):
    if value == "none":
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'[\d\.]+', '__POWERCONSUMPTION__', value)
        lex = re.search(r'([\d\.]+) ', value).groups()[0]
        return {
            "delex": "__POWERCONSUMPTION__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_resolution(value):
    if value == "none":
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__RESOLUTION__",
        "lex": value,
        "lex_value": value,
        "delex_value": "__RESOLUTION__",
    }

def extract_mr_color(value):
    if value == "none":
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__COLOR__",
        "delex_value": "__COLOR__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_hdmiport(value):
    if value is None:
        return {"no_lex_value": None}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__HDMIPORT__",
        "delex_value": "__HDMIPORT__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_family(value):
    if value is None:
        return {"no_lex_value": value}
    if value == 'dontcare':
        return {"no_lex_value": value}
    return {
        "delex": "__FAMILY__",
        "delex_value": "__FAMILY__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_screensize(value):
    if value == "none":
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'[\d.]+', '__SCREENSIZE__', value)
        lex = re.search(r'([\d.]+) ', value).groups()[0]
        return {
            "delex": "__SCREENSIZE__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_accessories(value):
    if value == "none":
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__ACCESSORIES__",
        "delex_value": "__ACCESSORIES__",
        "lex": value,
        "lex_value": value,
    }



def extract_mr_audio(value):
    if value is None:
        return {"no_lex_value": None}
    if value == "none":
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__AUDIO__",
        "delex_value": "__AUDIO__",
        "lex": value,
        "lex_value": value,
    }


def delexicalize(text, mr):

    if mr["da"] in ["inform_no_match", "inform_count", "inform", "recommend",
                    "inform_only_match", "?confirm", "inform_all",
                    "inform_no_info"]:
        delexed = delexicalize_inform(text, mr)
    elif mr["da"] in ["?select", "suggest"]:
        delexed = delexicalize_suggest(text, mr)
    elif mr["da"] in ["?request"]:
        delexed = delexicalize_request(text, mr)
    elif mr["da"] in ["?compare"]:
#        print(text)
#        print(text.encode("utf8"))
#        print(mr)
        delexed = delexicalize_compare(text, mr)
#        print(delexed)

#        input()
    else:
        #print(mr["da"])
        #print(text)
        delexed = text

    return delexed

    if mr["da"] == "?request":
        return delexicalize_request(text, mr)
    if mr["da"] == "suggest" or mr["da"] == "?select":
        return delexicalize_suggest(text, mr)
    if mr["da"] != "?compare":
        return delexicalize_normal(text, mr)
    else:
        return delexicalize_compare(text, mr)

def delexicalize_compare(text, mr):
    
    ordered_fields1 = sorted(
        [field for field in mr["fields"]["item1"].keys() 
         if "delex_value" in mr["fields"]["item1"][field]],
        key=lambda x: len(mr["fields"]["item1"][x]["delex_value"].split()),
        reverse=True)
 
    ordered_fields2 = sorted(
        [field for field in mr["fields"]["item2"].keys() 
         if "delex_value" in mr["fields"]["item2"][field]],
        key=lambda x: len(mr["fields"]["item2"][x]["delex_value"].split()),
        reverse=True)
    
    for field in ordered_fields1:
        field_info = mr["fields"]["item1"][field]
        text = re.sub(r"( |^)" + re.escape(field_info["lex_value"]) + r'( |$)',
            r'\1' + re.sub(r"(__$|__ )", r"1\1", field_info["delex_value"]) \
                + r'\2',
            text, count=1)


    for field in ordered_fields2:
        field_info = mr["fields"]["item2"][field]
        text = re.sub(r"( |^)" + re.escape(field_info["lex_value"]) + r'( |$)',
            r'\1' + re.sub(r"(__$|__ )", r"2\1", field_info["delex_value"]) \
                + r'\2',
            text, count=1)


    return text


def delexicalize_request(text, mr):
    print(mr["fields"])

    text = re.sub(r"expensive , moderate , or cheap", "__VALLIST__", text)
    text = re.sub(r"cheap , moderate , or expensive", "__VALLIST__", text)
    text = re.sub(r"price range", "__FIELD__", text)
    text = re.sub(r"small , medium or large", "__VALLIST__", text)
    text = re.sub(r"screen size( range)?", "__FIELD__", text)
    text = re.sub(r"hdmi ports", "__FIELD__", text)

    return text

def delexicalize_suggest(text, mr):

    text = re.sub(r"a\+\+ eco rating , a c eco rating and b eco rating",
        "__VAL1__ __FIELD__ , a __VAL2__ __FIELD__ and __VAL3__ __FIELD__",
        text)
    text = re.sub(r"b eco rating , in the c eco rating , or in the a\+\+ eco rating",
        "__VAL1__ __FIELD__ , in the __VAL2__ __FIELD__ , or in the __VAL3__ __FIELD__",
        text)
    for i in range(1,len(mr["fields"]) + 1):
        field, value = list(mr["fields"]["item{}".format(i)].items())[0]
        if "lex_value" in value:
            text = re.sub(re.escape(value["lex_value"]), "__VAL{}__".format(i), text,
                          count=1)        
    if field == "family":
        field_name = "family"
        text = re.sub(field_name, "__FIELD__", text)        


    if field == "screensizerange":
        text = re.sub("screen size( range)?", "__FIELD__", text)
    text = re.sub("hdmi ?ports?", "__FIELD__", text)
    text = re.sub("eco ?rating", "__FIELD__", text)

    if "range" in field:
        field_name = field.replace("range", "")
        text = re.sub(field_name, "__FIELD__", text)        

    if field == "batteryrating":
        field_name = "battery rating"
        text = re.sub(field_name, "__FIELD__", text)        

    return text




def delexicalize_inform(text, mr):
    
    ordered_fields = sorted(
        [field for field in mr["fields"].keys() 
         if "delex_value" in mr["fields"][field]],
        key=lambda x: len(mr["fields"][x]["delex_value"].split()),
        reverse=True)
#    print(mr["fields"])

    text = re.sub(r'eco( )?rating of (a|c|b|a\+|a\+\+)',
                  r'eco\1rating of __ECORATING__',
                  text)

    for field in ordered_fields:
        field_info = mr["fields"][field]
        
        if field_info["lex_value"] == 'none':
            print(field)
            raise Exception()

        if field == "screensize":   
            patt = r'( |^|"|\')' + re.escape(field_info["lex_value"])
            repl = r"\1" + field_info["delex_value"]

        if field == "powerconsumption":   
            patt = r'( |^|"|\')' + re.escape(field_info["lex_value"]) + r"(s?)"
            repl = r"\1" + field_info["delex_value"] + r"\2"

        else:
            patt = r'( |^|"|\')' + re.escape(field_info["lex_value"]) + r'( |$|"|\')'
            repl = r"\1" + field_info["delex_value"] + r"\2"
            if field == "ecorating":
                patt += r"(eco)"
                repl += r"\3"
        text = re.sub(patt, repl, text, count=1)
    return text

def lexicalize(text, mr):

    if mr["da"] == "?request":
        return lexicalize_request(text, mr)
    if mr["da"] == "suggest" or mr["da"] == "?select":
        return lexicalize_suggest(text, mr)
    if mr["da"] != "?compare":
        return lexicalize_normal(text, mr)
    else:
        return lexicalize_compare(text, mr)

def lexicalize_normal(text, mr):
    
    for field, field_info in mr["fields"].items():
        if "delex_value" in field_info:
            text = re.sub(field_info["delex_value"], field_info["lex_value"], text, 
                          count=1)
    return text


def lexicalize_compare(text, mr):
 
    for field, field_info in mr["fields"]["item1"].items():
        if "delex_value" in field_info:
            text = re.sub(field_info["delex_value"][:-2] + "1__", 
                          field_info["lex_value"], text, 
                          count=1)
    for field, field_info in mr["fields"]["item2"].items():
        if "delex_value" in field_info:
            text = re.sub(field_info["delex_value"][:-2] + "2__", 
                          field_info["lex_value"], text, 
                          count=1)

    return text

def lexicalize_suggest(text, mr):

    for i in range(1,len(mr["fields"]) + 1):
        field, value = list(mr["fields"]["item{}".format(i)].items())[0]
        if "lex_value" in value:
            text = re.sub("__VAL{}__".format(i), value["lex_value"], text)        
    if field == "family":
        field_name = "family"
        text = re.sub("__FIELD__", field_name, text)        

   

    if "range" in field:
        field_name = field.replace("range", "")
        text = re.sub("__FIELD__", field_name, text)        


    if field == "batteryrating":
        field_name = "battery rating"
        text = re.sub("__FIELD__", "battery rating", text)        

    return text

  



def lexicalize_request(text, mr):
#    print(mr["fields"])
    if "pricerange" in mr["fields"]:
        text = re.sub(
            r"__VALLIST__", r"cheap , moderate , or expensive", text)
        text = re.sub(r"__FIELD__", r"price range", text)
    if "screensizerange" in mr["fields"]:
        text = re.sub(
            r"__VALLIST__", r"small , medium or large", text)
        text = re.sub(
            r"__FIELD__( range)?", r"screen size( range)?", text)
    if "hdmiport" in mr["fields"]:
        text = re.sub(r"hdmi ports", "__FIELD__", text)


#    if "pricerange" in mr["fields"]:
#        text = re.sub(r'__FIELD__', r'price range', text)
#        text = re.sub(r'__VALLIST__', r"cheap , moderate , or expensive", text)
#    if "family" in mr["fields"]:
#        text = re.sub(r'__FIELD__', r"product family", text)
#        text = re.sub(r'__VALLIST__', 
#                      r'satellite , satellite pro , tecra or portege', text)
#    if "batteryrating" in mr["fields"]:
#        text = re.sub(r'__FIELD__', r"battery rating", text)
#        text = re.sub(r'__VALLIST__', r"standard , good , or exceptional", 
#                      text)
#    if "driverange" in mr["fields"]:
#        text = re.sub(r'__FIELD__', r"drive range", text)
#        text = re.sub(r'__VALLIST__', r'small , medium , or large', text)
#    if "weightrange" in mr["fields"]:
#        text = re.sub(r'__FIELD__', r"weight", text, count=1)
#        text = re.sub(r'__VALLIST__', r'light , midweight , or heavy', text)
#
    return text



