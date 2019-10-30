import re


FIELDS = ["name", "memory", "isforbusinesscomputing", "driverange", "platform",
          "dimension", "batteryrating", "family", "drive", "design", 
          "utility", "pricerange", "processor", "battery", "weightrange",
          "warranty", "price", "weight", "count",]


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

def extract_mr_name(value):
    return {
        "delex": "__NAME__",
        "lex": value,
        "lex_value": value,
        "delex_value": "__NAME__",
    }

def extract_mr_memory(value):
    if value == "none":
        return {"no_lex_value": "none"}
    else:
        delex_value = re.sub(r'\d+', '__MEMORY__', value)
        lex = re.search(r'(\d+) gb', value).groups()[0]
        return {
            "delex": "__MEMORY__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_isforbusinesscomputing(value):
    return {"no_lex_value": value}

def extract_mr_driverange(value):
    
    if value == "dontcare":
        return {"no_lex_value": value}
    if value is None:
        return {"no_lex_value": value}
    else:
        return {
            "delex": "__DRIVERANGE__",
            "delex_value": "__DRIVERANGE__",
            "lex": value,
            "lex_value": value,
        }

def extract_mr_platform(value):
    if value == "none":
        return {"no_lex_value": value}
    return {
        "delex": "__PLATFORM__",
        "delex_value": "__PLATFORM__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_dimension(value):
    if value == "none":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'[\d.]+', '__DIMENSION__', value)
        lex = re.search(r'([\d.]+) ', value).groups()[0]
        return {
            "delex": "__DIMENSION__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_batteryrating(value):

    if value is None:
        return {"no_lex_value": value}
    if value == 'dontcare':
        return {"no_lex_value": value}

    else:
        return {
            "delex": "__BATTERYRATING__",
            "delex_value": "__BATTERYRATING__",
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


def extract_mr_drive(value):
    if value == "none":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'\d+', '__DRIVE__', value)
        lex = re.search(r'(\d+) (g|t)b', value).groups()[0]
        return {
            "delex": "__DRIVE__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_design(value):
    if value == "none":
        return {"no_lex_value": value}
    return {
        "delex": "__DESIGN__",
        "delex_value": "__DESIGN__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_utility(value):
    if value == "none":
        return {"no_lex_value": value}
    return {
        "delex": "__UTILITY__",
        "delex_value": "__UTILITY__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_pricerange(value):
    if value == None:
        return {"no_lex_value": value}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__PRICERANGE__",
        "delex_value": "__PRICERANGE__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_processor(value):
    if value == "none":
        return {"no_lex_value": value}
    return {
        "delex": "__PROCESSOR__",
        "delex_value": "__PROCESSOR__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_battery(value):
    if value == "none":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'[\d\.]+', '__BATTERY__', value)
        lex = re.search(r'([\d\.]+) hour', value).groups()[0]
        return {
            "delex": "__BATTERY__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_weightrange(value):
    if value is None:
        return {"no_lex_value": None}
    if value == "dontcare":
        return {"no_lex_value": value}
    return {
        "delex": "__WEIGHTRANGE__",
        "delex_value": "__WEIGHTRANGE__",
        "lex": value,
        "lex_value": value,
    }

def extract_mr_warranty(value):
    if value == "none":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'[\d\.]+', '__WARRANTY__', value)
        lex = re.search(r'([\d\.]+) ', value).groups()[0]
        return {
            "delex": "__WARRANTY__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_price(value):
    if value == "none":
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

def extract_mr_weight(value):
    if value == "none":
        return {"no_lex_value": value}
    else:
        delex_value = re.sub(r'[\d\.]+', '__WEIGHT__', value)
        lex = re.search(r'([\d\.]+) ', value).groups()[0]
        return {
            "delex": "__WEIGHT__",
            "delex_value": delex_value,
            "lex": lex,
            "lex_value": value,
        }

def extract_mr_count(value):
    delex_value = re.sub(r'[\d\.]+', '__COUNT__', value)
    lex = re.search(r'([\d\.]+)', value).groups()[0]
    return {
        "delex": "__COUNT__",
        "delex_value": delex_value,
        "lex": lex,
        "lex_value": value,
    }

def delexicalize(text, mr):
    if mr["da"] == "?request":
        return delexicalize_request(text, mr)
    if mr["da"] == "suggest" or mr["da"] == "?select":
        return delexicalize_suggest(text, mr)
    if mr["da"] != "?compare":
        return delexicalize_normal(text, mr)
    else:
        return delexicalize_compare(text, mr)

def delexicalize_request(text, mr):
    if "pricerange" in mr["fields"]:
        text = re.sub(r"price range", r'__FIELD__', text)
        text = re.sub(r"cheap , moderate , or expensive", r"__VALLIST__", text)
    if "family" in mr["fields"]:
        text = re.sub(r"product family", r'__FIELD__', text)
        text = re.sub(r'satellite , satellite pro , tecra or portege', 
                      r'__VALLIST__', text)
        text = re.sub(r'tecra , portege , satellite , or satellite pro', 
                      r'__VALLIST__', text)
    if "batteryrating" in mr["fields"]:
        text = re.sub(r"battery rating", r'__FIELD__', text)
        text = re.sub(r"standard , good , or exceptional", 
                      r'__VALLIST__', text)
    if "driverange" in mr["fields"]:
        text = re.sub(r"drive range", r'__FIELD__', text)
        text = re.sub(r'small , medium , or large', r'__VALLIST__', text)
        text = re.sub(r'large , medium , or small', r'__VALLIST__', text)
    if "weightrange" in mr["fields"]:
        text = re.sub(r"weight", r'__FIELD__', text, count=1)
        text = re.sub(r'light , midweight , or heavy', r'__VALLIST__', text)
        text = re.sub(r'heavy , light , or mid weight', r'__VALLIST__', text)


    return text

def lexicalize_request(text, mr):
    if "pricerange" in mr["fields"]:
        text = re.sub(r'__FIELD__', r'price range', text)
        text = re.sub(r'__VALLIST__', r"cheap , moderate , or expensive", text)
    if "family" in mr["fields"]:
        text = re.sub(r'__FIELD__', r"product family", text)
        text = re.sub(r'__VALLIST__', 
                      r'satellite , satellite pro , tecra or portege', text)
    if "batteryrating" in mr["fields"]:
        text = re.sub(r'__FIELD__', r"battery rating", text)
        text = re.sub(r'__VALLIST__', r"standard , good , or exceptional", 
                      text)
    if "driverange" in mr["fields"]:
        text = re.sub(r'__FIELD__', r"drive range", text)
        text = re.sub(r'__VALLIST__', r'small , medium , or large', text)
    if "weightrange" in mr["fields"]:
        text = re.sub(r'__FIELD__', r"weight", text, count=1)
        text = re.sub(r'__VALLIST__', r'light , midweight , or heavy', text)

    return text




def delexicalize_suggest(text, mr):

    for i in range(1,len(mr["fields"]) + 1):
        field, value = list(mr["fields"]["item{}".format(i)].items())[0]
        if "lex_value" in value:
            text = re.sub(value["lex_value"], "__VAL{}__".format(i), text,
                          count=1)        
    if field == "family":
        field_name = "family"
        text = re.sub(field_name, "__FIELD__", text)        

    if "range" in field:
        field_name = field.replace("range", "")
        text = re.sub(field_name, "__FIELD__", text)        

    if field == "batteryrating":
        field_name = "battery rating"
        text = re.sub(field_name, "__FIELD__", text)        

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




def delexicalize_normal(text, mr):
    
    ordered_fields = sorted(
        [field for field in mr["fields"].keys() 
         if "delex_value" in mr["fields"][field]],
        key=lambda x: len(mr["fields"][x]["delex_value"].split()),
        reverse=True)
    
    for field in ordered_fields:
        field_info = mr["fields"][field]
        if field_info["lex_value"] == 'none':
            continue
        text = re.sub(field_info["lex_value"], field_info["delex_value"], text,
                      count=1)
    return text


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
        text = re.sub(field_info["lex_value"], 
                      re.sub(r"(__$|__ )", r"1\1", field_info["delex_value"]), 
                      text,
                      count=1)

    for field in ordered_fields2:
        field_info = mr["fields"]["item2"][field]
        text = re.sub(field_info["lex_value"], 
                      re.sub(r"(__$|__ )", r"2\1", field_info["delex_value"]),
                      text,
                      count=1)


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
            text = re.sub(field_info["delex"], field_info["lex"], text, 
                          count=1)
    return text


def lexicalize_compare(text, mr):
 
    for field, field_info in mr["fields"]["item1"].items():
        if "delex_value" in field_info:
            text = re.sub(field_info["delex"][:-2] + "1__", 
                          field_info["lex"], text, 
                          count=1)
    for field, field_info in mr["fields"]["item2"].items():
        if "delex_value" in field_info:
            text = re.sub(field_info["delex"][:-2] + "2__", 
                          field_info["lex"], text, 
                          count=1)

    return text

   



