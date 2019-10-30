import re


FIELDS = ["eat_type", "near", "area", "family_friendly", 
          "customer_rating", "price_range", "food", "name"]
FIELD_DICT = {

    "food": [
         'French',
         'Japanese',
         'Chinese',
         'English',
         'Indian',
         'Fast food',
         'Italian'
    ],
    "family_friendly": ['no', 'yes'],
    "area": ['city centre', 'riverside'],
    "near": [
         'Café Adriatic',
         'Café Sicilia',
         'Yippee Noodle Bar',
         'Café Brazil',
         'Raja Indian Cuisine',
         'Ranch',
         'Clare Hall',
         'The Bakers',
         'The Portland Arms',
         'The Sorrento',
         'All Bar One',
         'Avalon',
         'Crowne Plaza Hotel',
         'The Six Bells',
         'Rainbow Vegetarian Café',
         'Express by Holiday Inn',
         'The Rice Boat',
         'Burger King',
         'Café Rouge',
    ],
    "eat_type": ['coffee shop', 'pub', 'restaurant'],
    "customer_rating": ['3 out of 5', '5 out of 5', 'high', 
                        'average', 'low', '1 out of 5'],
    "price_range": ['more than £30', 'high', '£20-25', 'cheap', 
                    'less than £20', 'moderate'],
    "name": [
         'Cocum',
         'Midsummer House',
         'The Golden Curry',
         'The Vaults',
         'The Cricketers',
         'The Phoenix',
         'The Dumpling Tree',
         'Bibimbap House',
         'The Golden Palace',
         'Wildwood',
         'The Eagle',
         'Taste of Cambridge',
         'Clowns',
         'Strada',
         'The Mill',
         'The Waterman',
         'Green Man',
         'Browns Cambridge',
         'Cotto',
         'The Olive Grove',
         'Giraffe',
         'Zizzi',
         'Alimentum',
         'The Punter',
         'Aromi',
         'The Rice Boat',
         'Fitzbillies',
         'Loch Fyne',
         'The Cambridge Blue',
         'The Twenty Two',
         'Travellers Rest Beefeater',
         'Blue Spice',
         'The Plough',
         'The Wrestlers',
    ],


}


def name(utterance, delex=False):

    if delex:
        if "__NAME__" in utterance:
            return "__NAME__"
        else:
            return None

    for val in FIELD_DICT["name"]:
        v = val.replace("The ", "").lower()
        if re.search("(?!near )(the )?" + v, utterance):
            return val    

def near(utterance, delex=False):

    if delex:
        if "__NEAR__" in utterance:
            return "__NEAR__"
        else:
            return "N/A"

    for val in FIELD_DICT["near"]:
        v = val.replace("The ", "").lower() 
        patt = r'(near|by|close|around|next) (to )?(the )?' + v
        if re.search(patt, utterance):
            return val
        
    return "N/A"

def area(utterance, delex=False):
    
    if re.search(
            r'(near the|by the|in the) (river|riverside|water|waterfront)',
            utterance):
        return "riverside"
 
    if re.search(r'(city|centre)', utterance):
        return "city centre"

    return "N/A"

def eat_type(utterance, delex=False):
    if "pub" in utterance:
        return "pub"
    if "restaurant" in utterance:
        return "restaurant"
    if "coffee" in utterance:
        return "coffee shop" 
    return "N/A"


def food(utterance, delex=False):
    foods = ['French', 'Japanese', 'Chinese', 'English', 'Fast food', 
             'Italian']
    for food in foods:
        if food.lower() in utterance:
            return food
    if "indian" in utterance.replace("raja indian", ""):
        return "Indian"

    if "sushi" in utterance:
        return "Japanese"
    if "pasta" in utterance:
        return "Italian"
    if "british" in utterance:
        return "English" 
    if "wine" in utterance:
        return "French"
    if "fries" in utterance:
        return "Fast food"
    if "spaghetti" in utterance:
        return "Italian"
    if "fast - food" in utterance:
        return "Fast food"
    return "N/A"

def price_range(utterance, delex=False):

    if re.search("not cheap", utterance):
        return "high"
    if re.search("cheap", utterance):
        return "cheap"
    if re.search("not budget friendly", utterance):
        return "high"
    if re.search("budget", utterance):
        return "cheap"
    if re.search("low price", utterance):
        return "cheap"
    if re.search("inexpensive|not (very )?expensive", utterance):
        return "cheap"
    if re.search("expensive", utterance):
        return "high"

    if re.search("low( |-)?cost", utterance):
        return "cheap"

    if re.search("low( |-)?price", utterance):
        return "cheap"

    if re.search("price range is high", utterance):
         return "high"

    if re.search("high( |-)?price", utterance):
         return "high"

    if "20" in utterance and "25" in utterance:
         return "£20-25"
    if "20" in utterance:
         return "less than £20"
    if "30" in utterance:
        return "more than £30"
    mod_patt = "(competitive|decent|fair|mid|mid-range|reasonable|reasonabl|medium|affordable|affordab|moderate|average)(ly)?(-)? ?price"
    if re.search(mod_patt, utterance):
        return "moderate"

    if "not pricy" in utterance:
        return "moderate"


    return "N/A"

def family_friendly(utterance, delex=False):
    if re.search("is (a )?(kid|child|children|family)(-| )?friendly", utterance):
        return "yes"
    if re.search(
            "(n't|not|non)( |-)?(kid|child|children|family)(-| )?friendly",
            utterance):
        return "no"
    if "adult" in utterance:
        return "no"
    return "N/A"

def customer_rating(utterance, delex=False):
    if re.search("(1|one) (out )?of (5|five)|one star", utterance):
        return "1 out of 5"
    if re.search("(3|three) (out )?of (5|five)|three star", utterance):
        return "3 out of 5"
    if re.search("(5|five) (out )?of (5|five)|five star", utterance):
        return "5 out of 5"

    if re.search("low( |-)?(customer )?(rat|review|satisfaction)", utterance):
        return "low"
    if re.search("high(ly)?( |-)?(customer )?(rat|review|satisfaction)", utterance):
        return "high"

    if re.search("average( |-)?(customer )?(rat|review|satisfaction)", utterance):
        return "average"



    return "N/A"
