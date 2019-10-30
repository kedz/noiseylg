import re


def name(utterance, slot_value, delex=False):
    
    if delex:
        if "__NAME__" in utterance:
            return "__NAME__"
        else:
            return "N/A"
    pattern = slot_value.replace("The ", "").lower()

    if pattern in utterance:
        return slot_value
    else:
        return "N/A"

def near(utterance, slot_value, delex=False):
    
    if delex:
        if "__NEAR__" in utterance:
            return "__NEAR__"
        else:
            return "N/A"
    pattern = slot_value.replace("The ", "").lower()

    if pattern in utterance:
        return slot_value
    else:
        return "N/A"



def area(utterance, slot_value, delex=False):

    RIVER_TERMS = ["river", "riverside", "water", "waterfront"]
    CITY_TERMS = ["city", "centre"]
    if slot_value == "riverside": 
        if any([t in utterance for t in RIVER_TERMS]):
            return "riverside"
    elif slot_value == "city centre":
        if any([t in utterance for t in CITY_TERMS]):
            return "city centre"

    return "N/A"


def eat_type(utterance, slot_value, delex=False):

    RESTAURANT_TERMS = ["restaurant"]
    COFFEE_TERMS = ["coffee", "coffee shop"]
    PUB_TERMS = ["pub"]
    if slot_value == "restaurant": 
        if any([t in utterance for t in RESTAURANT_TERMS]):
            return "restaurant"
    elif slot_value == "coffee shop":
        if any([t in utterance for t in COFFEE_TERMS]):
            return "coffee shop"
    elif slot_value == "pub":
        if any([t in utterance for t in PUB_TERMS]):
            return "pub"
    return "N/A"


def price_range(utterance, slot_value, delex=False):

    CHEAP_TERMS = ["cheap", "inexpensive", "not expensive", 

                   "not very expensive", "low price", "budget"]

    if slot_value == "cheap": 
        if any([t in utterance for t in CHEAP_TERMS]):
            return "cheap"
        if "low" in utterance and "cost" in utterance:
            return "cheap"
        if "low" in utterance and "price" in utterance:
            return "cheap"
        if "value" in utterance and "price" in utterance:
            return "cheap"

    if slot_value == "high":
        if "high price" in utterance:
            return "high"
        if " expensive" in utterance:
            return "high"
        if "highly - priced" in utterance:
            return "high"
        if "high - priced" in utterance:
            return "high"
        if "high cost" in utterance:
            return "high"
        if "higher - priced" in utterance:
            return "high"
        if "price range is high" in utterance:
            return "high"
        if "price range is slightly higher" in utterance:
            return "high"
        if "costly" in utterance:
            return "high"
        if "cost" in utterance:
            return "high"
        if "higher price" in utterance:
            return "high"
        if "upper price" in utterance:
            return "high"
        if "highly priced" in utterance:
            return "high"
        if "prices are high" in utterance:
            return "high"
        if "high" in utterance:
            return "high"
        if "high range" in utterance:
            return "high"
        if "pricey" in utterance:
            return "high"
        if "expensive" in utterance:
            return "high"
        if "not cheap" in utterance:
            return "high"
        if "above average" in utterance:
            return "high"

    if slot_value == "moderate":
        if "moderate price" in utterance:
            return "moderate"
        if "average price" in utterance:
            return "moderate"
        if "average pricing" in utterance:
            return "moderate"
        if "moderate - priced" in utterance:
            return "moderate"
        if "moderately priced" in utterance:
            return "moderate"
        if "moderate" in utterance:
            return "moderate"
        if "affordable" in utterance:
            return "moderate"
        if "medium price" in utterance:
            return "moderate"
        if "mid - range price" in utterance:
            return "moderate"
        if "mid - price" in utterance:
            return "moderate"
        if "mid price" in utterance:
            return "moderate"
        if "reasona" in utterance and "price" in utterance:
            return "moderate"
        if "fair" in utterance and "price" in utterance:
            return "moderate"
        if "average - priced" in utterance:
            return "moderate"
        if "mid range for price" in utterance:
            return "moderate"
        if "medium - priced" in utterance:
            return "moderate"
        if "average" in utterance and "price" in utterance:
            return "moderate"
        if "decent" in utterance and "price" in utterance:
            return "moderate"
        if "mid" in utterance and "price" in utterance:
            return "moderate"
        if "not pricy" in utterance:
            return "moderate"
        if "competitive" in utterance:
            return "moderate"


    if slot_value == "more than £30":
        if "more than £ 30" in utterance:
            return "more than £30"
        if "prices are over £ 30":
            return "more than £30"
        if "over £ 30":
            return "more than £30"


    if slot_value == "less than £20":
        if "less than £ 20" in utterance:
            return "less than £20"
        if "less than 20 pounds" in utterance:
            return "less than £20"
        if "less than twenty pounds" in utterance:
            return "less than £20"
        if "price range under 20" in utterance:
            return "less than £20"
        if "less than" in utterance and "20" in utterance:
            return "less than £20"
        if "under" in utterance and "20" in utterance:
            return "less than £20"
        if "less than" in utterance and "twenty" in utterance:
            return "less than £20"
        if "£ 20 or less" in utterance:
            return "less than £20"
        if "in the £ 20 price range" in utterance:
            return "less than £20"
        if "20" in utterance and "25" not in utterance:
            return "less than £20"
        if "twenty pounds" in utterance:
            return "less than £20"


    if slot_value == "£20-25":
        if "£ 20 - £ 25" in utterance:
            return "£20-25"
        if "£ 20 - 25" in utterance:
            return "£20-25"
        if "20 - 25" in utterance:
            return "£20-25"
        if "20" in utterance and "25" in utterance:
            return "£20-25"
        if "20" in utterance and "30" in utterance:
            return "£20-25"
        if "twenty" in utterance and "twenty five" in utterance:
            return "£20-25"
        if "twenty" in utterance and "twenty - five" in utterance:
            return "£20-25"
        if "under £ 25" in utterance:
            return "£20-25"

    return "N/A"

def family_friendly(utterance, slot_value, delex=False):

    if slot_value == "no":
        if "not kid" in utterance and "friendly" in utterance:
            return "no"
        if "not family" in utterance and "friendly" in utterance:
            return "no"
        if "not child" in utterance and "friendly" in utterance:
            return "no"
        if "no kid" in utterance and "friendly" in utterance:
            return "no"
        if "adult" in utterance:
            return "no"
        if "non kid" in utterance:
            return "no"
        if "non family" in utterance:
            return "no"
        if "non child" in utterance:
            return "no"
        if "non - kid" in utterance:
            return "no"
        if "non - family" in utterance:
            return "no"
        if "non - child" in utterance:
            return "no"
        if "not - kid" in utterance and "friendly" in utterance:
            return "no"
        if "not - family" in utterance and "friendly" in utterance:
            return "no"
        if "not - child" in utterance and "friendly" in utterance:
            return "no"
        if "not a kid" in utterance:
            return "no"
        if "not a family" in utterance:
            return "no"
        if "not a child" in utterance:
            return "no"
        if "no kids allowed" in utterance:
            return "no"
        if "no children allowed" in utterance:
            return "no"
        if "do not allow" in utterance:
            return "no"
        if "n't" in utterance and "family - friendly" in utterance:
            return "no"
        if "not" in utterance and "family" in utterance:
            return "no"
        if "not" in utterance and "children" in utterance:
            return "no"
        if "not" in utterance and "kid" in utterance:
            return "no"
        if "no for children" in utterance:
            return "no"
        if "we welcome" in utterance and "children" in utterance:
            return "no"
        if "family" in utterance and any([x in utterance for x in [" no ", " not ", " n't ", " lacking "]]):
            return "no"
        if "kid" in utterance and any([x in utterance for x in [" no ", " not ", " n't ", " lacking " ]]):
            return "no"
        if "child" in utterance and any([x in utterance for x in [" no ", " not ", " n't ", " lacking "]]):
            return "no"
        if "families" in utterance and any([x in utterance for x in [" no ", " not ", " n't ", " lacking " ]]):
            return "no"
        if "bad family - friendly" in utterance:
            return "no"
        if "unfriendly" in utterance:
            return "no"
        if "non-family-friendly" in utterance:
            return "no"


    if slot_value == "yes":
        if "we welcome" in utterance and "children" in utterance:
            return "yes"
        if "families" in utterance and all([x not in utterance for x in [" no ", " not ", " n't "]]):
            return "yes"
        if "family" in utterance and all([x not in utterance for x in [" no ", " not ", " n't "]]):
            return "yes"
        if "kid" in utterance and all([x not in utterance for x in [" no ", " not ", " n't "]]):
            return "yes"
        if "child" in utterance and all([x not in utterance for x in [" no ", " not ", " n't "]]):
            return "yes"
        if "a family" in utterance:
            return "yes"
        if "for the whole family" in utterance:
            return "yes"
        if "welcomes" in utterance and "children" in utterance:
            return "yes"
        if "child" in utterance and "friendly" in utterance and "not" not in utterance:
            return "yes"
        if "kid" in utterance and "friendly" in utterance and "not" not in utterance: 
            return "yes"
        if "family" in utterance and "friendly" in utterance and "not" not in utterance:
            return "yes"
        if "family" in utterance and  "friendly" in utterance and "not" not in utterance:
            return "yes"

    return "N/A"
   
   
def food(utterance, slot_value, delex=False):

    if slot_value.lower() in utterance:
        return slot_value
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

def customer_rating(utterance, slot_value, delex=False):


    if slot_value == "3 out of 5":
        if " 3 " in utterance and any([x in utterance for x in [" 5 ", " five ", "star", "out of"]]):
            return "3 out of 5"
        if " three " in utterance and any([x in utterance for x in [" 5 ", " five ", "star", "out of"]]):
            return "3 out of 5"

    if slot_value == "1 out of 5":
        if " 1 " in utterance and any([x in utterance for x in [" 5 ", " five ", "star", "out of"]]):
            return "1 out of 5"
        if " one " in utterance and any([x in utterance for x in [" 5 ", " five ", "star", "out of"]]):
            return "1 out of 5"

    if slot_value == "5 out of 5":
        if " 5 " in utterance and any([x in utterance for x in [" 5 ", " five ", "star", "out of"]]):
            return "5 out of 5"
        if " five " in utterance and any([x in utterance for x in [" 5 ", "star", "out of"]]):
            return "5 out of 5"

    if slot_value == "low":
        if "low customer satisfaction" in utterance:
            return "low"
        if "customer rating is quite low" in utterance:
            return "low"
        if re.search("not have high customer ratings", utterance):
            return "low"
        if re.search("not highly rated", utterance):
            return "low"
        if "below than average customer rating" in utterance:
            return "low"
        if "below average customer rating" in utterance:
            return "low"
        if "below than average rating" in utterance:
            return "low"
        if "below average rating" in utterance:
            return "low"
           
        if 'low customer rating' in utterance:
            return "low" 

        if "low" in utterance and "rating" in utterance:
            return "low"
        if "low" in utterance and "rate" in utterance:
            return "low"

        if "poor" in utterance and "rating" in utterance:
            return "low"
        if "poor" in utterance and "rate" in utterance:
            return "low"

        if "not" in utterance and " rat" in utterance:
            return "low"
    
    if slot_value == "high":
        if "is a highly rated" in utterance:
            return "high"
        if "a high quality " in utterance:
            return "high"
        if "has received great reviews" in utterance:
            return "high"
        if "and have great reviews" in utterance:
            return "high" 
        if "respected" in utterance:
            return "high"
        if "recommended" in utterance:
            return "high"
        if "not highly - reviewed" in utterance:
            return "high"
        if "highly - reviewed" in utterance:
            return "high"
        if re.search("has a high customer rating", utterance):
            return "high" 
        if re.search("with a high customer rating", utterance):
            return "high"
        if re.search("average \w+ food", utterance):
            return "average"
        if re.search("highly rated", utterance):
            return "high"
        if re.search("high customer ratings", utterance):
            return "high"
           
        if "high" in utterance and "rating" in utterance:
            return "high"
        if "high" in utterance and "rate" in utterance:
            return "high"


        if "high customer satisfaction" in utterance:
            return "high"

        if re.search("has received high reviews", utterance):
            return "high" 
        if re.search("and has high customer rating", utterance):
            return "high"

        if "and great customer rating" in utterance:
            return "high"
         
    if "an average but" in utterance:
        return "average"
    if "is a good fast food" in utterance:
        return "average"
    if "is a good indian" in utterance:
        return "average"
    if "is a good french" in utterance:
        return "average"
    if "is a good chinese" in utterance:
        return "average"
    if "is a good italian" in utterance:
        return "average"
    if "is an good" in utterance:
        return "average"
    if "with moderate price range and customer rating" in utterance:
        return "average"
    if re.search(r"an average (restaurant|bar|pub|coffee)",  utterance):
        return "average"
    if "with average customer review" in utterance:
        return "average"
    if "with average customers reviews" in utterance:
        return "average"
    if "moderate review" in utterance:
        return "average"
    if "moderate rat" in utterance:
        return "average"
    if "customer rating is average" in utterance:
        return "average" 
    if "is a decent" in utterance:
        return "average"
    if "average rated" in utterance:
        return "average"
    if "rated average" in utterance:
        return "average"
    if re.search("rating for this space is average", utterance):
        return "average"
    if re.search("average food", utterance):
        return "average" 

    
    if "average customer rating" in utterance:
        return "average"
    if "average rat" in utterance:
        return "average"

    if "average" in utterance and "rating" in utterance:
        return "average"
    if "average" in utterance and "rate" in utterance:
        return "average"
    return "N/A" 
