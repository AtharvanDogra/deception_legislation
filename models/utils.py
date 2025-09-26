from itertools import combinations
import re
import pandas as pd
import json
from collections import defaultdict


def summary_hack(text: str):
    return re.split('300 words:|500 words:', text)[-1]
    
def extract_single_word_name(name: set):
    # return [re.split('[,. \-]+', name) for name in names] 
    return re.split('[,. \-]+', name)[0]

def pairwise_combinations_with_keywords(companies: list[str]):
    pairs = list(combinations(companies, 2))
    pairs_with_keywords = {pair:(extract_single_word_name(pair[0]).upper(), extract_single_word_name(pair[1]).upper()) for pair in pairs}
    return pairs_with_keywords

def pair_details_prompt(data: pd.DataFrame, pair: tuple, summ: bool=False):
    details_prompt = ""
    for comp in pair:
        assert len(list(data[data.company_name == comp].business_summarized if summ else data[data.company_name == comp].business_description)[-1]) > 5 # to ensure there's something in in the summary
        details_prompt += f"""\n{comp}: {list(data[data.company_name == comp].business_summarized if summ         
                                            else data[data.company_name == comp].business_description)[-1]}\n\n"""  #removed comp.upper() for now
    # there'd only be one company in company_name==comp but list() just to avoid unanticipated bugs
    return details_prompt

def get_fullname_from_keyword_map(companies):
    return {extract_single_word_name(x).upper() : x for x in companies}

def fullname_id_map(companies):
    company_id_map = {}
    for idx, i in enumerate(list(set(companies))):
        company_id_map[i] = idx
    return company_id_map

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

def jsonKV2int(x):
    if isinstance(x, dict):
        return {int(k):(int(v) if isinstance(v, str) else v) for k,v in x.items()}
    return x


# memory dict utils

def rec_dict(): return defaultdict(rec_dict)

def dict_to_default(d):
    memory = rec_dict()
    if isinstance(d, dict):
        for k, v in d.items():
            # memory[int(k)] = regular_to_default(v)
            memory[k] = dict_to_default(v)
    else:
        return d
    return memory



### Not being used ###

def extract_stored_amends(record: int, store: dict=None, turn: int=0):
    """
    Currently applied to extract store from store.json if no store is provided to fn.
    """
    amend_string=""
    
    if store is None:
        with open('store.json', 'r') as f:
            store = json.load(f)
            
    amends = store[str(record)]['amendment'][str(turn)]
    for i, amend in enumerate(amends):
        amend_string += f"{i+1}) {amend}\n"
    return amend_string


def process_store_amend(i: int, turn: int, output: str, store: dict, data: pd.DataFrame) -> list:
    """
    structure and store 'amendments' 
    and 'benefits' generated
    
    TODO: also remove the general information about the amendments and benefits at the end, filtering based on \n\n
    """
    bill_id = data.bill_id[i]
    company = data.company_name[i]
    
    try:
        store[i]['amendment'][turn] = []
        store[i]['benefit'][turn] = []
    except:
        print('storing ds came to exception')
        store[i] = {'bill_id': bill_id, 'company': company, 'amendment': {turn: []}, 'benefit': {turn: []}}
    
    amends_and_benefits = re.findall(r'\d\)\D*', output, flags=re.DOTALL)
    for point in amends_and_benefits:
        amendment = re.findall(r'AMENDMENT.*BENEFIT', point, flags=re.DOTALL)[0].replace(r'BENEFIT', '').replace(r'AMENDMENT: ', '').strip()
        benefit = re.findall(r'BENEFIT:.*', point, flags=re.DOTALL)[0].replace(r'BENEFIT: ', '').strip()
        
        store[i]['amendment'][turn] += [amendment]
        store[i]['benefit'][turn] += [benefit]
        
    with open('store.json', 'a+') as f:
        f.write(json.dumps(store, indent=4, sort_keys=False,))
        
    return store

# Incomplete; not being used

def log_generation(generation: str, gen_type: str=None):
    with open('generation_log.log', 'a+') as f:
        f.write(generation+'\n\n')
        
    pass