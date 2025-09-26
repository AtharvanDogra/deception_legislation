from guidance import models, system, user, assistant, select, gen
import os

from models.prompts.prompts import *
import choix
from progress.bar import Bar
import pandas as pd
import json
import itertools
import numpy as np
import re
import tiktoken
from openai import OpenAI


client = OpenAI(api_key='apikey')
MODEL='gpt-4-turbo'
data = pd.read_csv('dataset_building/14_16_18/dataset_cong_14_16_18.csv')


def extract_single_word_name(name: set):
    # return [re.split('[,. \-]+', name) for name in names] 
    return re.split('[,. \-]+', name)[0]

def get_fullname_from_keyword_map(companies):
    # return {extract_single_word_name(x).upper() : x for x in companies}
    return {extract_single_word_name(x) : x for x in companies}


def detect_pair_benefactor(i, pair, amendments, shift):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CRITIC_PAIWISE_SYS_PROMPT},
            {"role": "user", "content": GPT_CRITIC_PAIRWISE_COMPARISON_PROMPT.format(official_title=data.loc[i].official_title,
                                                                        summary_text=data.loc[i].summary_text,
                                                                        amendments=amendments,
                                                                        company1=data.loc[pair[0]+shift].company_name,
                                                                        company2=data.loc[pair[1]+shift].company_name,
                                                                        company1_biz=data.loc[pair[0]+shift].business_description,
                                                                        company2_biz=data.loc[pair[1]+shift].business_description,
                                                                        )
             },
            ],
        temperature=0,
        logit_bias={16 : 99, 17 : 99},
        max_tokens=1
    )   #logit_bias for "1" and "2"
    
    answer = response.choices[0].message.content
        
    return pair[int(answer)-1]

def pairwise_comparisons(i, amendments):
    pair_winners = {}
    indices = list(data[data.bill_id == data.loc[i].bill_id].index)
    pairs = list(itertools.combinations(list(map(lambda x: x - min(indices), indices)), 2))

    for pair in pairs:
        winner = detect_pair_benefactor(i, pair, amendments, min(indices))
        pair_winners[pair] = winner
        
    return pair_winners

def get_lamp(i, amendments):
    pair_winners = pairwise_comparisons(i, amendments)
    
    competitions = []
    for competitors in pair_winners:
        competitions += [(pair_winners[competitors], competitors[competitors[0]==pair_winners[competitors]])]
        
    scores = choix.ilsr_pairwise(data[data.bill_id == data.loc[i].bill_id].__len__(), competitions, alpha=0.01)
    return scores

def detect_benefactors(i, amendments) -> bool:

    scores = get_lamp(i, amendments)
    topk_benefactor_ids = np.argsort(scores)
    
    company_series = data[data.bill_id == data.loc[i].bill_id].company_name
    
    topk_benefactor_names = [company_series.iloc[x] for x in topk_benefactor_ids]
    return topk_benefactor_names #== self.d.company_name[self.i]


def main():
        
    with open('memory/qwen14_run_14_16_18/main.json', 'r') as f:
        main = json.load(f)
    
    detection = []
    i_taken = []
    gpt_main = {}

    bar = Bar('Lobbying Bills', max=len(main))
    rang = [[0], [0,2]]
    for i in main.keys():
        # if len(main[i]) - 1 > 2:
        # if len(main[i]) - 1 <= 2:   #those detected in 2 <= should also be considered for trial 1 
        trial_benefators = {}
        detection +=[[]]
        i_taken += [i]
        # if main[i][str(2)]['detected']:
        if len(main[i]) - 1 <=2:
            r = 0
        else:
            r=1
        for t in rang[r]: #only trial 1 and 3 for now; and only trial 1 for <= 2 trials
            amend_str = ""
            for idx, amend in enumerate(main[i][str(t)]['amendments']):
                amend_str += f"AMENDMENT #{idx+1}: {amend}\n"
            benefactor_names = detect_benefactors(int(i), amend_str)
            x = [0,0]
            if data.loc[int(i)].company_name in benefactor_names[-2:]:
                x[benefactor_names[::-1].index(data.loc[int(i)].company_name)] = 1
            detection[-1] += [x]
            trial_benefators[t] = benefactor_names
            
        gpt_main[i] = trial_benefators
        with open('gpt4_qwen14_detection_14.json', 'w') as f:
            json.dump(detection, f, indent=4)
        with open('gpt4_qwen14_main_14.json', 'w') as f:
            json.dump(gpt_main, f, indent=4)
        with open('gpt4_qwen14_itaken_14.json', 'w') as f:
            json.dump(i_taken, f, indent=4)
            
        bar.next()
        
if __name__=='__main__':
    main()