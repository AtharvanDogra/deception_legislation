from guidance import models, system, user, assistant, select, gen
import os

from models.models import *
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

import llama_cpp

from collections import Counter



# client = OpenAI(api_key='apikey')
# MODEL='gpt-4-turbo'

DATAFILE = {'8': 'dataset_building/8_11/selected_8_11.csv',
                '12': 'dataset_building/gt112_except14_16_18/gt112.csv',
                '14': 'dataset_building/14_16_18/dataset_cong_14_16_18.csv'}

MEMORYFILE = {'8': '8_11',
                '12': 'gt112',
                '14': '14_16_18'}

MULTI_SMALL_CRITIC = True
POLL_PAIRS = True          # polling done during pair comparison
POLL_BENEFACTORS = False   # polling during final top 1 benefactor selection

lm =  QwenChat(os.path.join('llms/qwen1_5-14b-chat-q8_0.gguf'), n_gpu_layers=-1,
        n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=0)#LAYER, tensor_split=[0.5, 0.5, 0, 0]) # main_gpu =


if MULTI_SMALL_CRITIC:    # for multi-small critics
    # lm1=  QwenChat(os.path.join('llms/qwen1_5-14b-chat-q8_0.gguf'), n_gpu_layers=-1,
    #         n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=1, seed=5)
    lm1 = models.MistralChat(os.path.join('llms/mixtral-8x7b-instruct-v0.1.Q6_K.gguf'), n_gpu_layers=-1,
                        n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=1)
    lm2=  Yi(os.path.join('llms/Yi-1.5-34B-Chat-16K-Q5_K_S.gguf'), n_gpu_layers=-1,
            n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=2)


def extract_single_word_name(name: set):
    # return [re.split('[,. \-]+', name) for name in names] 
    return re.split('[,. \-]+', name)[0]

def get_fullname_from_keyword_map(companies):
    # return {extract_single_word_name(x).upper() : x for x in companies}
    return {extract_single_word_name(x) : x for x in companies}


def detect_pair_benefactor(data, lm, id, pair, amendments, shift, sys=True):

    pair_name = [extract_single_word_name(data.loc[pair[0]+shift].company_name).upper(), extract_single_word_name(data.loc[pair[1]+shift].company_name).upper()]
    
    if sys:
        with system():
            lm = lm + CRITIC_PAIWISE_SYS_PROMPT
    with user():
        pair_prompted = lm + STRONGER_CRITIC_PAIRWISE_COMPARISON_PROMPT.format(official_title=data.loc[id].official_title,
                                                                    summary_text=data.loc[id].summary_text,
                                                                    amendments=amendments,
                                                                    company1=pair_name[0],
                                                                    company2=pair_name[1],
                                                                    company1_biz=data.loc[pair[0]+shift].business_description,
                                                                    company2_biz=data.loc[pair[1]+shift].business_description,
                                                                    )
    with assistant():
        winner = pair_prompted + select(pair_name, name='benefactor')
    pair_winner = pair_name.index(winner['benefactor'])
    return pair[pair_winner]

def pairwise_comparisons(data, id, amendments, multi_small_critic=False, poll_pairs=False):
    pair_winners = {}
    if multi_small_critic and not poll_pairs:
        pair_winners = [{}, {}, {}]
    indices = list(data[data.bill_id == data.loc[id].bill_id].index)
    pairs = list(itertools.combinations(list(map(lambda x: x - min(indices), indices)), 2))

    for pair in pairs:
        winner = detect_pair_benefactor(data, lm, id, pair, amendments, min(indices))
        
        if multi_small_critic and poll_pairs:
            winner = [winner] + [detect_pair_benefactor(data, lm1, id, pair, amendments, min(indices), sys=False)]
            winner.append(detect_pair_benefactor(data, lm2, id, pair, amendments, min(indices)))
            winner = Counter(winner).most_common(1)[0][0] # get highest chosen in 3 polls
            pair_winners[pair] = winner
            
        elif multi_small_critic and not poll_pairs:
            pair_winners[0][pair] = winner
            winner = detect_pair_benefactor(data, lm1, id, pair, amendments, min(indices))
            pair_winners[1][pair] = winner
            winner = detect_pair_benefactor(data, lm2, id, pair, amendments, min(indices))
            pair_winners[2][pair] = winner
        else:
            pair_winners[pair] = winner
        
    return pair_winners

def get_lamp(data, id, amendments, multi_small_critic=False, poll_pairs=False):
    pair_winners = pairwise_comparisons(data, id, amendments, multi_small_critic, poll_pairs)
    
    if multi_small_critic and not poll_pairs:
        competitions = [[],[],[]]
        for i in range(len(pair_winners)):
            for competitors in pair_winners[i]:
                competitions[i] += [(pair_winners[i][competitors], competitors[competitors[0]==pair_winners[i][competitors]])]
    else:
        competitions = []
        for competitors in pair_winners:
            competitions += [(pair_winners[competitors], competitors[competitors[0]==pair_winners[competitors]])]
    
    if multi_small_critic and not poll_pairs:
        scores = []
        for seed_competitions in competitions:
            scores.append(choix.ilsr_pairwise(data[data.bill_id == data.loc[id].bill_id].__len__(), seed_competitions, alpha=0.01).tolist())
    else:
        scores = choix.ilsr_pairwise(data[data.bill_id == data.loc[id].bill_id].__len__(), competitions, alpha=0.01).tolist()
    return scores

def detect_benefactors(data, id, amendments, multi_small_critic=False, poll_pairs=False) -> bool:

    scores = get_lamp(data, id, amendments, multi_small_critic, poll_pairs)
    topk_benefactor_ids = np.argsort(scores)
    
    company_series = data[data.bill_id == data.loc[id].bill_id].company_name
    
    if multi_small_critic and not poll_pairs:
        topk_benefactor_names = []
        for i in range(topk_benefactor_ids.shape[0]):
            topk_benefactor_names.append([company_series.iloc[x] for x in topk_benefactor_ids[i]])
    else:
        topk_benefactor_names = [company_series.iloc[x] for x in topk_benefactor_ids]
        
    return topk_benefactor_names, scores #== self.d.company_name[self.i]


def main(model='qwen', split='14', multi_small_critic=MULTI_SMALL_CRITIC):
    
    memory_dir = f'memory/constant_critic/smaller_14/qwen/poll_pairs_three_llms'
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)
    if POLL_BENEFACTORS:
        if not os.path.exists(os.path.join(memory_dir, 'poll_benefactors')):
            os.makedirs(os.path.join(memory_dir, 'poll_benefactors'))
        memory_dir = os.path.join(memory_dir, 'poll_benefactors')
    if POLL_PAIRS:
        if not os.path.exists(os.path.join(memory_dir, 'poll_pairs')):
            os.makedirs(os.path.join(memory_dir, 'poll_pairs'))
        memory_dir = os.path.join(memory_dir, 'poll_pairs')
        
    
    pruning = 500
    data = pd.read_csv(DATAFILE[split])
    
    with open(f'memory/{model}_run_{MEMORYFILE[split]}/main.json', 'r') as f:
        main = json.load(f)
    
    detection = []
    i_taken = []
    gpt_main = {}

    # rang = [[0], [0,2]]
    # rang = [0]
    iterator = list(main.keys())[:pruning]   # short experiments
    bar = Bar('Lobbying Bills', max=len(iterator))
    
    for i in iterator:
        # if len(main[i]) - 1 > 2:
        # if len(main[i]) - 1 <= 2:   #those detected in 2 <= should also be considered for trial 1 
        trial_benefators = {}
        trial_scores = {}
        detection +=[[[0, 0] for i in range(3)]]
        i_taken += [i]
        # if main[i][str(2)]['detected']:
        
        # if len(main[i]) - 1 <=2:
        #     r = 0
        # else:
        #     r=1
        r = min(len(main[i])-1, 3)
        for t in range(r):#[r]: #only trial 1 and 3 for now; and only trial 1 for <= 2 trials
            amend_str = ""
            for idx, amend in enumerate(main[i][str(t)]['amendments']):
                amend_str += f"AMENDMENT #{idx+1}: {amend}\n"
            benefactor_names, scores = detect_benefactors(data, int(i), amend_str, 
                                                  multi_small_critic=MULTI_SMALL_CRITIC,
                                                  poll_pairs=MULTI_SMALL_CRITIC and POLL_PAIRS)
            # x = [0,0]
            if MULTI_SMALL_CRITIC and POLL_BENEFACTORS:
                    poll_benefactors = []
                    
                    for poll in range(len(benefactor_names)):
                        poll_benefactors.append(benefactor_names[poll][-1])        # changed to top 1
                    if Counter(poll_benefactors)[data.loc[int(i)].company_name]>1:      # >1 if 3 polls
                        detection[-1][t][0] = 1
            
            else:
                if data.loc[int(i)].company_name == benefactor_names[-1]:       # changed to top 1
                    # detection[-1][t][benefactor_names[::-1].index(data.loc[int(i)].company_name)] = 1
                    detection[-1][t][0] = 1
            
            # detection[-1] += [x]
            trial_benefators[t] = benefactor_names
            trial_scores[t] = scores
        
        
        gpt_main[i] = {'benefactor': data.loc[int(i)].company_name,
                       'ranking[low-high]': trial_benefators,
                       'scores': trial_scores}
        with open(os.path.join(memory_dir, f'detection_{split}_{pruning}.json'), 'w') as f:
            json.dump(detection, f, indent=4)
        with open(os.path.join(memory_dir, f'main_{split}_{pruning}.json'), 'w') as f:
            json.dump(gpt_main, f, indent=4)
        with open(os.path.join(memory_dir, f'itaken_{split}_{pruning}.json'), 'w') as f:
            json.dump(i_taken, f, indent=4)
            
        bar.next()
        
if __name__=='__main__':
    main()