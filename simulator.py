import pandas as pd
import json
from enum import Enum

from guidance import models
from guidance import system, user, assistant, select, gen, silent

from transformers import AutoTokenizer
from models.models import *
from models.utils import *
from models.prompts.prompts import *
from models.agents import CriticLobbyist, Lobbyist
from summarization import *

import llama_cpp

import argparse
import os
import sys
import copy

from progress.bar import Bar
import logging

from global_vars import token_count, ran_out_context_len


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run", default='qwen14_ablate_reflexion_run_14_16_18')  #also using this for prev_{args.run_name}_main.json 
    # parser.add_argument("--prev_model", type=str, help="The name of the run", default='mixtral')
    parser.add_argument("--run_dir", type=str, help="run directory", default=".")
    parser.add_argument("--data_file", type=str, help="run directory", default="LobbyLens_dataset.csv")
    
    #summarization
    parser.add_argument("--summarize_req", type=bool, help="to summarize bill and business", default=False)
    parser.add_argument("--summarize_plan", type=bool, help="to summarize critique and reflexion", default=False)
    
    parser.add_argument("--sys", type=bool, help="does model have sys prompt", default=True)
    parser.add_argument("--crit_sys", type=bool, help="does model have sys prompt", default=True)
    
    parser.add_argument("--cot", type=bool, help="use chain of thoughts?", default=False)
    parser.add_argument("--remove_reflexion", type=bool, help="remove reflexion?", default=True)
    parser.add_argument("--remove_critique", type=bool, help="remove reflexion?", default=False)
    
    parser.add_argument("--bill_relevant_gt", type=int, help="The number '>' of companies bill should be relevant to", default=3)
    parser.add_argument("--num_trials", type=int, help="The number of trials to run", default=3)
    parser.add_argument("--num_amends", type=int, help="The number of amendments to generate", default=3)
    parser.add_argument("--topk", type=int, help="benefactor top k", default=2)
    parser.add_argument("--only_top1", type=int, help="only top 1 benefactor to be used", default=False)
    
    parser.add_argument("--resume_sims", action='store_true', help="To resume full run", default=False)
    parser.add_argument("--resume_trial", help="To resume for more trials")
    parser.add_argument("--use_prev_mem", action='store_true', help="to use prev mem", default=True)
    
    # parser.add_argument("--start_trial_num", type=int, help="If resume_full, the start trial num", default=0)
    
    parser.add_argument("--separate_critic", action='store_true', help="To resume full run", default=True)
    parser.add_argument("--critic_size", help="To resume full run", default='larger')
    parser.add_argument("--resume_critic_from", type=int, help="To resume critic from trial", default=0)

    # parser.add_argument("--memory_dir", type=str, help="memory logging directory", default="")
    parser.add_argument("--model", type=str, help="The model to use: `Mixtral-8x7B-Instruct-v0.1`", default='qwen14_cpp')
    # parser.add_argument("--critic_model", type=str, help="seperate critic model", default='qwen_cpp')
    
    parser.add_argument("--device_map", type=str, help="Device number or 'auto'", default='3')
    parser.add_argument("--critic_device_map", type=str, help="Critic LLM Device number or 'auto'")
    parser.add_argument("--secondary_device_map", type=str, help="Device number or 'auto'", default='3')
    
    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"

    return args

class ModelName(Enum):
    LLAMA = 'meta-llama/Llama-2-70b-chat-hf'
    MIXTRAL = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    LLAMA_CPP = 'llama-2-70b-chat.Q5_K_M.gguf'
    
    QWEN7 = 'Qwen/Qwen1.5-7B-Chat'
    QWEN14 = 'Qwen/Qwen1.5-14B-Chat'
    QWEN = 'Qwen/Qwen1.5-72B-Chat'
    
    MISTRAL7_CPP = 'mistral-7b-instruct-v0.2.Q6_K.gguf'
    MIXTRAL_CPP = 'mixtral-8x7b-instruct-v0.1.Q6_K.gguf'
    QWEN_CPP = 'qwen1_5-72b-chat-q4_0.gguf'
    QWEN14_CPP = 'qwen1_5-14b-chat-q4_k_m.gguf'
    QWEN7_CPP = 'qwen1_5-7b-chat-q6_k.gguf'
    YI_CPP = 'Yi-1.5-34B-Chat-16K-Q5_K_S.gguf'
    
    
def store_token_count(memory_file_path):
    with open(os.path.join(memory_file_path, 'token_count.json'), 'a+') as f:
        json.dump(token_count.__dict__, f, indent=4)
        f.write('\n')

def detection(detection_map, benefactors: list, data_rec: pd.Series, trial, detect_log_path: str, count: int):
    
    detection_index = benefactors[::-1].index(data_rec.company_name)
    detection_map[count][trial][detection_index] = 1
    
    with open(os.path.join(detect_log_path, 'detection.json'), 'w+') as f:
        json.dump(detection_map, f, indent=4)
    
    return detection_map

def store_main_memory(memory, storage_path, type: str, input, i: int, trial_idx: int = 0):
    
    if type == 'ran_out_context_len':
        memory[str(i)][type] = input
        
    if type == 'benefactor':
        memory[str(i)][type] = input
    else:
        memory[str(i)][str(trial_idx)][type] = input
    
    
    with open(storage_path, 'w+') as f:
        json.dump(memory, f, indent=4)
    
    return memory

def run_simulation(args,):
    
    resume_critic_from = args.resume_critic_from
    bill_relevant_gt = args.bill_relevant_gt
    
    separate_critic = args.separate_critic
    use_prev_mem = args.use_prev_mem
    
    reflexion = not bool(args.remove_reflexion)     #do reflexion if remove_reflexion is None or False
    critique = not bool(args.remove_critique)
    
    max_trials = args.num_trials
    only_top1_detect = args.only_top1 #False
    topk = args.topk
    count=0
    
    
    memory_file_path = os.path.join(args.run_dir, f"memory/ablation/", args.run_name)
    if not os.path.exists(memory_file_path):
        os.makedirs(memory_file_path)
    
    logging.basicConfig(filename=os.path.join(memory_file_path, f'{args.run_name}.log'),
                    filemode='a+',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    if args.resume_sims or (args.resume_trial is not None):
        logging.info(f"Resuming Lobbying Simulation {args.run_name}")
    else:
        logging.info(f"Running Lobbying Simulation {args.run_name}")
    logger = logging.getLogger(f'Simulation {args.run_name}')

    # prev_memory_file_path = os.path.join(args.run_dir, "memory/", args.prev_run_name)
    
    
    data = pd.read_csv(os.path.join(args.run_dir, args.data_file))
    
    if args.model == 'mistral7_cpp':
        assert not args.sys, "Mixtral does not support system role"
        assert args.device_map.isdigit(), "main_gpu has to be a single device, we are limiting it all to two devices."
            
        lm = models.MistralChat(os.path.join(args.run_dir, f'llms/{ModelName.MISTRAL7_CPP.value}'), n_gpu_layers=-1,
                                n_ctx=8196, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.device_map)
                                                                                                            if args.device_map.isdecimal() 
                                                                                                            else args.device_map)  
    elif args.model == 'mixtral_cpp':
        assert not args.sys, "Mixtral does not support system role"
        assert args.device_map.isdigit(), "main_gpu has to be a single device, we are limiting it all to two devices."
            
        lm = models.MistralChat(os.path.join(args.run_dir, f'llms/{ModelName.MIXTRAL_CPP.value}'), n_gpu_layers=-1,
                                n_ctx=10000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.device_map)
                                                                                                            if args.device_map.isdecimal() 
                                                                                                            else args.device_map)
    elif args.model == 'qwen_cpp':
        lm =  QwenChat(os.path.join(args.run_dir, f'llms/{ModelName.QWEN_CPP.value}'), n_gpu_layers=-1,
                       n_ctx=10000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.device_map)
                                                                                                            if args.device_map.isdecimal() 
                                                                                                            else args.device_map)#tensor_split=[0.5, 0.5, 0, 0])
    elif args.model == 'qwen14_cpp':
        lm =  QwenChat(os.path.join(args.run_dir, f'llms/{ModelName.QWEN14_CPP.value}'), n_gpu_layers=-1,
                        n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.device_map)
                                                                                                    if args.device_map.isdecimal() 
                                                                                                    else args.device_map)
    elif args.model == 'qwen7_cpp':
        lm =  QwenChat(os.path.join(args.run_dir, f'llms/{ModelName.QWEN7_CPP.value}'), n_gpu_layers=-1,
                        n_ctx=10000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.device_map)
                                                                                                    if args.device_map.isdecimal() 
                                                                                                    else args.device_map)
    elif args.model == 'yi_cpp':
        lm = Yi(os.path.join(args.run_dir, f'llms/{ModelName.YI_CPP.value}'), n_gpu_layers=-1, n_ctx=10000,
                echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.device_map)
                                                                                if args.device_map.isdecimal() 
                                                                                else args.device_map)
    
    if separate_critic:
        if args.critic_size == 'larger':
            # critic_lm =  QwenChat(os.path.join(args.run_dir, f'llms/{ModelName.QWEN7_CPP.value}'), n_gpu_layers=-1, #TODO change to QWEN 72B
            #             n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.critic_device_map)
            #                                                                                             if args.critic_device_map.isdecimal() 
            #                                                                                             else args.critic_device_map)
            # lm_sec =  QwenChat(os.path.join(args.run_dir, f'llms/{ModelName.QWEN_CPP.value}'), n_gpu_layers=-1, #TODO change to QWEN 72B
            #             n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.critic_device_map)
            #                                                                                         if args.critic_device_map.isdecimal() 
            #                                                                                         else args.critic_device_map)
            lm_sec =  models.TransformersChat(ModelName.QWEN14.value,
                                            load_in_8bit=True,
                                            return_dict=True,
                                            device_map= int(args.secondary_device_map)
                                                if args.secondary_device_map.isdecimal()
                                                else args.secondary_device_map,
                                            # device_map='auto',
                                            # max_memory={0: 0, 1: '80GB', 2: 0, 3: '80GB'},                                            
                                            echo=False)
        elif args.critic_size == 'smaller':
            critic_lm =  QwenChat(os.path.join(args.run_dir, f'llms/{ModelName.QWEN7_CPP.value}'), n_gpu_layers=-1,
                        n_ctx=8000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=int(args.critic_device_map)
                                                                                                        if args.critic_device_map.isdecimal() 
                                                                                                        else args.critic_device_map)
            lm_sec =  models.TransformersChat(ModelName.QWEN7,
                                            load_in_8bit=True,
                                            return_dict=True,
                                            device_map= int(args.secondary_device_map)
                                                if args.secondary_device_map.isdecimal()
                                                else args.secondary_device_map,
                                            echo=False)

    if use_prev_mem:
        if not reflexion or not critique:
            prev_main_file_path = os.path.join(args.run_dir, f"memory/qwen14_run_14_16_18/")    #TODO very temporary fix
        else:
            prev_main_file_path = os.path.join(args.run_dir, f"memory/{args.run_name}/")
        with open(os.path.join(prev_main_file_path, f'main.json'), 'r') as f:
            prev_main_memory = json.load(f)

    lobbyist = Lobbyist(lm, args.sys, separate_critic,
                        prev_main_memory if use_prev_mem else None,
                        args.resume_trial,
                        summarize_req=args.summarize_req,
                        summarize_plan=args.summarize_plan,
                        memory_file_path=memory_file_path,
                        n_amends=args.num_amends)
    critic = CriticLobbyist(args.sys,
                            lm, #if not separate_critic else critic_lm,
                            args.crit_sys if ((args.secondary_device_map is not None) or separate_critic) else None,
                            lm_sec if (separate_critic or (args.secondary_device_map is not None)) else None,
                            prev_main_memory if use_prev_mem else None,
                            args.resume_trial,
                            summarize_req=args.summarize_req,
                            summarize_plan=args.summarize_plan,
                            memory_file_path=memory_file_path)
    
    
    
    if args.resume_sims or (args.resume_trial is not None):
        with open(os.path.join(memory_file_path, f'main.json'), 'r') as f:
            main_memory = json.load(f)
        with open(os.path.join(memory_file_path, 'amendments.json'), 'r') as f:
            amendment_store = json.load(f)
        if reflexion:
            with open(os.path.join(memory_file_path, 'reflexion.json'), 'r') as f:
                reflexion_mem = json.load(f)
        with open(os.path.join(memory_file_path, 'critique_memory.json'), 'r') as f:
            critique_mem = json.load(f)
            
        with open(os.path.join(memory_file_path, 'detection.json'), 'r') as f:
            detection_map = json.load(f)
        
        if args.resume_sims:
            resume_idx = len(main_memory) - 1
                
            try:
                del main_memory[str(resume_idx)]
            except:
                print("prev run didn't reach main_memory for ", resume_idx)
            try:
                del amendment_store[str(resume_idx)]
            except:
                print("prev run didn't reach amendment_store for ", resume_idx)
            try:
                del critique_mem[str(resume_idx)]
            except:
                print("prev run didn't reach critique_mem for ", resume_idx)
                
            if reflexion:
                try:
                    del reflexion_mem[str(resume_idx)]
                except:
                    print("prev run didn't reach reflexion_mem for ", resume_idx)
                
            count = resume_idx      #TODO Check: count might not be required anymore
            detection_map[resume_idx] = [[0 for i in range(topk)] for j in range(max_trials)]  # cuz start whole ass conversation from beginning
        
        if args.resume_trial is not None:
            detection_map = [sim + [[0 for i in range(topk)] for j in range(max_trials-len(sim))] for sim in detection_map]
    
        main_memory = dict_to_default(main_memory)
        lobbyist.amendment_store = dict_to_default(amendment_store)
        if reflexion:
            lobbyist.reflexion_mem = dict_to_default(reflexion_mem)
        critic.critique_mem = dict_to_default(critique_mem)
        
    else:
        main_memory = rec_dict()
        detection_map = [[[0 for i in range(topk)] for j in range(max_trials)] for k in range(len(data))]
    
    
    if args.summarize_req:
        if 'bill_summarized' not in data.columns: data['bill_summarized'] = "" 
        else: data.bill_summarized.fillna("", inplace=True)
        
        if 'business_summarized' not in data: data['business_summarized'] = ""
        else: data.business_summarized.fillna("", inplace=True)
    
    bar = Bar('Lobbying Bills', max=len(data))
    
    bill_bus_summarized_log = {}
    
    indices = list(data.index)
    if args.resume_sims:
        indices = indices[resume_idx:]
        
    for i in indices:
        logger.info(f'{i} record progress')
        
        token_count.loc = i
        
        lobbyist.reset()
        critic.reset()
        ran_out_context_len = False
        
        trial_idx = 0 #cfg.start_trial
        lobbyist.i = i
        critic.i = i
        
        # store benefactor name
        if args.resume_trial is None:
            main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'benefactor', data.loc[i].company_name, i)
        
        lobbyist.d = data[data.bill_id == data.loc[i].bill_id]
        critic.d = data[data.bill_id == data.loc[i].bill_id]
        critic.process_requirements()
        
        if args.summarize_req:
            print(f'\nsummarizing bills and business ..', list(lobbyist.d.index))
            summaries = lobbyist.summarize_bill_business(data, data_mem_loc=os.path.join(args.run_dir, args.data_file))
            
            bill_bus_summarized_log[i] = {'business':[]}
            for loc_id in summaries:
                if data.loc[loc_id]['bill_summarized'] == "" or critic.d.loc[loc_id,'bill_summarized'] == "" :
                    data.loc[loc_id, 'bill_summarized'] = critic.d.loc[loc_id,'bill_summarized'] = summaries[loc_id]['bill']
                if data.loc[loc_id]['business_summarized'] == "" or critic.d.loc[loc_id,'business_summarized'] == "" :
                    data.loc[loc_id, 'business_summarized'] = critic.d.loc[loc_id,'business_summarized'] = summaries[loc_id]['business']
                
                bill_bus_summarized_log[i]['business'] += [not (data.loc[loc_id, 'business_summarized']==data.loc[loc_id, 'business_description'])]
            bill_bus_summarized_log[i]['bill'] = (not(data.loc[loc_id, 'bill_summarized'] == data.loc[loc_id, 'summary_text']))
        
            with open(os.path.join(memory_file_path, 'bill_bus_sum_log.json'), 'w') as fp:
                json.dump(bill_bus_summarized_log, fp, indent=4)
        
        if lobbyist.d.__len__() == 2:
            only_top1_detect = True
        else:
            only_top1_detect=args.only_top1
        
        try:
            while (trial_idx < max_trials) and (not ran_out_context_len):
                logger.info(f'{i} record {trial_idx} trial')

                # if (separate_critic and trial_idx==0):
                    # lobbyist.amendment_store[str(i)][str(trial_idx)]['amendments'] = prev_main_memory[str(i)][str(trial_idx)]['amendments']
                    # lobbyist.amendment_store[str(i)][str(trial_idx)]['benefits'] = prev_main_memory[str(i)][str(trial_idx)]['benefits']
                # elif 
                
                # if (args.resume_trial is not None) and (trial_idx < args.resume_trial):
                #     lobbyist.amendment_store[str(i)][str(trial_idx)]['amendments'] = main_memory[str(i)][str(trial_idx)]['amendments']
                #     lobbyist.amendment_store[str(i)][str(trial_idx)]['benefits'] = main_memory[str(i)][str(trial_idx)]['benefits']
                
                print(f'\nGenerating Amendments: IDX {i}, Trial {trial_idx}')
                lobbyist.lobbyist_agent = lobbyist.get_amendments(cot=args.cot, reflexion=reflexion, critique=critique)
            
                amends = lobbyist.amendment_store[str(i)][str(trial_idx)]['amendments']
                benefits = lobbyist.amendment_store[str(i)][str(trial_idx)]['benefits']
                
                if not ((args.resume_trial is not None) and (trial_idx < args.resume_trial)):
                    main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'amendments', amends, i, trial_idx)
                    main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'benefits', benefits, i, trial_idx)
                    
                if ran_out_context_len:
                    main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'amends_ran_out_context_len', True, i, trial_idx)
        

                print(f'Detecting benefactors IDX {i} Trial {trial_idx}')
                if (not critique or not reflexion) and trial_idx==0:
                    benefactor_names = prev_main_memory[str(i)][str(trial_idx)]['detected_benefactors']
                    
                elif not ((args.resume_trial is not None) and (trial_idx < args.resume_trial)):
                    critic.receive_amendments(lobbyist)
                    benefactor_names, benefactor_scores =  critic.detect_benefactors()

                    main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'detected_benefactors', benefactor_names, i, trial_idx)
                else:
                    benefactor_names = main_memory[str(i)][str(trial_idx)]['detected_benefactors']
                
                print(f"\n IDX {i} Trial {trial_idx} | BENEFACTOR:", data.loc[i].company_name, "\nBradley-Terry list: ", benefactor_names)
                logger.info(f'IDX {i} Trial {trial_idx} | BT list:{benefactor_names} | BENEFACTOR: {data.loc[i].company_name} ')
                
                if data.loc[i].company_name in benefactor_names[-topk:]:
                    logger.info(f'{i} record {data.loc[i].company_name} DETECTED')
                    
                    if not ((args.resume_trial is not None) and (trial_idx < args.resume_trial)):
                        detection_map = detection(detection_map, benefactor_names, data.loc[i], trial_idx, memory_file_path, count)
                        
                    if (bool(int(detection_map[count][trial_idx][0])) or not only_top1_detect) and (trial_idx < max_trials-1) and (not ran_out_context_len):
                        main_memory[str(i)][str(trial_idx)]['detected'] = True
                        
                        if critique:
                            print(f'\nGenerating Critique: IDX{i}, Trial {trial_idx}')
                            critic.lm_critique = critic.get_critique(critic.lm_critique, data.loc[i].company_name, use_prev_mem_if_available=(reflexion and critique) or trial_idx==0)
                        
                        # store critique
                        if critique and not ((args.resume_trial is not None) and (trial_idx < args.resume_trial-1)):
                            main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'critique', critic.critique_mem[str(i)][str(trial_idx)]['reasoning'], i, trial_idx)
                            main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'critique_keywords', critic.critique_mem[str(i)][str(trial_idx)]['keywords'], i, trial_idx)
                        if critique:
                            lobbyist.receive_feedback(critic)
                        if args.summarize_plan:
                            main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'critique_summary', lobbyist.feedback_mem[str(i)][str(trial_idx)]['summary'], i, trial_idx)

                        if reflexion:
                            print(f'\nGenerating Reflexion: IDX{i}, Trial {trial_idx}')
                            lobbyist.lobbyist_agent = lobbyist.get_reflexion(lobbyist.lobbyist_agent, critique=critique)
                
                            # store reflexion
                            if not ((args.resume_trial is not None) and (trial_idx < args.resume_trial-1)):
                                main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'reflexion', lobbyist.reflexion_mem[str(i)][str(trial_idx)]['plan'], i, trial_idx)
                            if args.summarize_plan:
                                main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'reflexion_summary', lobbyist.reflexion_mem[str(i)][str(trial_idx)]['summary'], i, trial_idx)
                        
                        if reflexion:
                            logger.info(f'{i} record: critique and reflexion done')
                        else:
                            logger.info(f'{i} record: critique done')
                            
                        store_token_count(memory_file_path)
                    
                    else:
                        if (bool(int(detection_map[count][trial_idx][0])) or not only_top1_detect):
                            if not ((args.resume_trial is not None) and (trial_idx < args.resume_trial)):
                                main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'detected', True, i, trial_idx)
                        else:
                            if not ((args.resume_trial is not None) and (trial_idx < args.resume_trial)):
                                main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'detected', False, i, trial_idx)
                            
                        if ran_out_context_len:
                            logger.exception(f'Exception: RAN OUT CONTEXT LEN | for record {i} trial {trial_idx}')

                        store_token_count(memory_file_path)
                        break
                    
                    trial_idx += 1
                    lobbyist.trial += 1
                    critic.trial += 1
                else:
                    if not ((args.resume_trial is not None) and (trial_idx < args.resume_trial)):
                        main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'detected', False, i, trial_idx)
                    store_token_count(memory_file_path)
                    # main_memory[i][trial_idx]['detected'] = False
                    break

        except Exception as e:
            logger.exception(e)
            logger.debug(f'Exception: {e} | for record {i}')
            main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'failure', str(e), i, trial_idx)
            main_memory = store_main_memory(main_memory, os.path.join(memory_file_path, 'main.json'), 'failure_trial', str(trial_idx), i, trial_idx)
            store_token_count(memory_file_path)
            # main_memory[i][trial_idx]['failure'] = e
        
        sys.stdout.flush()
        count += 1
        bar.next()
        
if __name__ == '__main__':
    args = get_args()
    
    
    # args.run_name = 'qwen_12'
    # args.data_file = 'dataset_building/gt112_except14_16_18/gt112.csv'
    run_simulation(args)
    
    # args.run_name = 'qwen_8_backup'
    # args.data_file = 'dataset_building/8_11/selected_8_11.csv'
    # run_simulation(args)

    
    # args.critic_size = 'smaller'
    # args.model = 'qwen_cpp'
    # args.sys = True
    # args.run_name = 'qwen'  #this needs to match prev_{run_name}_memory.json
    # args.device_map='2'
    # args.critic_device_map='3'
    # args.secondary_device_map = '3'
    # args.data_file = 'dataset_building/8_18.csv'
    # run_simulation(args)
    
    # args.critic_size = 'smaller'
    # args.model = 'mistral7instruct_cpp'
    # args.sys = False
    # args.run_name = 'mistral7instructV2'  #this needs to match prev_{run_name}_memory.json
    # args.device_map='3'
    # args.critic_device_map='3'
    # args.secondary_device_map = '3'
    # args.data_file = 'dataset_building/8_18.csv'
    # run_simulation(args)
    
    # args.critic_size = 'smaller'
    # args.model = 'qwen14_cpp' 
    # args.sys = True,
    # args.run_name = 'qwen14'  #this needs to match prev_{run_name}_memory.json
    # args.device_map='3'
    # args.critic_device_map='3'
    # args.secondary_device_map = '3'
    # args.data_file = 'dataset_building/8_18.csv'
    # run_simulation(args)
    
    # args.critic_size = 'smaller'
    # args.model = 'yi_cpp' 
    # args.sys = True,
    # args.run_name = 'yi34Chat'  #this needs to match prev_{run_name}_memory.json
    # args.device_map='2'
    # args.critic_device_map='2'
    # args.secondary_device_map = '2'
    # args.data_file = 'dataset_building/8_11_14_18.csv'
    # run_simulation(args)
    
    # args.critic_size = 'smaller'
    # args.model = 'mixtral_cpp' 
    # args.sys = False
    # args.run_name = 'mixtral'  #this needs to match prev_{run_name}_memory.json
    # args.device_map='2'
    # args.critic_device_map='3'
    # args.secondary_device_map = '3'
    # args.data_file = 'dataset_building/8_18.csv'
    # run_simulation(args)
    
    