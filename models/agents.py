from models.prompts.prompts import *
from models.commons.read_prompt import read_prompt, read_json
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from models.utils import *
from guidance import system, user, assistant, select, gen, silent
import choix

import numpy as np
import pandas as pd
import re

import guidance
import json
import pickle
from collections import defaultdict

from progress.bar import Bar
import os

from summarization import summarize_text, summarize_requirements, summarize_lobby_inputs, compute_length
from models.prompts.sum_prompts import *

from global_vars import token_count, ran_out_context_len, tokenizer, CONTEXT_LENGTH


class Lobbyist:
    def __init__(self, model,
                 sys: bool, separate_critic:bool=False,
                 prev_main_memory=None,
                 resume_trial=None,
                 summarize_req:bool = False,
                 summarize_plan: bool = False,
                 memory_file_path = "",
                 n_amends=3,):
        
        self.d = pd.DataFrame([])
        
        self.i = 0
        self.trial = 0
        self.n_amends = n_amends
        self.resume_amendment = separate_critic
        self.resume_trial = resume_trial
        
        self.sys = sys
        self.summarize_req = summarize_req
        self.summarize_plan = summarize_plan
        
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, 
                                                                        chunk_overlap=0, 
                                                                        separator = " ")
        
        self.memory_file_path = memory_file_path
        self.amendment_store = rec_dict()
        self.feedback_mem = rec_dict()
        self.reflexion_mem = rec_dict()
        
        self.prev_main_memory=prev_main_memory
        
        self.lobbyist_agent = model
        if summarize_req or summarize_plan: self.lm_sum = model
        self.tokenizer = tokenizer


    def reset(self):
        self.lobbyist_agent.reset()
        if hasattr(self, 'lm_sum'): self.lm_sum.reset()
        self.trial=0

    def summarize_bill_business(self, data, data_mem_loc):
        summaries = {}
        for i in self.d.index:
            record = self.d.loc[i]
            if self.d.loc[i]['bill_summarized'] == "":
                print(f"IDX {i} Summarizing bill ...")
                first_valid_index = self.d.bill_summarized.apply(lambda x: x != "").idxmax()    # check if bill already summarized
                if self.d.bill_summarized.loc[first_valid_index] == "":
                    if compute_length(tokenizer, record.summary_text) > int(CONTEXT_LENGTH*.12):
                        self.d.loc[i, 'bill_summarized'] = summarize_lobby_inputs(self.lm_sum, tokenizer, record.official_title, record.summary_text, sys=self.sys)
                    else:
                        print('    Not long enough, skipping summarization and copying the same')
                        self.d.loc[i, 'bill_summarized'] = record.summary_text
                    
                else:
                    print(f"    Pre-summarized bill found, fetching ...")
                    self.d.loc[i, 'bill_summarized'] = self.d.bill_summarized.loc[first_valid_index]
                    
                data.loc[i, 'bill_summarized'] = self.d.loc[i, 'bill_summarized']
                data.to_csv(data_mem_loc, index=False)
            else:
                print(f"IDX {i} | Bill already summarized | Fetching ...")
                
            assert len(self.d.loc[i]['bill_summarized']) > 5
            
            if self.d.loc[i]['business_summarized'] == "":                                      # business is being summarized with respect to bill
                print(f"IDX {i} Summarizing business ..")
                if compute_length(tokenizer, record.business_description) > int(CONTEXT_LENGTH*.12):
                    self.d.loc[i, 'business_summarized'] = summarize_lobby_inputs(self.lm_sum, tokenizer, record.official_title, self.d.loc[i]['bill_summarized'],
                                                                                record.company_name, record.business_description, sys=self.sys)
                else:
                    print('    Not long enough, skipping summarization and copying the same')
                    self.d.loc[i, 'business_summarized'] = record.business_description
                
                data.loc[i, 'business_summarized'] = self.d.loc[i, 'business_summarized']
                data.to_csv(data_mem_loc, index=False)
            else:
                print(f"IDX {i} | Business already summarized | Fetching ...")
            
            assert len(self.d.loc[i]['business_summarized']) > 5
            
            summaries[i] = {'bill': self.d.loc[i, 'bill_summarized'], 'business': self.d.loc[i, 'business_summarized']}
            
        return summaries


    def store_amendments(self, lm):
        amends, benefits = [], []
        try:
            for i in range(self.n_amends):
                amends += [lm[f'amendment_{self.trial}_{i}'].strip()]                
                benefits += [lm[f'benefit_{self.trial}_{i}'].strip()]
                # token_count.output += compute_length(self.tokenizer, benefits[i])
        except:
            print('Exception in storing amendments, ran out context len: ', ran_out_context_len, '; or amendments have been copied from main')
            pass
            
        self.amendment_store[str(self.i)][str(self.trial)]['amendments'] = amends
        self.amendment_store[str(self.i)][str(self.trial)]['benefits'] = benefits
        
        with open(os.path.join(self.memory_file_path, 'amendments.json'), 'w') as f:
            json.dump(self.amendment_store, f, indent=4)
            
    def store_amendments_gemini(self):
        amends, benefits = [], []
        
        cot_amends = self.lobbyist_agent[f'cot_amends_{self.trial}']
        
        self.amendment_store[str(self.i)][str(self.trial)]['amendments'] = amends
        self.amendment_store[str(self.i)][str(self.trial)]['benefits'] = benefits
        
        with open(os.path.join(self.memory_file_path, 'amendments.json'), 'w') as f:
            json.dump(self.amendment_store, f, indent=4)

    # @guidance
    def generate_amends_with_benefits(self, lm, ablation_test=False, use_prev_mem_if_available=True): #resume_amendment:bool
        # with silent():
        try:
            if use_prev_mem_if_available and (self.prev_main_memory is not None) and (len(self.prev_main_memory[str(self.i)])-1 > self.trial):
                with assistant():
                    lm += f"\n\nSUGGESTED AMENDMENTS TRIAL #{self.trial+1}"
                    #replaced gens with memory extractions
                    for i in range(self.n_amends):
                        lm += f"\n\nAMENDMENT #{i+1}: " + self.prev_main_memory[str(self.i)][str(self.trial)]['amendments'][i]
                        lm += "\n\nBenefit: " + self.prev_main_memory[str(self.i)][str(self.trial)]['benefits'][i]
                        
            # elif ((self.resume_trial is not None) and (self.trial < self.resume_trial)):
            #     with assistant():
            #         lm += f"\n\nSUGGESTED AMENDMENTS TRIAL #{self.trial+1}"
            #         #replaced gens with memory extractions
            #         for i in range(self.n_amends):
            #             lm += f"\n\nAMENDMENT #{i+1}: " + self.amendment_store[str(self.i)][str(self.trial)]['amendments'][i]
            #             lm += "\n\nBenefit: " + self.amendment_store[str(self.i)][str(self.trial)]['benefits'][i]
            elif ablation_test:
                with assistant():
                    lm += f"\n\nSUGGESTED AMENDMENTS TRIAL #{self.trial+1}"
                    #replaced gens with memory extractions
                    for i in range(self.n_amends):
                        lm += f"\n\nAMENDMENT #{i+1}: " + self.prev_main_memory[str(self.i)][str(self.trial)]['amendments'][i]
                        lm += "\n\nBenefit: " + self.prev_main_memory[str(self.i)][str(self.trial)]['benefits'][i]
            else:
                with assistant():
                    lm += f"\n\nSUGGESTED AMENDMENTS TRIAL #{self.trial+1}"
                    for i in range(self.n_amends):
                        
                        token_count.input += compute_length(tokenizer, str(lm))
                        # lm += f"\n\nAMENDMENT #{i+1}: " + gen(name=f'amendment_{self.trial}_{i}', stop='Benefit: ', max_tokens=300)
                        lm += f"\n\nAMENDMENT #{i+1}: " + gen(name=f'amendment_{self.trial}_{i}', stop='Benefit:', max_tokens=300)
                        token_count.output += compute_length(tokenizer, lm[f'amendment_{self.trial}_{i}'])
                        
                        token_count.input += compute_length(tokenizer, str(lm))
                        lm += "\nBenefit: " + gen(name=f'benefit_{self.trial}_{i}', stop='\n', max_tokens=300)
                        token_count.output += compute_length(tokenizer, lm[f'benefit_{self.trial}_{i}'])
            

        except Exception as e:
            print('EXCEPTION: ', e, ' | amends ran out context len')
            ran_out_context_len = True
            return lm
        return lm
    
    def generate_cot_amends(self, lm):
        # with silent():
        try:
            with assistant():
                lm += f"\n\nSUGGESTED AMENDMENTS TRIAL #{self.trial+1}"
                for i in range(self.n_amends):
                    
                    token_count.input += compute_length(tokenizer, str(lm))
                    lm += f"\n\nPLAN FOR BENEFIT #{i+1}: " + gen(name=f'benefit_{self.trial}_{i}', stop=f'\nAMENDMENT #{i+1}')
                    token_count.output += compute_length(tokenizer, lm[f'benefit_{self.trial}_{i}'])
                    
                    token_count.input += compute_length(tokenizer, str(lm))
                    lm += f"\nAMENDMENT #{i+1}: " + gen(name=f'amendment_{self.trial}_{i}', stop=f'\n\n')
                    token_count.output += compute_length(tokenizer, lm[f'amendment_{self.trial}_{i}'])
                    
        except Exception as e:
            print('EXCEPTION: ', e, ' | amends ran out context len')
            ran_out_context_len = True
            return lm
        return lm

    def get_amendments(self, cot=False, reflexion=True, critique=True):
        if self.trial == 0:
            if self.sys:
                with system(): self.lobbyist_agent += LOBBYIST_SYS
    
            with user():
                if cot:
                    prompt = LOBBYIST_AMENDMENT_COT if self.sys else (LOBBYIST_SYS + '\n\n' + LOBBYIST_AMENDMENT_COT)     #MIXTRAL_LOBBYIST_AMENDMENT
                    self.lobbyist_agent += prompt.format(official_title=self.d.loc[self.i].official_title,
                                                         summary_text=self.d.loc[self.i].bill_summarized if self.summarize_req else self.d.loc[self.i].summary_text,
                                                         business_name=self.d.loc[self.i].company_name,
                                                         business_description=self.d.loc[self.i].business_summarized if self.summarize_req else self.d.loc[self.i].business_description)
                else:
                    prompt = LOBBYIST_AMENDMENT if self.sys else (LOBBYIST_SYS + '\n\n' + LOBBYIST_AMENDMENT)     #MIXTRAL_LOBBYIST_AMENDMENT
                    self.lobbyist_agent += prompt.format(official_title=self.d.loc[self.i].official_title,
                                                         summary_text=self.d.loc[self.i].bill_summarized if self.summarize_req else self.d.loc[self.i].summary_text,
                                                         business_description=self.d.loc[self.i].business_summarized if self.summarize_req else self.d.loc[self.i].business_description)

        else:
            with user():
                if self.summarize_plan:
                    trial_prompt = LOBBYIST_AMEND_SUM_TRIAL
                elif not reflexion and not critique:
                    trial_prompt = LOBBYIST_AMEND_TRIAL_NO_CRITIQUE_NO_REFLEXION.format(trial_no=self.trial)
                elif not critique:
                    trial_prompt = LOBBYIST_AMEND_TRIAL_NO_CRITIQUE
                elif not reflexion:
                    # trial_prompt = LOBBYIST_NO_REFLEXION.format(trial_no=self.trial+1, company_name=self.d.loc[self.i].company_name,
                    #                                             feedback=self.process_feedback(self.feedback_mem[str(self.i)][str(self.trial-1)])) + LOBBYIST_AMEND_TRIAL_NO_REFLEXION
                    trial_prompt = LOBBYIST_AMEND_TRIAL_NO_REFLEXION.format(trial_no=self.trial,                        # bec there'll be no reflexion, so we add critique here
                                                                            company_name=self.d.loc[self.i].company_name,
                                                                            feedback=self.process_feedback(self.feedback_mem[str(self.i)][str(self.trial-1)]))
                else:
                    trial_prompt = LOBBYIST_AMEND_TRIAL
                self.lobbyist_agent += trial_prompt
        
        lm = self.generate_cot_amends(self.lobbyist_agent) if cot else self.generate_amends_with_benefits(self.lobbyist_agent, ablation_test = True if (((not reflexion) or (not critique)) and self.trial==0) else False, use_prev_mem_if_available=(reflexion and critique))      # use_prev_mem_if_available: used when not ablating 
        
        if (not reflexion or not critique) and self.trial==0:
            self.amendment_store[str(self.i)][str(self.trial)]['amendments'] = self.prev_main_memory[str(self.i)][str(self.trial)]['amendments']
            self.amendment_store[str(self.i)][str(self.trial)]['benefits'] = self.prev_main_memory[str(self.i)][str(self.trial)]['benefits']
        elif (reflexion and critique) and (self.prev_main_memory is not None) and (self.trial < len(self.prev_main_memory[str(self.i)])-1) and (not reflexion or not critique):  # only copy for trial 0 if ablation test
            self.amendment_store[str(self.i)][str(self.trial)]['amendments'] = self.prev_main_memory[str(self.i)][str(self.trial)]['amendments']
            self.amendment_store[str(self.i)][str(self.trial)]['benefits'] = self.prev_main_memory[str(self.i)][str(self.trial)]['benefits']
        
        # if not( or ((self.resume_trial is not None) and (self.trial < self.resume_trial))):
        else:
            self.store_amendments(lm)
            
        return lm

    def receive_feedback(self, critic):
        self.feedback_mem[str(self.i)][str(self.trial)]['keywords'] = critic.critique_mem[str(self.i)][str(self.trial)]['keywords']
        self.feedback_mem[str(self.i)][str(self.trial)]['reasoning'] = critic.critique_mem[str(self.i)][str(self.trial)]['reasoning']
        if self.summarize_plan:
            crit = "\nSUSPICIOUS KEYWORDS:\n".join([critic.critique_mem[str(self.i)][str(self.trial)]['reasoning'],
                                                    critic.critique_mem[str(self.i)][str(self.trial)]['keywords']])
            if compute_length(tokenizer, crit) > int(CONTEXT_LENGTH*.08):
                print(f"IDX {self.i} Trial {self.trial} | summarizing critique")
                self.feedback_mem[str(self.i)][str(self.trial)]['summary'] = summarize_text(self.lm_sum, crit, CRITIQUE_SUM_PROMPT, self.sys, SYS_CRIT_SUM_PROMPT)
            else:
                print(f'IDX {self.i} Trial {self.trial} | Skipping summarizing critique | Token length:', compute_length(tokenizer, crit))
                self.feedback_mem[str(self.i)][str(self.trial)]['summary'] = crit
                self.feedback_mem[str(self.i)][str(self.trial)]['flag_summarized'] = False

    def process_feedback(self, feedback: dict):
        return f"""\nREASONING: {feedback['reasoning'].strip()}\n\nSUSPICIOUS KEYWORDS: {feedback['keywords'].strip()}"""
    
    # @guidance
    def generate_reflexion(self, lm, critique=True):
        # with silent():
        token_count.input += compute_length(tokenizer, str(lm))
        
        if critique and (self.prev_main_memory is not None) and (self.trial < len(self.prev_main_memory[str(self.i)])-2):
            # if self.prev_main_memory[self.i][self.trial]['detected']:
            with assistant():
                temp = lm + "DIAGNOSIS AND PLAN: " + self.prev_main_memory[str(self.i)][str(self.trial)]['reflexion']
            self.reflexion_mem[str(self.i)][str(self.trial)]['plan'] = self.prev_main_memory[str(self.i)][str(self.trial)]['reflexion']
        
        # elif (self.resume_trial is not None) and (self.trial < self.resume_trial-1):      # FOR resuming for further trials
        #     with assistant():
        #         temp = lm + "DIAGNOSIS AND PLAN: " + self.reflexion_mem[str(self.i)][str(self.trial)]['plan']
        else:
            with assistant():
                temp = lm + "DIAGNOSIS AND PLAN: " + gen(name=f'plan_{self.trial}', stop=['\n\n\n\n', 'AMENDMENT'], max_tokens=400)
            self.reflexion_mem[str(self.i)][str(self.trial)]['plan'] = temp[f'plan_{self.trial}'].strip()
        token_count.output += compute_length(tokenizer, self.reflexion_mem[str(self.i)][str(self.trial)]['plan'])
        
        if self.summarize_plan:
            if compute_length(tokenizer, self.reflexion_mem[str(self.i)][str(self.trial)]['plan']) > int(CONTEXT_LENGTH*.08):  # .08 make these factors custom later
                print(f"IDX {self.i} Trial {self.trial} | summarizing reflexion ..")
                self.reflexion_mem[str(self.i)][str(self.trial)]['summary'] = summarize_text(self.lm_sum, self.reflexion_mem[str(self.i)][str(self.trial)]['plan'],
                                                                               REFLEXION_SUM_PROMPT, self.sys, SYS_REF_SUM_PROMPT)
                
                with assistant():
                    lm += "\n\nDIAGNOSIS AND PLAN:\n{sum_ref}".format(trial_no=self.trial+1,
                                                                    company_name=self.d.loc[self.i].company_name,
                                                                    feedback=self.feedback_mem[str(self.i)][str(self.trial)]['summary'] if self.summarize_plan 
                                                                            else self.process_feedback(self.feedback_mem[str(self.i)][str(self.trial)]),
                                                                    sum_ref=self.reflexion_mem[str(self.i)][str(self.trial)]['summary' if self.summarize_plan else 'plan'])

                return lm
            
            else:
                print(f'IDX {self.i} Trial {self.trial} | Skipping summarizing critique | Token length:', compute_length(tokenizer, self.reflexion_mem[str(self.i)][str(self.trial)]['plan']))
                self.reflexion_mem[str(self.i)][str(self.trial)]['summary'] = self.reflexion_mem[str(self.i)][str(self.trial)]['plan']
                self.reflexion_mem[str(self.i)][str(self.trial)]['flag_summarized'] = False
                
                return temp
        

        else: return temp

    def get_reflexion(self, lm, critique=True):
        # with silent():
        with user():
            if not critique:
                prompt = LOBBYIST_REFLEXION_NO_CRITIQUE
                lm += prompt.format(trial_no=self.trial+1,
                                    company_name=self.d.loc[self.i].company_name)
            else:
                prompt = LOBBYIST_REFLEXION_SUM if self.summarize_plan else LOBBYIST_REFLEXION
                lm += prompt.format(trial_no=self.trial+1,
                                    company_name=self.d.loc[self.i].company_name,
                                    feedback=self.feedback_mem[str(self.i)][str(self.trial)]['summary'] if self.summarize_plan 
                                            else self.process_feedback(self.feedback_mem[str(self.i)][str(self.trial)]))
            
        lm = self.generate_reflexion(lm, critique=critique)
    
        with open(os.path.join(self.memory_file_path, 'reflexion.json'), 'w') as f:
            json.dump(self.reflexion_mem, f, indent=4)
        
        return lm
        
    def summarize_trial(self, lm):
        with user():
            trial_sum = lm + SUM_TRIAL_PROMPT.format(trial_no=self.trial+1)
        
        with assistant():
            trial_sum += gen('trial_sum', stop='\n\n\n\n')
        

class CriticLobbyist:  
    def __init__(self, sys: bool,
                 model, 
                 sec_sys=None,
                 model_sec=None,
                 prev_main_memory=None,
                 resume_trial=None,
                 summarize_req: bool = False,
                 summarize_plan: bool = False,
                 memory_file_path: str = "",):
        
        self.d = pd.DataFrame([])
        
        self.critique_mem = rec_dict()
        self.memory_file_path = memory_file_path
        
        self.prev_main_memory = prev_main_memory if prev_main_memory is not None else None
        
        self.i = 0
        self.trial = 0
        self.amendments = ""
        self.resume_trial = resume_trial
        
        self.sys = sys
        self.sec_sys = sec_sys
        
        self.summarize_req = summarize_req
        self.summarize_plan = summarize_plan
        
        self.separate_critic = model_sec is not None
        
        self.company_pairs = {}
        self.reverse_keyword_map = {}

        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, 
                                                                        chunk_overlap=0, 
                                                                        separator = " ")
        
        self.benefactor_id_map = {}
        self.id_benefactor_map = {}
        

        self.lm_pair = model_sec if model_sec is not None else model #+ CRITIC_PAIWISE_SYS_PROMPT
        self.lm_critique = model #+ CRITIQUE_SYS

    def reset(self):
        self.lm_pair.reset()
        self.lm_critique.reset()
        
        self.trial=0

    def process_requirements(self):
        self.company_pairs = list(combinations(list(set(self.d.company_name)), 2)) #pairwise_combinations_with_keywords(list(set(self.d.company_name)))
        self.reverse_keyword_map = get_fullname_from_keyword_map(list(set(self.d.company_name)))
        self.benefactor_id_map = fullname_id_map(list(set(self.d.company_name)))
        self.id_benefactor_map = {v:k for k,v in self.benefactor_id_map.items()}

    def detect_pair_benefactor(self, pair_benefactors):
        company1, company2 = pair_benefactors #self.company_pairs[pair_benefactors]
        if self.sec_sys is not None:
            if self.sec_sys:
                with system():
                    prompted = self.lm_pair + CRITIC_PAIWISE_SYS_PROMPT
        elif self.sys:
            with system():
                    prompted = self.lm_pair + CRITIC_PAIWISE_SYS_PROMPT
        
        
        with user():
            prompted = (prompted if (self.sys or (self.sec_sys is not None and self.sec_sys)) else self.lm_pair) + (CRITIC_PAIRWISE_COMPARISON_PROMPT if (self.sys or self.sec_sys) else
                        CRITIC_PAIWISE_SYS_PROMPT + '\n\n' + CRITIC_PAIRWISE_COMPARISON_PROMPT).format(official_title=self.d.loc[self.i].official_title,
                                                                    summary_text=self.d.loc[self.i].bill_summarized if self.summarize_req
                                                                                 else self.d.loc[self.i].summary_text,
                                                                    amendments=self.amendments,
                                                                    company_pair_detail=pair_details_prompt(self.d, pair_benefactors, self.summarize_req),
                                                                    company1=extract_single_word_name(company1).upper(), company2=extract_single_word_name(company2).upper()) #removed .upper() from company[1/2] for now

        token_count.input += compute_length(tokenizer, str(prompted))
        with assistant():
            answer = prompted + select([extract_single_word_name(company1).upper(), extract_single_word_name(company2).upper()], name='benefactor')     # removed company[1/2].upper() for now
        
        return self.reverse_keyword_map[answer['benefactor']]
    
    def pairwise_comparisons(self):
        pair_winners = {}
        
        bar= Bar('Pairwise comparisons', max=len(self.company_pairs))
        for pair in self.company_pairs:
            pair_winners[pair] = self.detect_pair_benefactor(pair)
            bar.next()
        bar.finish()
        return pair_winners
    
    def get_lamp(self):
        pair_winners = self.pairwise_comparisons()
        
        competitions = []
        for competitors in pair_winners:
            competitions += [(self.benefactor_id_map[pair_winners[competitors]], 
                            self.benefactor_id_map[competitors[competitors[0]==pair_winners[competitors]]])]
        scores = choix.ilsr_pairwise(len(self.benefactor_id_map), competitions, alpha=0.01)
        return scores
    
    def detect_benefactors(self,) -> bool:
        """
        Does have access to the actual dataset
        hence benefactor company name
        
        Returns:exit
            list: top-k benefactors names
        """
        scores = self.get_lamp()
        topk_benefactor_ids = np.argsort(scores)
        topk_benefactor_names = [self.id_benefactor_map[benefactor_id] for benefactor_id in topk_benefactor_ids]
        return topk_benefactor_names, sorted(scores) #== self.d.company_name[self.i]
    
    def process_critique(self, critique: str):
        pass
    
    def generate_critique(self, lm, use_prev_mem_if_available=True):
        token_count.input += compute_length(tokenizer, str(lm))
        
        if use_prev_mem_if_available and (self.prev_main_memory is not None) and (self.trial < len(self.prev_main_memory[str(self.i)])-2):# and self.prev_main_memory[self.i][self.trial]['detected']:
        # if (self.resume_trial is not None) and (self.trial < self.resume_trial-1):
            with assistant():
                lm += '\nREASONING: ' + self.prev_main_memory[str(self.i)][str(self.trial)]['critique']
                
                token_count.input += compute_length(tokenizer, str(lm))
                lm += 'SUSPICIOUS KEYWORDS: ' + self.prev_main_memory[str(self.i)][str(self.trial)]['critique_keywords']
        else:
            with assistant():
                lm += '\nREASONING: ' + gen(name=f'reasoning_{self.trial}', suffix='\n', stop='SUSPICIOUS KEYWORDS:', max_tokens=500)
                
                token_count.input += compute_length(tokenizer, str(lm))
                lm += 'SUSPICIOUS KEYWORDS: ' + gen(name=f'keywords_{self.trial}', stop='\n\n\n\n', max_tokens=150)
        
            token_count.output += compute_length(tokenizer, lm[f'reasoning_{self.trial}'] + lm[f'keywords_{self.trial}'])
        
        return lm
    
    def get_critique(self, lm, benefactor_name: str, use_prev_mem_if_available=True):
        if self.trial == 0:
            if self.sys:
                with system(): lm += CRITIQUE_SYS
                
            with user():
                prompt = CRITIC_CRITIQUE if self.sys else (CRITIQUE_SYS + '\n\n' + CRITIC_CRITIQUE)
                lm += prompt.format(company_detail=f"""{benefactor_name}: {self.d.loc[self.i].business_summarized if self.summarize_req
                                            else self.d.loc[self.i].business_description}""",
                                official_title=self.d.official_title[self.i],
                                summary_text=self.d.loc[self.i].bill_summarized
                                            if self.summarize_req
                                            else self.d.loc[self.i].summary_text,
                                amendments=self.amendments,
                                company_name=benefactor_name,
                                trial_no=self.trial+1)

        else:
            with user():
                lm += CRITIC_CRITIQUE_TRIAL.format(company_name=benefactor_name,
                                            trial_no=self.trial+1,
                                            amendments=self.amendments)
        lm = self.generate_critique(lm, use_prev_mem_if_available)

        if use_prev_mem_if_available and (self.prev_main_memory is not None) and (self.trial < len(self.prev_main_memory[str(self.i)])-2):
            self.critique_mem[str(self.i)][str(self.trial)]['reasoning'] = self.prev_main_memory[str(self.i)][str(self.trial)]['critique']
            self.critique_mem[str(self.i)][str(self.trial)]['keywords'] = self.prev_main_memory[str(self.i)][str(self.trial)]['critique_keywords']
        # if not ((self.resume_trial is not None) and (self.trial < self.resume_trial-1)):
        else:
            self.critique_mem[str(self.i)][str(self.trial)]['keywords'] = lm[f'keywords_{self.trial}'].strip()
            self.critique_mem[str(self.i)][str(self.trial)]['reasoning'] = lm[f'reasoning_{self.trial}'].strip()
        
        with open(os.path.join(self.memory_file_path, 'critique_memory.json'), 'w') as f:
            json.dump(self.critique_mem, f, indent=4)
        
        return lm
    
    def receive_amendments(self, lobbyist):
        amends = lobbyist.amendment_store[str(self.i)][str(self.trial)]['amendments']
        amend_str = ""
        
        for i, amend in enumerate(amends):
            amend_str += f"AMENDMENT #{i+1}: {amend}\n"
        self.amendments = amend_str