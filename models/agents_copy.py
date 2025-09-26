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
    def __init__(self,
                 sys: bool,
                 model,
                 memory_file_path = "",
                 n_amends=3,):
        
        self.sys = sys
        
        self.d = pd.DataFrame([])
        
        self.i = 0
        self.trial = 0
        self.n_amends = n_amends
        
        self.memory_file_path = memory_file_path
        self.amendment_store = rec_dict()
        self.feedback_mem = rec_dict()
        self.reflexion_mem = rec_dict()
        
        self.lobbyist_agent = model
        self.lobbyist_agent_ablate_replan = model


    def reset(self):
        self.lobbyist_agent.reset()
        if hasattr(self, 'lm_sum'): self.lm_sum.reset()
        self.trial=0

    def store_amendments(self, lm, ablate_replan=False):
        amends, benefits = [], []
        try:
            for i in range(self.n_amends):
                amends += [lm[f'amendment_{self.trial}_{i}'].strip()]                
                benefits += [lm[f'benefit_{self.trial}_{i}'].strip()]
                # token_count.output += compute_length(self.tokenizer, benefits[i])
        except:
            print('Exception in storing amendments, ran out context len: ', ran_out_context_len, '; or amendments have been copied from main')
            pass
        
        if ablate_replan:
            self.amendment_store[str(self.i)][str(self.trial)]['amendments_ablate'] = amends
            self.amendment_store[str(self.i)][str(self.trial)]['benefits_ablate'] = benefits
        else:    
            self.amendment_store[str(self.i)][str(self.trial)]['amendments'] = amends
            self.amendment_store[str(self.i)][str(self.trial)]['benefits'] = benefits
        
        
        with open(os.path.join(self.memory_file_path, 'amendments.json'), 'w') as f:
            json.dump(self.amendment_store, f, indent=4)
            

    def generate_amends_with_benefits(self, lm): #resume_amendment:bool
        try:
            with assistant():
                lm += f"\n\nSUGGESTED AMENDMENTS TRIAL #{self.trial+1}"
                for i in range(self.n_amends):
                    
                    lm += f"\n\nAMENDMENT #{i+1}: " + gen(name=f'amendment_{self.trial}_{i}', stop='Benefit:', max_tokens=300)
                    lm += "\nBenefit: " + gen(name=f'benefit_{self.trial}_{i}', stop='\n', max_tokens=300)

        except Exception as e:
            print('EXCEPTION: ', e, ' | amends ran out context len')
            ran_out_context_len = True
            return lm
        return lm
    

    def get_amendments(self, not_deceptive=True):
        
        if self.trial == 0:
            if self.sys:
                with system(): self.lobbyist_agent += LOBBYIST_SYS
    
            with user():
                    amendment_prompt = LOBBYIST_AMENDMENT if not not_deceptive else LOBBYIST_AMENDMENT_NOT_DECEPTIVE
                    prompt = amendment_prompt if self.sys else (LOBBYIST_SYS + '\n\n' + amendment_prompt)
                    self.lobbyist_agent += prompt.format(official_title=self.d.loc[self.i].official_title,
                                                         summary_text=self.d.loc[self.i].summary_text,
                                                         business_name=self.d.loc[self.i].company_name,
                                                         business_description=self.d.loc[self.i].business_description)
                    
            lm = self.generate_amends_with_benefits(self.lobbyist_agent,)
            self.store_amendments(lm)
            
            return lm#, lm
            

        else:
            with user():
                trial_prompt = (LOBBYIST_AMEND_TRIAL if not not_deceptive else LOBBYIST_AMEND_TRIAL_NO_DECEPTION).format(trial_no=self.trial)
                # trial_prompt_ablate_replan = LOBBYIST_AMEND_TRIAL_NO_CRITIQUE_NO_REFLEXION.format(trial_no=self.trial)
                
                self.lobbyist_agent += trial_prompt
                # self.lobbyist_agent_ablate_replan += trial_prompt_ablate_replan
        
            lm = self.generate_amends_with_benefits(self.lobbyist_agent,)
            # lm_ablate_replan = self.generate_amends_with_benefits(self.lobbyist_agent_ablate_replan,)
        
        
            self.store_amendments(lm)
            # self.store_amendments(lm_ablate_replan, ablate_replan=True)
            
            return lm#, lm_ablate_replan

    def receive_feedback(self, critic):
        self.feedback_mem[str(self.i)][str(self.trial)]['keywords'] = critic.critique_mem[str(self.i)][str(self.trial)]['keywords']
        self.feedback_mem[str(self.i)][str(self.trial)]['reasoning'] = critic.critique_mem[str(self.i)][str(self.trial)]['reasoning']

    def process_feedback(self, feedback: dict):
        return f"""\nREASONING: {feedback['reasoning'].strip()}\n\nSUSPICIOUS KEYWORDS: {feedback['keywords'].strip()}"""
    
    # @guidance
    def generate_reflexion(self, lm,):
        # with silent():
        
        with assistant():
            temp = lm + "DIAGNOSIS AND PLAN: " + gen(name=f'plan_{self.trial}', stop=['\n\n\n\n', 'AMENDMENT'], max_tokens=400)
        self.reflexion_mem[str(self.i)][str(self.trial)]['plan'] = temp[f'plan_{self.trial}'].strip()
        token_count.output += compute_length(tokenizer, self.reflexion_mem[str(self.i)][str(self.trial)]['plan'])
        
        return temp

    def get_reflexion(self, lm, critique=True):
        # with silent():
        with user():
            if not critique:
                prompt = LOBBYIST_REFLEXION_NO_CRITIQUE
                lm += prompt.format(trial_no=self.trial+1,
                                    company_name=self.d.loc[self.i].company_name)
            else:
                prompt = LOBBYIST_REFLEXION
                lm += prompt.format(trial_no=self.trial+1,
                                    company_name=self.d.loc[self.i].company_name,
                                    feedback= self.process_feedback(self.feedback_mem[str(self.i)][str(self.trial)]))
        
        lm = self.generate_reflexion(lm,)
    
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
                 memory_file_path: str = "",):
        
        self.sys = sys
        
        self.d = pd.DataFrame([])
        
        self.critique_mem = rec_dict()
        self.memory_file_path = memory_file_path
        
        self.i = 0
        self.trial = 0
        self.amendments = ""
        self.amendments_ablate_plan = ""
        
        self.company_pairs = {}
        self.reverse_keyword_map = {}
        
        self.benefactor_id_map = {}
        self.id_benefactor_map = {}
        

        self.lm_pair = model #+ CRITIC_PAIWISE_SYS_PROMPT
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

    def detect_pair_benefactor(self, pair_benefactors, ablate_plan=False):
        company1, company2 = pair_benefactors #self.company_pairs[pair_benefactors]

        if self.sys:
            with system():
                    prompted = self.lm_pair + CRITIC_PAIWISE_SYS_PROMPT
        
        
        with user():
            prompted = (prompted if self.sys else self.lm_pair) + (CRITIC_PAIRWISE_COMPARISON_PROMPT if self.sys else
                        CRITIC_PAIWISE_SYS_PROMPT + '\n\n' 
                        + CRITIC_PAIRWISE_COMPARISON_PROMPT).format(official_title=self.d.loc[self.i].official_title,
                                                                    summary_text=self.d.loc[self.i].bill_summarized,
                                                                    amendments=self.amendments if not ablate_plan else self.amendments_ablate_plan,
                                                                    company_pair_detail=pair_details_prompt(self.d, pair_benefactors),
                                                                    company1=extract_single_word_name(company1).upper(), company2=extract_single_word_name(company2).upper()) #removed .upper() from company[1/2] for now

        token_count.input += compute_length(tokenizer, str(prompted))
        with assistant():
            answer = prompted + select([extract_single_word_name(company1).upper(), extract_single_word_name(company2).upper()], name='benefactor')     # removed company[1/2].upper() for now
        
        return self.reverse_keyword_map[answer['benefactor']]
    
    def pairwise_comparisons(self, ablate_plan=False):
        pair_winners = {}
        
        bar= Bar('Pairwise comparisons', max=len(self.company_pairs))
        for pair in self.company_pairs:
            pair_winners[pair] = self.detect_pair_benefactor(pair, ablate_plan)
            bar.next()
        bar.finish()
        return pair_winners
    
    def get_lamp(self, ablate_plan=False):
        pair_winners = self.pairwise_comparisons(ablate_plan)
        
        competitions = []
        for competitors in pair_winners:
            competitions += [(self.benefactor_id_map[pair_winners[competitors]], 
                            self.benefactor_id_map[competitors[competitors[0]==pair_winners[competitors]]])]
        scores = choix.ilsr_pairwise(len(self.benefactor_id_map), competitions, alpha=0.01)
        
        return scores
    
    def detect_benefactors(self, ablate_plan=False) -> bool:
        """
        Does have access to the actual dataset
        hence benefactor company name
        
        Returns:exit
            list: top-k benefactors names
        """
        
        scores = self.get_lamp(ablate_plan)
        topk_benefactor_ids = np.argsort(scores)
        topk_benefactor_names = [self.id_benefactor_map[benefactor_id] for benefactor_id in topk_benefactor_ids]
        
        return topk_benefactor_names, sorted(scores)
    
    
    def process_critique(self, critique: str):
        pass
    
    def generate_critique(self, lm,):
        
        with assistant():
            lm += '\nREASONING: ' + gen(name=f'reasoning_{self.trial}', suffix='\n', stop='SUSPICIOUS KEYWORDS:', max_tokens=500)
            
            token_count.input += compute_length(tokenizer, str(lm))
            lm += 'SUSPICIOUS KEYWORDS: ' + gen(name=f'keywords_{self.trial}', stop='\n\n\n\n', max_tokens=150)
        
        return lm
    
    def get_critique(self, lm, benefactor_name: str,):
        if self.trial == 0:
            if self.sys:
                with system(): lm += CRITIQUE_SYS
                
            with user():
                prompt = CRITIC_CRITIQUE if self.sys else (CRITIQUE_SYS + '\n\n' + CRITIC_CRITIQUE)
                lm += prompt.format(company_detail=f"""{benefactor_name}: {self.d.loc[self.i].business_description}""",
                                official_title=self.d.official_title[self.i],
                                summary_text=self.d.loc[self.i].summary_text,
                                amendments=self.amendments,
                                company_name=benefactor_name,
                                trial_no=self.trial+1)

        else:
            with user():
                lm += CRITIC_CRITIQUE_TRIAL.format(company_name=benefactor_name,
                                            trial_no=self.trial+1,
                                            amendments=self.amendments)
        lm = self.generate_critique(lm)

        self.critique_mem[str(self.i)][str(self.trial)]['keywords'] = lm[f'keywords_{self.trial}'].strip()
        self.critique_mem[str(self.i)][str(self.trial)]['reasoning'] = lm[f'reasoning_{self.trial}'].strip()
        
        with open(os.path.join(self.memory_file_path, 'critique_memory.json'), 'w') as f:
            json.dump(self.critique_mem, f, indent=4)
        
        return lm
    
    def receive_amendments(self, lobbyist, ablate_plan=False):
        amends = lobbyist.amendment_store[str(self.i)][str(self.trial)]['amendments']
        amend_str = ""
        for i, amend in enumerate(amends):
            amend_str += f"AMENDMENT #{i+1}: {amend}\n"
        self.amendments = amend_str
        
        if ablate_plan:
            amends_ablate_plan = lobbyist.amendment_store[str(self.i)][str(self.trial)]['amendments_ablate']
            amend_str = ""
            for i, amend in enumerate(amends_ablate_plan):
                amend_str += f"AMENDMENT #{i+1}: {amend}\n"
            self.amendments_ablate_plan = amend_str