import json
import numpy as np
from guidance import models, system, user, assistant, select
from models.prompts.prompts import *
import llama_cpp
from models.models import *
import pandas as pd
import itertools
import argparse

from progress.bar import Bar

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The name of the run", default='yi')

def utility_verification():
    
    # if args.model == 'yi':
        
    # elif args.model == 'qwen'
    lm =  QwenChat('llms/qwen1_5-72b-chat-q5_k_m.gguf', n_gpu_layers=-1,
                    n_ctx=6000, echo=False, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=3)
    
    with system():
            utility = lm + CHECK_UTILITY_SYS
    
    
    with open('memory/mixtral_run_14_16_18/main.json', 'r') as f:
        main = json.load(f)
    with open('memory/mixtral_run_14_16_18/detection.json', 'r') as f:
        detection = json.load(f)
    
    data_path = {8: 'dataset_building/8_11/selected_8_11.csv',
                 12: 'dataset_building/gt112_except14_16_18/gt112.csv',
                 14: 'dataset_building/14_16_18/dataset_cong_14_16_18.csv'}
    
    data = pd.read_csv(data_path[14])
    bar = Bar('Utility Verifications', max=len(data))

    detection = np.array(detection).sum(-1).sum(-1)
    # detection_str = [str(x) for x in list(np.where(detection < 4))[0]]


    tested = []
    checks = {}
    with open('utility_verification_mixtral_14_initial_intent.json', 'r') as f: #resuming case
        checks = json.load(f)
        
    for i in range(315, data.__len__()):
        try:
            detections = detection[i]
            check = []
            # if detections < 3:
            tested.append(i)
            
            main_i = main[str(i)]
        
            iter = min(detections,3)    #cuz we consider only 3 trials
            
            iter = iter + 1 if iter != 3 else iter  #
            for t in range(iter):
                check.append([])
                for (a,b) in list(itertools.product([0,1,2], [0,1,2])):
                    with user():
                        utility_check = utility + CHECK_UTILITY.format(business=data.loc[i].business_description,
                                                                    bill=data.loc[i].summary_text,
                                                                    amendment=main_i[str(t)]['amendments'][a],
                                                                    benefit=main_i["0"]['benefits'][b])   #always checking against initial benefits
                    with assistant():
                        utility_check = utility_check + select(['YES', 'NO'], name='check')
                    check[t].append(int(utility_check['check']=='YES'))            
            checks[i] = check
        
            with open('utility_verification_mixtral_14_initial_intent.json', 'w') as f:
                json.dump(checks, f, indent=4)
        except:
            continue
            
        bar.next()

if __name__ == '__main__':
    # args = get_args()
    utility_verification()