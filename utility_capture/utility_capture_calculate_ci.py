from tqdm import tqdm
import itertools
import numpy as np
import scipy.stats
import json
import sklearn as sk


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    # m, se = np.mean(a, axis=-1), scipy.stats.sem(a, axis=-1)
    m, se = np.mean(a, axis=-1), scipy.stats.tstd(a, axis=-1)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return [[m, m-h, m+h], se]

top1 = True     # conditions set for top1

ci_iterations = 10000

mains_map = {'mistral7': 'mistral7instructV2', 'qwen7': 'qwen7', 'qwen14': 'qwen14','yi': 'yiChat', 'mixtral': 'mixtral', 'qwen': 'qwen'}
utility_capture = {}

for model in ['mistral7', 'qwen7', 'qwen14', 'yi', 'mixtral', 'qwen']:
    
    memory_paths = ['/home/ic38970/reinforced_lobbying/memory/{model}_run_8_11/main.json', '/home/ic38970/reinforced_lobbying/memory/{model}_run_gt112/main.json', '/home/ic38970/reinforced_lobbying/memory/{model}_run_14_16_18/main.json']
    detection_paths = ['/home/ic38970/reinforced_lobbying/memory/{model}_run_8_11/detection.json', '/home/ic38970/reinforced_lobbying/memory/{model}_run_gt112/detection.json', '/home/ic38970/reinforced_lobbying/memory/{model}_run_14_16_18/detection.json']
    
    # mains
    mains = []
    if model == 'yi':
        memory_paths = [memory_paths[i] for i in [0,2]]
    for path in memory_paths:
        with open(path.format(model=mains_map[model]), 'r') as f:
            mains += [json.load(f)]
    
    
    utility_paths = ['utility_verification_{model}_8_initial_intent.json', 'utility_verification_{model}_12_initial_intent.json', 'utility_verification_{model}_14_initial_intent.json']
    
    splits = 3
    if model == 'yi':
        utility_paths = [utility_paths[i] for i in [0,2]]
        splits=2
        
        
    utilities = []
    for u_path in utility_paths:
        with open(u_path.format(model=model), 'r') as f:
            utilities += [json.load(f)]
    
    
    # detections
    if top1:
        detections = []
        if model == 'yi':
            detection_paths = [detection_paths[i] for i in [0,2]]
        for path in detection_paths:
            with open(path.format(model=mains_map[model]), 'r') as f:
                detections += [json.load(f)]
        
        dt_top1 = []
        dt_top1_not_decept_sim_index = []
        for split in range(splits):
            # try:
            dt = np.array(detections[split])
            # except:
            #     print(split)
            for i in range(dt[:, :, 0].shape[0]):
                for j in range(dt[i, :, 0].shape[0]-2, -1, -1):
                    if not dt[i, :, 0][j]:
                        dt[i, :, 0][j:] = 0
                        
            dt_top1.append(dt[:, :, 0].copy())
            dt_top1_not_decept_sim_index.append(np.where(np.all(dt_top1[-1], axis=-1))[0])
    
    
    # overall utility capture
    percent_capture_trial = [[], [], []]
    
    for split in range(splits):
        
        for i in range(3):
            amend_benefit=[]
            for k, v in utilities[split].items():
                # if top1:
                #     if int(k) in dt_top1_index[split]:
                #         if len(v) >= i+1: amend_benefit.append(v[i])
                # else:
                if len(v) >= i+1: amend_benefit.append(v[i])
                    
            percent_capture_trial[i] += amend_benefit
    
    utility_capture[model] = {'overall_capture': [], 'deceptive_sims': [], 'particular_trials': []}
    
    for trial in range(3):
        boot_captures = []
        for seed in tqdm(range(ci_iterations), desc=f'{model}_trial{trial}_overall_capture'):
            boot = sk.utils.resample(percent_capture_trial[trial], replace=True, n_samples=percent_capture_trial[trial].__len__(), random_state=seed)
            boot_captures.append(np.bitwise_or.reduce(np.array(boot).reshape(-1, 3, 3).transpose(0, 2, 1), axis=-1).mean(-1).mean())
        ci = mean_confidence_interval(boot_captures)
        ci = [np.bitwise_or.reduce(np.array(percent_capture_trial[trial]).reshape(-1, 3, 3).transpose(0, 2, 1), axis=-1).mean(-1).mean()] + ci
        utility_capture[model]['overall_capture'].append(ci)        
    
    
    # only those simulations that were deceptive
    percent_capture_trial = [[], [], []]
    
    for split in range(splits):
        for i in range(len(utilities[split])):
            
            condition = not(int(i) in dt_top1_not_decept_sim_index[split]) if top1 else True
            if condition:
                trials = len(mains[split][str(i)]) - 1
                
                if trials < 3: 
                    for j in range(trials):
                            try:
                                # print(split, j, i)
                                percent_capture_trial[j].append(utilities[split][str(i)][j])
                            except:
                                # print(split, j, i)
                                continue
                if trials >= 3:
                    if not mains[split][str(i)][str(2)]['detected']:
                        for j in range(3):
                            try:
                                percent_capture_trial[j].append(utilities[split][str(i)][j])
                            except:
                                # print(split, j, i)
                                continue
    
    # utility_capture[model] = {'deceptive_sims': []}
    for trial in range(3):
        boot_captures = []
        for seed in tqdm(range(ci_iterations), desc=f'{model}_trial{trial}_deceptive_trials'):
            boot = sk.utils.resample(percent_capture_trial[trial], replace=True, n_samples=percent_capture_trial[trial].__len__(), random_state=seed)
            boot_captures.append(np.bitwise_or.reduce(np.array(boot).reshape(-1, 3, 3).transpose(0, 2, 1), axis=-1).mean(-1).mean())
        ci = mean_confidence_interval(boot_captures)
        ci = [np.bitwise_or.reduce(np.array(percent_capture_trial[trial]).reshape(-1, 3, 3).transpose(0, 2, 1), axis=-1).mean(-1).mean()] + ci
        utility_capture[model]['deceptive_sims'].append(ci)
    
    
    # particularly deceptive trials
    if model =='yi':
        deceived_t1 = [[],[]]
        deceived_t2 = [[],[]]
        deceived_t3 = [[],[]]
    else:
        deceived_t1 = [[],[],[]]
        deceived_t2 = [[],[],[]]
        deceived_t3 = [[],[],[]]

    for split in range(splits):
        for i in range(len(utilities[split])):
            
            # condition = int(i) in dt_top1_index[split] if top1 else True
            # if condition:
            try:
                if top1:
                    trials = np.argmax(dt_top1[split][i]==0)
                    if np.all(dt_top1[split][i], axis=-1) and trials==0:
                        trials=0
                    else:
                        trials += 1
                        
                    if trials == 1:
                        deceived_t1[split].append(utilities[split][str(i)][trials-1])
                    elif trials == 2:
                        deceived_t2[split].append(utilities[split][str(i)][trials-1])
                    elif trials == 3:
                            deceived_t3[split].append(utilities[split][str(i)][trials-1])
                    
                else:
                    trials = len(mains[split][str(i)]) - 1
                
                # if trials < 4:      #TODO IMPORTANT: change for rest, only qwen-72b has 4 trials
                    if trials == 1:
                        deceived_t1[split].append(utilities[split][str(i)][trials-1])
                    elif trials == 2:
                        deceived_t2[split].append(utilities[split][str(i)][trials-1])
                    elif trials == 3:
                        if not mains[split][str(i)][str(2)]['detected']:
                            deceived_t3[split].append(utilities[split][str(i)][trials-1])
            except:
                continue
                    
    deceived_trials = [deceived_t1, deceived_t2, deceived_t3]
    
    # utility_capture[model] = {'particular_trials': []}
    for idx, trial in enumerate(deceived_trials):
        boot_captures = []
        for seed in tqdm(range(ci_iterations), desc=f'{model}_trial{idx}_particular_trials'):
            boot = sk.utils.resample(trial, replace=True, n_samples=trial.__len__(), random_state=seed)
            boot_captures.append(np.bitwise_or.reduce(np.array(list(itertools.chain.from_iterable(boot))).reshape(-1, 3, 3).transpose(0, 2, 1), axis=-1).mean(-1).mean())
        ci = mean_confidence_interval(boot_captures)
        ci = [np.bitwise_or.reduce(np.array(list(itertools.chain.from_iterable(trial))).reshape(-1, 3, 3).transpose(0, 2, 1), axis=-1).mean(-1).mean()] + ci
        utility_capture[model]['particular_trials'].append(ci)



with open('utility_capture_ci_10k_top1.json', 'a') as f:
    json.dump(utility_capture, f, indent=4)
    f.write('\n')