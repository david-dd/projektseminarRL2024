import io
import json
import os.path
from os import mkdir, path
 
root = 'experiments'
if not os.path.exists(root):
    mkdir(root)
 
stngrps = [
    "<Diffusion_FE_120>",      
]
 
for seed in [0]:
    for dataset, dispatcher in [('HVLM', 'fifo')]:
        for action_count in [9]:
            for training_period in [630]:
                for reward in [2]:          # set the reward fuction to use see gym/environment.py
                    for stngrp in stngrps:
                        case_name = f'{seed}_ds_{dataset}_a{action_count}_tp{training_period}_reward{reward}_di_{dispatcher}_{str(stngrp)[1:3]}'
                        d = path.join(root, case_name)
                        mkdir(d)
                        with io.open(path.join(d, 'config.json'), 'w') as f:
                            case = {
                                'name': case_name,
                                'params': {
                                    'seed': seed,
                                    'dataset': dataset,
                                    'action_count': action_count,
                                    'training_period': training_period,
                                    'dispatcher': dispatcher,
                                    'reward': reward,
                                    'station_group': stngrp,
                                }
                            }
                            json.dump(case, f, indent=2)
 