import io
import json
import os.path
import os
import sys
from os import mkdir, path

from dotenv import load_dotenv
load_dotenv()

toolgroup = os.getenv("TOOLGROUP")
training_period_env = int(os.getenv("TRAINING_PERIOD"))
reward_env = int(os.getenv("REWARD"))
system_path = os.getenv("SYSTEM_PATH")
total_breakdown = os.getenv("TOTAL_BREAKDOWN")
partial_breakdown = os.getenv("PARTIAL_BREAKDOWN")
longer_breakdowns = os.getenv("LONGER_BREAKDOWNS")

root = 'experiments'

#root = os.path.join(system_path, 'experiments')

if not os.path.exists(root):
    mkdir(root)

stngrps = [
    toolgroup,       # Set the toolgroup the agent is responsible for
]

for seed in [0]:
    for dataset, dispatcher in [('HVLM', 'fifo')]:
        for action_count in [9]:
            for training_period in [training_period_env]:   # set the training period
                for reward in [reward_env]:          # set the reward fuction to use see gym/environment.py
                    for stngrp in stngrps:
                        case_name = f'{str(stngrp)[1:-1]}_{seed}_ds_{dataset}_a{action_count}_tp{training_period}_reward{reward}_{dispatcher}_tb{total_breakdown}_pb{partial_breakdown}_lb{longer_breakdowns}'
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
                                    #Hinzufügen der Breakdown_Events
                                    'total_breakdowns': total_breakdown,
                                    'partial_breakdowns': partial_breakdown,
                                    'longer_breakdowns': longer_breakdowns,
                                }
                            }
                            json.dump(case, f, indent=2)
                            print('add this to the .env file:')
                            print('======================================================')
                            print(f'EXPERIMENT_NAME={case_name}')
                            print('======================================================')
