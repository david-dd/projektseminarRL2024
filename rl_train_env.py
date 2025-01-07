import datetime
import json
import os
import sys
import time

import wandb

from stable_baselines3 import PPO

from wandb.integration.sb3 import WandbCallback

from simulation.gym.environment import DynamicSCFabSimulationEnvironment
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from simulation.greedy_RL import run_greedy
from sys import argv

from simulation.gym.sample_envs import DEMO_ENV_1

from rl_test_env import run


from dotenv import load_dotenv
load_dotenv()

system_path = os.getenv("SYSTEM_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
user_number = int(os.getenv("USER_NUMBER"))
experiment_number = int(os.getenv("EXPERIMENT_NUMBER"))
greedy_days = int(os.getenv("GREEDY_DAYS"))
training_steps = int(os.getenv("TRAINING_STEPS"))
evaluate_after_train = os.getenv("EVALUATE_AFTER_TRAIN")


experiment_path = os.path.join(system_path, 'experiments', experiment_name)

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

rewards_list = []

def get_station_group(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        station_group = data['params']['station_group']
        reward_config = data['params']['reward']
        return station_group, reward_config

def set_file_name():    
    file_path = os.path.join(experiment_path, 'config.json')
    station_group, reward_config = get_station_group(file_path)
    
    experiment_subfolder = str(station_group)[1:-1] + '_' + str(user_number) + '_' + str(experiment_number) + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '_' + str(random.randint(10000,99999))
    
    experiment_subfolder_path = os.path.join(experiment_path, experiment_subfolder)
    if not os.path.exists(experiment_subfolder_path):
        os.mkdir(experiment_subfolder_path)
        
    print(f"\n==========================\nExperiment-Ordner erstellt:\n{experiment_subfolder_path}\n==========================\n")
    
    return experiment_subfolder, experiment_subfolder_path, reward_config

def calculate_moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Am Ende des Trainings Rewards-Liste speichern
def save_rewards_list():
    file_destination = os.path.join(experiment_subfolder_path, 'rewards_list.pkl')
    with open(file_destination, 'wb') as f:
        pickle.dump(rewards_list, f)
    print(f"Rewards-Liste gespeichert als '{file_destination}'")


experiment_subfolder, experiment_subfolder_path, reward_config = set_file_name()

# function to train the model

def main():
    
    t = time.time()
    save_freq = 500 
    class MyCallBack(CheckpointCallback):

        def on_step(self) -> bool:
            # Globale Liste zum Speichern der Rewards            

            # Aktuellen Reward sammeln (falls vorhanden)
            if 'rewards' in self.locals:
                reward = self.locals['rewards']
                if isinstance(reward,(list,np.ndarray)):
                    rewards_list.append(float(reward[0]))
                else:
                    rewards_list.append(float(reward))
            
            # ======================================================
            # print reward here 
            if self.num_timesteps % 10000 == 0 and len(rewards_list) > 0:
                img_path = os.path.join(experiment_subfolder_path, f'rewards_plot_{experiment_subfolder}.png')
                plt.figure(figsize=(10, 5))
                moving_avg = calculate_moving_average(rewards_list, window_size=10000)
                plt.plot(moving_avg, label='Moving Average (window=10000)')
                plt.title('Reward Progress')
                plt.xlabel('Steps')
                plt.ylabel('Reward')
                plt.legend()
                plt.grid()
                plt.savefig(img_path)  # Optional: Plot speichern
                plt.close()

            # ======================================================
            
            #Fortschritt ausgeben
            if self.num_timesteps % 100 == 0:
                ratio = self.num_timesteps / training_steps
                perc = round(ratio * 100)
                remaining = (time.time() - t) / ratio * (1 - ratio) if ratio > 0 else 9999999999999
                remaining /= 3600

                sys.stderr.write(f'\r{self.num_timesteps} / {training_steps} {perc}% {round(remaining, 2)} hours left    {env.instance.current_time_days}      ')
            return super().on_step()

    
    if len(argv) > 1:
        fn = argv[1]
    else:
        fn = os.path.join(experiment_path, 'config.json')
    with open(fn, 'r') as config:
        p = json.load(config)['params']
    args = dict(num_actions=p['action_count'], active_station_group=p['station_group'],
                days=p['training_period'], dataset='SMT2020_' + p['dataset'],
                dispatcher=p['dispatcher'])
    args_eval= dict(num_actions=p['action_count'], active_station_group=p['station_group'], dataset='SMT2020_' + p['dataset'],
                dispatcher=p['dispatcher'])
    

    #special_events_list: List[bool]
    special_events_list =[]
    #p = json.load(config)['params']
    total_breakdowns = p['total_breakdowns'] == "True"
    partial_breakdowns = p['partial_breakdowns'] == "True"
    longer_breakdowns = p['longer_breakdowns'] == "True"

    special_events_list.append(total_breakdowns)
    special_events_list.append(partial_breakdowns)
    special_events_list.append(longer_breakdowns)
    
    print(f'Greedy ENV bis {greedy_days} Tage erstellt')
    greedy_instance =run_greedy('SMT2020_' + p['dataset'],p['training_period'] , greedy_days, p['dispatcher'], 0, False, False, alg='l4m',special_events_list=special_events_list)
    print("Greedy Instance abgeschlossen")
    print("Args angenommen")
    env = DynamicSCFabSimulationEnvironment(**DEMO_ENV_1, **args, seed=p['seed'], max_steps=10000000, reward_type=p['reward'],greedy_instance=greedy_instance, plugins=[],special_events_list=special_events_list )
    print("Env erstellt")
    eval_env = DynamicSCFabSimulationEnvironment(**DEMO_ENV_1, **args_eval, days= 265, seed=777, max_steps=0, reward_type=p['reward'], greedy_instance=greedy_instance,plugins=[],special_events_list=special_events_list)
    print("Alles erstellt - ich lerne jz")
    model = PPO("MlpPolicy", env, verbose=1)
    
    
    #Callbacks
    checkpoint_callback_MyCallBack = MyCallBack(save_freq=100000, save_path=experiment_subfolder_path + '/checkpoint', name_prefix='checkpoint_')
    checkpoint_callback_eval = EvalCallback(eval_env, best_model_save_path=experiment_subfolder_path +'/eval/best_model/',log_path=experiment_subfolder_path +'/eval/', eval_freq=2000000, deterministic=True, render=False )
    callback= [checkpoint_callback_MyCallBack,checkpoint_callback_eval] 
    # model.learn(
    #    total_timesteps=training_steps, eval_freq=4000000, eval_env=eval_env, n_eval_episodes=1,
    #    callback=checkpoint_callback
    # )
    model.learn(
        total_timesteps=training_steps,
        callback=callback,
    )
    print("Ich sichere")
    model.save(os.path.join(experiment_subfolder_path, 'trained.weights'))    
    
    # Rewards-Liste speichern
    save_rewards_list()
    
    end_time = time.time()
    elapsed_time = end_time - t
    formatted_time = str(datetime.timedelta(seconds=elapsed_time))
    print("=========================================")
    print("Experiment name: " + experiment_name )
    print("Experiment subfolder: " + experiment_subfolder)
    print("Total time taken: " + formatted_time)
    print("=========================================")
    
    if (evaluate_after_train == "True"):
        print("\nstart evaluation")
        print("=========================================")
        run(experiment_name, experiment_subfolder, reward_config)
    


if __name__ == '__main__':
    main()
