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


from dotenv import load_dotenv
load_dotenv()

system_path = os.getenv("SYSTEM_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
user_number = int(os.getenv("USER_NUMBER"))
experiment_number = int(os.getenv("EXPERIMENT_NUMBER"))
greedy_days_env = int(os.getenv("GREEDY_DAYS"))
training_steps = int(os.getenv("TRAINING_STEPS"))

experiment_path = os.path.join(system_path, 'experiments', experiment_name)

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

rewards_list = []

def get_station_group(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['params']['station_group']

def set_file_name():    
    file_path = os.path.join(experiment_path, 'config.json')
    station_group = get_station_group(file_path)
    
    experiment_subfolder = str(station_group) + '_' + str(user_number) + '_' + str(experiment_number) + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '_' + str(random.randint(10000,99999))
    
    path = os.path.join(experiment_path, experiment_subfolder)
    if not os.path.exists(path):
        os.mkdir(path)
        
    print(f"Experiment-Ordner erstellt: '{path}'")
    
    return experiment_subfolder

def calculate_moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Am Ende des Trainings Rewards-Liste speichern
def save_rewards_list():
    file_destination = os.path.join(experiment_path, experiment_subfolder, 'rewards_list.pkl')
    with open(file_destination, 'wb') as f:
        pickle.dump(rewards_list, f)
    print(f"Rewards-Liste gespeichert als '{file_destination}'")


experiment_subfolder = set_file_name()

# function to train the model

def main():
    to_train = training_steps  #10000000 # 608000 für 730 Tage --> 32 Jahre Trainingszeit (mit Initialisierungsphase)
    #greedy_days = 365
    greedy_days = greedy_days_env
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
                img_path = os.path.join(experiment_path, experiment_subfolder, f'rewards_plot_step.png')
                plt.figure(figsize=(10, 5))
                moving_avg = calculate_moving_average(rewards_list, window_size=500)
                plt.plot(moving_avg, label='Moving Average (window=500)')
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
                ratio = self.num_timesteps / to_train
                perc = round(ratio * 100)
                remaining = (time.time() - t) / ratio * (1 - ratio) if ratio > 0 else 9999999999999
                remaining /= 3600

                sys.stderr.write(f'\r{self.num_timesteps} / {to_train} {perc}% {round(remaining, 2)} hours left    {env.instance.current_time_days}      ')
            return super().on_step()

    
    if len(argv) > 1:
        fn = argv[1]
    else:
        #fn = experiment_path
        fn = os.path.join(experiment_path, 'config.json')
        #fn = "experiments/0_ds_HVLM_a9_tp730_reward2_di_fifo_TF\config.json"
    with open(fn, 'r') as config:
        p = json.load(config)['params']
    args = dict(num_actions=p['action_count'], active_station_group=p['station_group'],
                days=p['training_period'], dataset='SMT2020_' + p['dataset'],
                dispatcher=p['dispatcher'])
    args_eval= dict(num_actions=p['action_count'], active_station_group=p['station_group'], dataset='SMT2020_' + p['dataset'],
                dispatcher=p['dispatcher'])
    print(f'Greedy ENV bis {greedy_days} Tage erstellt')
    greedy_instance =run_greedy('SMT2020_' + p['dataset'],p['training_period'] , greedy_days, p['dispatcher'], 0, False, False, alg='l4m')
    print("Greedy Instance abgeschlossen")
    print("Args angenommen")
    env = DynamicSCFabSimulationEnvironment(**DEMO_ENV_1, **args, seed=p['seed'], max_steps=10000000, reward_type=p['reward'],greedy_instance=greedy_instance, plugins=[] )
    print("Env erstellt")
    eval_env = DynamicSCFabSimulationEnvironment(**DEMO_ENV_1, **args_eval, days= 265, seed=777, max_steps=0, reward_type=p['reward'], greedy_instance=greedy_instance,plugins=[])
    print("Alles erstellt - ich lerne jz")
    model = PPO("MlpPolicy", env, verbose=1)
    
    
    #Callbacks
    p = experiment_path
    checkpoint_callback_MyCallBack = MyCallBack(save_freq=100000, save_path=p, name_prefix='checkpoint_')
    checkpoint_callback_eval = EvalCallback(eval_env, best_model_save_path=p+'/' + experiment_subfolder +'/eval/best_model/',log_path=p+'/' + experiment_subfolder +'/eval/', eval_freq=2000000, deterministic=True, render=False )
    callback= [checkpoint_callback_MyCallBack,checkpoint_callback_eval] 
    # model.learn(
    #    total_timesteps=to_train, eval_freq=4000000, eval_env=eval_env, n_eval_episodes=1,
    #    callback=checkpoint_callback
    # )
    model.learn(
        total_timesteps=to_train,
        callback=callback,
    )
    print("Ich sichere")
    model.save(os.path.join(p, experiment_subfolder, 'trained.weights'))

    
    
    # Rewards-Liste speichern
    save_rewards_list()


if __name__ == '__main__':
    main()
