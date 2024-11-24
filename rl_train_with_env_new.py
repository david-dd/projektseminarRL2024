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



greedy_days_env = int(os.getenv("GREEDY_DAYS"))
system_path = os.getenv("SYSTEM_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
experiment_path = os.path.join(system_path, 'experiments', experiment_name, 'config.json')

import numpy as np
import matplotlib.pyplot as plt
import pickle

rewards_list = []

def calculate_moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



# Am Ende des Trainings Rewards-Liste speichern
def save_rewards_list():
    file_destination = os.path.join(system_path, 'experiments', experiment_name, 'rewards_list.pkl')
    with open(file_destination, 'wb') as f:
        pickle.dump(rewards_list, f)
    print(f"Rewards-Liste gespeichert als '{file_destination}'")


# function to train the model

def main():
    to_train = 608000  #10000000 # 608000 für 730 Tage --> 32 Jahre Trainingszeit (mit Initialisierungsphase)
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
                plt.figure(figsize=(10, 5))
                moving_avg = calculate_moving_average(rewards_list, window_size=500)
                plt.plot(moving_avg, label='Moving Average (window=500)')
                plt.title('Reward Progress')
                plt.xlabel('Steps')
                plt.ylabel('Reward')
                plt.legend()
                plt.grid()
                plt.savefig(f'rewards_plot_step_{self.num_timesteps}.png')  # Optional: Plot speichern
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
        fn = experiment_path
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
    p = os.path.dirname(os.path.realpath(fn))
    checkpoint_callback_MyCallBack = MyCallBack(save_freq=100000, save_path=p, name_prefix='checkpoint_')
    checkpoint_callback_eval = EvalCallback(eval_env, best_model_save_path=p+'/eval/best_model/',log_path=p+'/eval/', eval_freq=2000000, deterministic=True, render=False )
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
    model.save(os.path.join(p, 'trained.weights'))

    # Rewards-Liste speichern
    save_rewards_list()


if __name__ == '__main__':
    main()
