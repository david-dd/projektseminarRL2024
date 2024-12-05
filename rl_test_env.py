import datetime
import io
import json
import os
from sys import argv, stdout
import sys
import argparse

from stable_baselines3 import PPO

from dotenv import load_dotenv

from simulation.gym.environment import DynamicSCFabSimulationEnvironment
from simulation.gym.sample_envs import DEMO_ENV_1
from simulation.stats import print_statistics


def get_station_group(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        station_group = data['params']['station_group']
        reward_config = data['params']['reward']
        return station_group, reward_config

def main():
    experiment_name = os.getenv("EXPERIMENT_NAME_2")
    experiment_subfolder = os.getenv("EXPERIMENT_SUBFOLDER")
    system_path = os.getenv("SYSTEM_PATH")
    
    experiment_path = os.path.join(system_path, 'experiments', experiment_name)
    file_path = os.path.join(experiment_path, 'config.json')
    station_group, reward_config = get_station_group(file_path) 
    
    run( experiment_name, experiment_subfolder, reward_config)
    


def run( experiment_name, experiment_subfolder, reward_config):

    load_dotenv()

    system_path = os.getenv("SYSTEM_PATH")
    testing_days = int(os.getenv("TESTING_DAYS"))
    
    sys.path.append(os.path.join(system_path, 'simulation'))
    
    experiment_path = os.path.join(system_path, 'experiments', experiment_name)
    experiment_subfolder_path = os.path.join(experiment_path, experiment_subfolder)
    
    wandb = True 
    t = datetime.datetime.now()
    ranag =  "trained.weights"
    
    model = PPO.load(os.path.join(experiment_subfolder_path, ranag))
    with io.open(os.path.join(experiment_path, "config.json"), "r") as f:
        config = json.load(f)['params']
    
    args = dict(seed=0, num_actions=config['action_count'], active_station_group=config['station_group'], days=testing_days,
                dataset='SMT2020_' + config['dataset'], dispatcher=config['dispatcher'], reward_type=config['reward'])
    
    
    plugins = []
    if wandb:
        from simulation.plugins.wandb_plugin_env import WandBPlugin
        run_name = ("r" + str(reward_config) + experiment_subfolder)
        project_name= "projektseminarRL2024"
        plugins.append(WandBPlugin(project_name=project_name, run_name=run_name))
    env = DynamicSCFabSimulationEnvironment(**DEMO_ENV_1, **args, max_steps=1000000000, plugins=plugins, greedy_instance=None)
    obs = env.reset()
    reward = 0

    checkpoints = [ 365, testing_days]
    current_checkpoint = 0

    steps = 0
    shown_days = 0
    deterministic = True
    print("Starting Loop")
    while True:
        
        action, _states = model.predict(obs, deterministic=deterministic)
        #print("action", action)
        # if ranag:
        #     if ranag == 'random':
        #         action = env.action_space.sample()
        #     else:
        #         state = obs[4:]
        #         #print("State", state)
        #         actions = config['action_count']
        #         one_length = len(state) // actions
        #         #print("one_length", one_length)
        #         descending = True
        #         index = 0
        #         sortable = []
        #         for i in range(actions):
        #             sortable.append((state[one_length * i + index], i))
        #         sortable.sort(reverse=descending)
        #         #print("sortable", sortable)
        #         action = sortable[0][1]
        # #         print("action",action)
        obs, r, done, info = env.step(action)  
        if r < 0:
            deterministic = False
        else:
            deterministic = True
        reward += r
        steps += 1
        di = int(env.instance.current_time_days)

        if di % 10 == 0 and di > shown_days:
            print(f'Step {steps} day {shown_days}')
            shown_days = di
            stdout.flush()

        chp = checkpoints[current_checkpoint]
        if env.instance.current_time_days > chp:
            print(f'{checkpoints[current_checkpoint]} days')
            print_statistics(env.instance, chp, config['dataset'], config['dispatcher'], method=f'rl{chp}', dir=experiment_subfolder_path)
            print('=================')
            stdout.flush()
            current_checkpoint += 1
            if len(checkpoints) == current_checkpoint:
                break

        if done:
            print('Exiting with DONE')
            break

    print(f'Reward is {reward}')
    dt = datetime.datetime.now() - t
    print('Elapsed', str(dt))
    env.close()


if __name__ == '__main__':
    main()
