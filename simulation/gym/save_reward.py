import os
import numpy as np
import pickle

def save_reward_to_file(reward):
    # Get environment variables
    system_path = os.getenv("SYSTEM_PATH")
    experiment_name = os.getenv("EXPERIMENT_NAME")
    
    # Construct the experiment path
    experiment_path = os.path.join(system_path, 'experiments', experiment_name, 'config.txt')
    
    # Load existing data if the file exists
    if os.path.exists(experiment_path):
        with open(experiment_path, 'w') as file:
            pass

    # Calculate running average of scores
    #running_avg = np.zeros(len(rewards))
    #for i in range(len(running_avg)):
    #    running_avg[i] = np.mean(rewards[max(0, i-50):(i+1)])
    
    # Create new entries
    #new_entries = [{
    #    "rewards": reward,
    #    "running_avg": avg
    #} for reward, avg in zip(rewards, running_avg)]
    
    new_entries = ("rewards: " + str(reward))
    
    # Save the updated data to the file in .pkl format
    with open(experiment_path, 'a') as f:
        f.write(new_entries)

# Example usage
# rewards = [10, 20, 30, 40, 50]
# save_reward_to_file(rewards)