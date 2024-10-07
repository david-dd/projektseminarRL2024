import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from typing import Dict, Iterable, List, Optional, Tuple, Union
from gymnasium import spaces

class PPOMemory:
    def __init__(self, batch_size):
        self.states = [] 
        self.probs = []             #log-probs 
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='PPO_algorithmus_dr_phil/tmp/ppo/'):
        super(ActorNetwork, self).__init__()
        print("DIMS - A=", input_dims, alpha, fc1_dims, fc2_dims, n_actions)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims), #Linear Layer
                nn.ReLU(),  # Activation Function
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1) # Softmax Activation Function
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, availableActionSpace):
        dist = self.actor(state)                # state in -> probs out
        # if len(availableActionSpace) > 0:
        #     ava = True
        # else:
        #     ava = False
        ava = availableActionSpace > 0
        inv = ~ava
       
        invSum = T.sum(inv)
 
        if invSum != 0:
            distMasked = T.multiply(dist, availableActionSpace)
 
            try:
                cat = Categorical(distMasked)    
            except:
                epsi = T.multiply(availableActionSpace, -0.000001)
                distMaskedWithEpsi = T.add(distMasked, epsi)
 
                try:
                    cat = Categorical(distMaskedWithEpsi)
                except:
                    print("Error - Also distMaskedWithEpsi still containing nan")
 
        else:  
            try:
                cat = Categorical(dist)    
            except:  
                epsi = T.multiply(availableActionSpace, -0.000001)
                distEpsi = T.add(dist, epsi)
 
                try:
                    cat = Categorical(distEpsi)
                except:
                    print("Error - Also distEpsi still containing nan")
 
 
        return cat

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='PPO_algorithmus_dr_phil/tmp/ppo/'): #keine Actions -> Output ist nur ein single state
        super(CriticNetwork, self).__init__()

        print("DIMS - C=", input_dims, alpha, fc1_dims, fc2_dims)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, eval_mode=False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.eval_mode = eval_mode

        self.actor = ActorNetwork(n_actions, input_dims, alpha )
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    # def choose_action(self, observation):
    #     print(observation)
    #     for i in range(len(observation)):
    #         if observation[i] == -1000:
    #             observation[i] = 0
    #     print(observation)
        
    #     dist = self.actor(observation)
    #     value = self.critic(observation)
        
    #     if self.evalMode == True:          
    #         action = T.argmax(dist.probs, dim=1)
    #     else:
    #         action = dist.sample()

    #     probs = T.squeeze(dist.log_prob(action)).item()
    #     action = T.squeeze(action).item()
    #     value = T.squeeze(value).item()

    #     return action, probs, value
    
    def choose_action(self, observation, availableActionSpace, action=None):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        availableActionSpace = T.tensor([availableActionSpace], dtype=T.float).to(self.actor.device)
 
        dist = self.actor(state, availableActionSpace)
 
        value = self.critic(state)
 
        if action == None:
            # Nur dann Ausführen, wenn keine übermittelt wurde    
            if self.eval_mode == True:          
                myMax = T.argmax(dist.probs, dim=1)
               
                action = myMax
            else:
                action = dist.sample()  
        else:
            #print("Der Agent verwendet die vorgegeben Aktion")
            action = T.tensor([action]).to(self.actor.device)
 
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
 
        return action, probs, value

    def learn(self, available_actions):
        availableActionSpace = T.tensor([available_actions], dtype=T.float).to(self.actor.device)
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1): #für jeden Zeitschritt
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k]) 
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states, availableActionSpace)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()               

