import statistics
from collections import defaultdict
from typing import List
import os
import sys
import gym
from gym import Env

from dotenv import load_dotenv
load_dotenv()

system_path = os.getenv("SYSTEM_PATH")
use_system_path = os.getenv("USE_SYSTEM_PATH")

if use_system_path == 'True':
    # uses the path in the env file
    sys.path.append(os.path.join(system_path))
    sys.path.append(os.path.join(system_path, 'simulation'))
    sys.path.append(os.path.join(system_path, 'simulation', 'gym'))
elif use_system_path == 'C':
    sys.path.append(os.path.join(os.path.sep,'home','tosc270g','smt2020_4','projektseminarRL2024','simulation'))
    sys.path.append(os.path.join(os.path.sep,'home','tosc270g','smt2020_4','projektseminarRL2024','simulation', 'gym'))   
else:
    # ZIH
    sys.path.append(os.path.join(os.path.sep,'projects','p078','p_htw_promentat','smt2020_0','simulation'))
    sys.path.append(os.path.join(os.path.sep,'projects','p078','p_htw_promentat','smt2020_0','simulation', 'gym'))



#sys.path.append(os.path.join(os.path.sep,'data','horse','ws','wiro085f-WsRodmann','RL_Version','PySCFabSim', 'simulation'))
#sys.path.append(os.path.join(os.path.sep,'data','horse','ws','wiro085f-WsRodmann','RL_Version','PySCFabSim', 'simulation', 'gym'))

#Heik lokal
#sys.path.append(os.path.join('C:/','Users','David Heik','Desktop','Arbeit2024','Studium','Studentenbetreuung','Projektseminar','25-25 - SMT2020','Projekt', 'projektseminarRL2024','simulation'))
#sys.path.append(os.path.join('C:/','Users','David Heik','Desktop','Arbeit2024','Studium','Studentenbetreuung','Projektseminar','25-25 - SMT2020','Projekt', 'projektseminarRL2024','simulation', 'gym'))
 


from classes import Machine, Lot
from file_instance import FileInstance
from greedy import get_lots_to_dispatch_by_machine
from dispatching.dispatcher import Dispatchers, dispatcher_map
from E import E
from randomizer import Randomizer
from read import read_all
from sample_envs import DEMO_ENV_1
import datetime
import copy

from save_reward import save_reward_to_file

r = Randomizer()

STATE_COMPONENTS_DEMO = (
    E.A.L4M.S.OPERATION_TYPE.NO_LOTS_PER_BATCH,
    E.A.L4M.S.OPERATION_TYPE.CR.MAX,
    E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MAX,
    E.A.L4M.S.OPERATION_TYPE.SETUP.MIN_RUNS_OK,
    E.A.L4M.S.OPERATION_TYPE.SETUP.NEEDED,
    E.A.L4M.S.OPERATION_TYPE.SETUP.LAST_SETUP_TIME,
)


class DynamicSCFabSimulationEnvironment(Env):

    def __init__(self, num_actions, active_station_group, days, dataset, dispatcher, seed, max_steps,
                 reward_type, action, state_components, greedy_instance,plugins=None, special_events_list: List[bool]= None ):
        self.did_reset = False
        self.files = read_all(os.path.join(system_path, 'datasets/' , dataset))
        self.instance = None
        self.num_actions = num_actions
        self.days = days
        self.action_space = gym.spaces.Discrete(num_actions)
        self.action = action
        self.observation_space = gym.spaces.Box(low=-100, high=1000000,
                                                shape=(4 + num_actions * len(state_components),))
        self._state = None
        self.station_group = active_station_group
        self.lots_done = 0
        self.seed_val = seed
        self.dispatcher = dispatcher_map[dispatcher]
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.mavg = 0
        self.state_components = state_components
        self.plugins = plugins
        self.stepbuffer={}  
        self.greedy_instance = copy.deepcopy(greedy_instance)
        
        self.previous_setup = ''
        
        self.special_events_list = special_events_list or False
        self.reset()
        
        
       

        

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        self.seed_val = seed
        self.reset()

    def step(self, action):
        self.did_reset = False
        self.actual_step += 1
        # apply_priority_rule(self._machine)
        waiting_lots = self._machine.actions
        lot_index = action
        if lot_index < len(waiting_lots) and waiting_lots[lot_index] is not None:
            lot_group = waiting_lots[lot_index]
            lot = lot_group[0]
            lots = lot_group[:min(len(lot_group), lot.actual_step.batch_max)]
            violated_minruns = self._machine.min_runs_left is not None and self._machine.min_runs_setup == lot.actual_step.setup_needed
            self.instance.dispatch(self._machine, lots)
            if self.max_steps == 0:
                done = self.next_step() or self.instance.current_time > 3600 * 24 * self.days
            else:
                done = self.next_step() or self.max_steps < self.actual_step
            reward = 0
            if self.reward_type in [1, 2]:      # determines what reward structure is used 
                for i in range(self.lots_done, len(self.instance.done_lots)):
                    lot = self.instance.done_lots[i]
                    reward += 1000              # reward for all done lots
                    if self.reward_type == 2:
                        reward += 1000 if lot.deadline_at >= lot.done_at else 0     # reward for lots done within deadline
                    else:
                        reward += 1000 if lot.deadline_at >= lot.done_at else -min(500, (
                                lot.done_at - lot.deadline_at) / 3600)              # penalty for lots done after deadline 
            elif self.reward_type == 3:         # determines what reward structure is used
                reward += statistics.mean(
                    [min(1, j.cr(self.instance.current_time) - 1) for j in self.instance.active_lots])  # reward if CR is below 1
            # Diplomarbeit: William Rodmann
            elif self.reward_type == 4:         # determines what reward structure is used
                 #Flow-Faktor
                part_1 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)): # auf die letzten 30 Tage beschränken 
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 60):
                            CT = lot.done_at - lot.release_at           # time the lot is in the system 
                            RPT = lot.processing_time                   # time it takes for a lot to be processed (mashine time)
                            flow_factor =  CT/RPT                       # time the lot is in the system compoared to mashine time
                            part_1 -= (flow_factor-1)*lot.priority      # adds a prioity evaluation to the flow factor
                                                                        # the less a lot waits the better 
                #Zeitlicher Puffer für den aktuellen Routenschritt
                part_2 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family not in self.station_group:
                         continue
                    else:
                        part_2_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_2_1 += 100     # reward for steps that end within the corect timeframe
                                if self.instance.current_time <= step_start:
                                    part_2_1 += 100     # reward for steps that start within the correct timeframe
                                else:
                                    part_2_1 -= min(50, (self.instance.current_time - step_end) / 3600) # step_end is alway smaler than current_time
                                                        # penalty depending on how much the correct timeframe is exceeded
                                                        # penalty for lots that arent in the expected timeframe
                                part_2_1 = part_2_1*j.priority/10
                            else:
                                continue
                            part_2 += part_2_1
                #Fertigstellung der Lose    
                part_3 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 60):
                            part_3 += 1000 if lot.deadline_at >= lot.done_at else -min(50, (
                                    lot.done_at - lot.deadline_at) / 3600)
                                                        # reward for finnishing a lot or penalty for finishing a lot after deadline 
                            part_3 = part_2 / lot.priority/10
                #Warteschlangen der Maschinen
                part_4 = 0
                for tool in self.instance.machines:  
                    if tool.group == 'Delay_32':
                        continue            # no penalty for the Delay_32 machine
                    elif tool.group == 'Diffusion' and len(tool.events) == 0 and len(tool.waiting_lots) >= 6 * 5: #TODO: Woher den Faktor?
                        part_4 -= 10        # penalty if a lot of lots are waiting in the queue of the Diffusion machine
                    elif len(tool.events) == 0 and len(tool.waiting_lots) >= 6:
                        part_4 -= 10        # penalty if a lot of lots are waiting in the queue of other machines
                reward = 10*part_1 + 0.01*part_2 + 1*part_3 + 0.1*part_4
                
                # save the reward for the current step
                # save_reward_to_file(reward)
                
            # reward function Simon 
            elif self.reward_type == 12: 
                part_1 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 60):
                            CT = lot.done_at - lot.release_at
                            RPT = lot.processing_time
                            flow_factor = CT / RPT
                            part_1 -= (flow_factor-1)*lot.priority
                # Erhöhte Gewichtung für zeitlichen Puffer
                part_2 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family not in self.station_group:
                        continue
                    else:
                        part_2_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at + self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at + self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_2_1 += 200  
                                if self.instance.current_time <= step_start:
                                    part_2_1 += 200  
                                else:
                                    delay_penalty = min(100, (self.instance.current_time - step_end) / 1800)  
                                    part_2_1 -= delay_penalty
                                part_2_1 = part_2_1 * j.priority / 10
                            else:
                                continue
                            part_2 += part_2_1

                part_3 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 60):
                            part_3 += 1000 if lot.deadline_at >= lot.done_at else -min(50, (
                                    lot.done_at - lot.deadline_at) / 3600)
                                                        # reward for finnishing a lot or penalty for finishing a lot after deadline 
                            part_3 = part_2 / lot.priority/10               
                part_4 = 0
                for tool in self.instance.machines:
                    if tool.group == 'Delay_32':
                        continue
                    elif tool.group == 'Diffusion' and len(tool.events) == 0 and len(tool.waiting_lots) >= 30:
                        part_4 -= 10
                    elif len(tool.events) == 0 and len(tool.waiting_lots) >= 6:
                        part_4 -= 10

                # Erhöhte Gewichtung des zeitlichen Puffers 
                reward = 15*part_1 + 0.1*part_2 + 1*part_3 + 0.1*part_4
            
            # Reward Funktion Max   
            elif self.reward_type == 13:
                #Flow-Faktor
                part_1 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 30):
                            CT = lot.done_at - lot.release_at
                            RPT = lot.processing_time
                            flow_factor =  CT/RPT
                            part_1 -= (flow_factor)*lot.priority
                #Buffer pro Schritt
                part_2 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family not in self.station_group:
                            continue
                    else:
                        part_2_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_2_1 += 100
                                if self.instance.current_time <= step_start:
                                    part_2_1 += 100
                                else:
                                    part_2_1 -= min(50, (self.instance.current_time - step_end) / 3600)
                                part_2_1 = part_2_1*j.priority/10
                            else:
                                continue
                            part_2 += part_2_1
                #Fertigstellung            
                part_3 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 30):
                            part_3 += 1000
                            part_3 += 1000 if lot.deadline_at >= lot.done_at else -min(750, (
                                    lot.done_at - lot.deadline_at) / 3600)
                            part_3 = part_2 / lot.priority/5
                reward = 1*part_1 + 1*part_2 + 0.5*part_3        
                        
            # Reward Funktion Claire
            elif self.reward_type == 14:         # determines what reward structure is used
                 #Flow-Faktor
                part_1 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)): # auf die letzten 30 Tage beschränken 
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 60):
                            CT = lot.done_at - lot.release_at           # time the lot is in the system 
                            RPT = lot.processing_time                   # time it takes for a lot to be processed (mashine time)
                            flow_factor =  CT/RPT                       # time the lot is in the system compoared to mashine time
                            part_1 -= (flow_factor-1)*lot.priority      # adds a prioity evaluation to the flow factor
                                                                        # the less a lot waits the better 
                #Zeitlicher Puffer für den aktuellen Routenschritt
                part_2 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family not in self.station_group:
                         continue
                    else:
                        part_2_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_2_1 += 100     # reward for steps that end within the corect timeframe
                                if self.instance.current_time <= step_start:
                                    part_2_1 += 100     # reward for steps that start within the correct timeframe
                                else:
                                    part_2_1 -= min(50, (self.instance.current_time - step_end) / 3600) # step_end is alway smaler than current_time
                                                        # penalty depending on how much the correct timeframe is exceeded
                                                        # penalty for lots that arent in the expected timeframe
                                part_2_1 = part_2_1*j.priority/10
                            else:
                                continue
                            part_2 += part_2_1
                #Fertigstellung der Lose    
                part_3 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 60):
                            part_3 += 1000 if lot.deadline_at >= lot.done_at else -min(50, (
                                    lot.done_at - lot.deadline_at) / 3600)
                                                        # reward for finnishing a lot or penalty for finishing a lot after deadline 
                            part_3 = part_2 / lot.priority/10
                #Warteschlangen der Maschinen
                part_4 = 0
                for tool in self.instance.machines:  
                    if tool.group == 'Delay_32':
                        continue            # no penalty for the Delay_32 machine
                    elif tool.group == 'Implant' and len(tool.events) == 0 and len(tool.waiting_lots) >= 6 * 5: #TODO: Woher den Faktor?
                        part_4 -= 10        # penalty if a lot of lots are waiting in the queue of the Diffusion machine
                    
                    # removed penelty for other delays 
                    
                reward = 1*part_1 + 0.01*part_2 + 10*part_3 + 0.1*part_4
                
            elif self.reward_type == 141:         # determines what reward structure is used
                lots_done = 0
                for i in range(self.lots_done, len(self.instance.done_lots)):
                    lot = self.instance.done_lots[i]
                    lots_done += 1000              # reward for all done lots                  
                    lots_done += 1000 if lot.deadline_at >= lot.done_at else -min(200, (
                            lot.done_at - lot.deadline_at) / 3600)              # penalty for lots done after deadline 
                
                #Flow-Faktor
                flow_factor_reward = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)): # auf die letzten 30 Tage beschränken 
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 60):
                            CT = lot.done_at - lot.release_at           # time the lot is in the system 
                            RPT = lot.processing_time                   # time it takes for a lot to be processed (mashine time)
                            flow_factor =  CT/RPT                       # time the lot is in the system compoared to mashine time
                            flow_factor_reward -= (flow_factor-1)*lot.priority      # adds a prioity evaluation to the flow factor
                                                                        # the less a lot waits the better 
                #Zeitlicher Puffer für den aktuellen Routenschritt
                part_2 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family not in self.station_group:
                         continue
                    else:
                        part_2_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_2_1 += 100     # reward for steps that end within the corect timeframe
                                if self.instance.current_time <= step_start:
                                    part_2_1 += 100     # reward for steps that start within the correct timeframe
                                else:
                                    part_2_1 -= min(50, (self.instance.current_time - step_end) / 3600) # step_end is alway smaler than current_time
                                                        # penalty depending on how much the correct timeframe is exceeded
                                                        # penalty for lots that arent in the expected timeframe
                                part_2_1 = part_2_1*j.priority/10
                            else:
                                continue
                            part_2 += part_2_1
                reward = 10*lots_done + 3*flow_factor_reward + 1*part_2
                
            elif self.reward_type == 142:         # determines what reward structure is used
                lots_done = 0
                for i in range(self.lots_done, len(self.instance.done_lots)):
                    lot = self.instance.done_lots[i]
                    lots_done += 1000              # reward for all done lots                  
                    lots_done += 1000 if lot.deadline_at >= lot.done_at else -min(200, (
                            lot.done_at - lot.deadline_at) / 3600)              # penalty for lots done after deadline 
                
                # removed flow factor to keep setup time in mind 
                
                setup_reward = 0
                for tool in self.instance.machines:
                    if tool.group in self.station_group:
                        if tool.current_setup != self.previous_setup:
                            self.previous_setup = tool.current_setup
                            setup_reward -= 100            
                            
                que_reward = 0
                for tool in self.instance.machines:  
                    if tool.group == 'Implant' and len(tool.events) == 0 and len(tool.waiting_lots) >= 6 * 5: #TODO: Woher den Faktor?
                        que_reward -= 10        # penalty if a lot of lots are waiting in the queue of the Implant machine
                    
                    # removed penelty for other delays 
                reward = 10*lots_done + 1*que_reward + 5*setup_reward
                
            elif self.reward_type == 5:         # determines what reward structure is used
                #Flow-Faktor
                part_1 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 30):
                            CT = lot.done_at - lot.release_at
                            RPT = lot.processing_time
                            flow_factor =  CT/RPT
                            part_1 -= (flow_factor-1)*lot.priority
                #Buffer pro Schritt
                part_2 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family not in self.station_group:
                         continue
                    else:
                        part_2_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_2_1 += 100
                                if self.instance.current_time <= step_start:
                                    part_2_1 += 100
                                else:
                                    part_2_1 -= min(50, (self.instance.current_time - step_end) / 3600)
                                part_2_1 = part_2_1*j.priority/10
                            else:
                                continue
                            part_2 += part_2_1
                #Fertigstellung            
                part_3 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 30):
                            part_3 += 1000
                            part_3 += 1000 if lot.deadline_at >= lot.done_at else -min(500, (
                                    lot.done_at - lot.deadline_at) / 3600)
                            part_3 = part_2 / lot.priority/10
                reward = 1*part_1 + 0.1*part_2 + 1*part_3
            elif self.reward_type == 6:         # determines what reward structure is used
                #Fertigstellung            
                part_1 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        if lot.done_at >= self.instance.current_time - (3600 * 24 * 30):
                            part_1 += 1000
                            part_1 += 1000 if lot.deadline_at >= lot.done_at else -min(500, (
                                    lot.done_at - lot.deadline_at) / 3600)
                            part_1 = part_2 / lot.priority/10
                #Buffer pro Schritt
                part_2 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family not in self.station_group:
                         continue
                    else:
                        part_2_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_2_1 += 100
                                if self.instance.current_time <= step_start:
                                    part_2_1 += 100
                                else:
                                    part_2_1 -= min(50, (self.instance.current_time - step_end) / 3600)
                                part_2_1 = part_2_1*j.priority/10
                            else:
                                continue
                            part_2 += part_2_1
                reward = 1*part_1 + 0.1*part_2
            elif self.reward_type == 10:        # determines what reward structure is used
                #Buffer pro Schritt
                part_1 = 0
                for j in self.instance.active_lots:
                    if j.actual_step.family != self.station_group:
                         continue
                    else:
                        part_1_1 = 0
                        for route in self.stepbuffer:
                            nummer_aus_part = ''.join(filter(str.isdigit, j.part_name))
                            if nummer_aus_part in route:
                                step_start = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][0]
                                step_end = j.release_at+self.stepbuffer[route][j.name][j.actual_step.order][1]
                                if self.instance.current_time <= step_end:
                                    part_1_1 += 10
                                elif self.instance.current_time <= step_start:
                                    part_1_1 += 10
                                else:
                                    part_1_1 -= min(50, (self.instance.current_time - step_end) / 3600)
                                part_1_1 = part_1_1*j.priority/10
                            else:
                                continue
                            part_1 += part_1_1
                #Fertigstellung
                part_2 = 0
                if self.instance.current_time_days >= 100:
                    for i in range(self.lots_done, len(self.instance.done_lots)):
                        lot = self.instance.done_lots[i]
                        reward += 1000
                        if lot.done_at <= self.instance.current_time - (3600 * 24 * 60):
                            part_2 += 100 if lot.deadline_at >= lot.done_at else -min(500, (
                                    lot.done_at - lot.deadline_at) / 3600)
                            part_2 = part_2 / lot.priority/10
                #Warteschlange der Maschinen
                part_3 = 0
                for tool in self.instance.machines:  
                    if tool.group == 'Delay_32':
                        continue
                    elif tool.group == 'Diffusion' and len(tool.events) == 0 and len(tool.waiting_lots) >= 6 * 5: #TODO: Woher den Faktor?
                        part_3 -= 10
                    elif len(tool.events) == 0 and len(tool.waiting_lots) >= 6:
                        part_3 -= 10
                reward = 1*part_1 + 1*part_2 + 1*part_3
            # elif self.reward_type == 7:
            #     reward += statistics.mean(                                                                    #l.notlateness existiert nicht
            #         [l.notlateness(self.instance.current_time) for l in self.instance.active_lots])
            else:
                pass
            if violated_minruns:
                reward += -10
            self.lots_done = len(self.instance.done_lots)
            for plugin in self.plugins:
                plugin.on_step_reward(reward)
            return self.state, reward, done, {}
        else:
            for plugin in self.plugins:
                plugin.on_step_reward(-100)
            return self.state, -100, self.max_steps < self.actual_step, {}

    def reset(self):
        if not self.did_reset:
            self.did_reset = True
            self.actual_step = 0
            if self.greedy_instance is not None:
                self.instance = copy.deepcopy(self.greedy_instance)
                self.lots_done = len(self.instance.done_lots)
            else:
                self.lots_done = 0
                run_to = 3600 * 24 * self.days
                self.instance = FileInstance(self.files, run_to, True, self.plugins, self.special_events_list)
            Randomizer().random.seed(self.seed_val)
            self.seed_val += 1
            self.step_buffer()
            self.next_step()
        return self.state
    
    def date_time_parse(st):
        return datetime.datetime.strptime(st, '%m/%d/%y %H:%M:%S')
    
    def process_steps_per_route(self):
        process_steps = {}
        route_keys = [key for key in self.files.keys() if 'route' in key]
        for rk in route_keys:
            count= 0
            for step in self.files[rk]:
                if step['STNFAM'] != 'Delay_32':
                    count += 1
            process_steps[rk]=count
        return process_steps
    def step_buffer(self):
        process_steps_per_route = self.process_steps_per_route()
        route_keys = [key for key in self.files.keys() if 'route' in key]
        order_keys = [key for key in self.files.keys() if 'order' in key]
        parts = {p['PART']: p['ROUTEFILE'] for p in self.files['part.txt']}
        transport_time = self.files['fromto.txt'][0]['DTIME']
        for value in parts:
            for rk in route_keys:
                step_number = process_steps_per_route[rk]
                for order in self.files[order_keys[0]]:
                    if order['PART'] != value or rk != parts[value]:
                        continue
                    diff = (datetime.datetime.strptime(order['DUE'], '%m/%d/%y %H:%M:%S') - datetime.datetime.strptime(order['START'], '%m/%d/%y %H:%M:%S'))
                    relative_deadline = diff.total_seconds()
                    Load_and_Unload_time = 2*60 # 2min in sec
                    step_buffer = relative_deadline/step_number - Load_and_Unload_time - transport_time
                    step_start = 0
                    step_end = 0
                    step_counter = 0
                    for i in range(len(self.files[rk])):
                        if self.files[rk][i]['PTPER']== 'per_piece':
                            processing_time = self.files[rk][i]['PTIME']*60*25
                        elif self.files[rk][i]['PTPER'] == 'per_batch':
                            processing_time = self.files[rk][i]['PTIME']*(self.files[rk][i]['BATCHMX'] / 25)*60
                        else:
                            processing_time = self.files[rk][i]['PTIME']*60 
                        if self.files[rk][i]['STNFAM'] == 'Delay_32':
                            step_end += 0
                        else:
                            step_end += step_buffer
                            step_counter += 1
                        if self.files[rk][i]['ROUTE'] not in self.stepbuffer:
                            self.stepbuffer[self.files[rk][i]['ROUTE']] = {}
                        if order['LOT'] not in self.stepbuffer[self.files[rk][i]['ROUTE']]:
                            self.stepbuffer[self.files[rk][i]['ROUTE']][order['LOT']] = {}
                        if self.files[rk][i]['STEP'] not in self.stepbuffer[self.files[rk][i]['ROUTE']][order['LOT']]:
                            self.stepbuffer[self.files[rk][i]['ROUTE']][order['LOT']][self.files[rk][i]['STEP']]=[step_start, step_end]
                        else: 
                            continue
                        step_start = step_end + processing_time

    def next_step(self):
        found = False
        while not found:
            done = self.instance.next_decision_point()
            if done or self.instance.current_time > 3600 * 24 * self.days:
                return True
            for machine in self.instance.usable_machines:
                break
            if self.station_group is None or \
                    f'[{machine.group}]' in self.station_group or \
                    f'<{machine.family}>' in self.station_group:
                found = True
            else:
                machine, lots = get_lots_to_dispatch_by_machine(self.instance, machine=machine,
                                                                ptuple_fcn=self.dispatcher)
                if lots is None:
                    self.instance.usable_machines.remove(machine)
                else:
                    self.instance.dispatch(machine, lots)
        self._machine = machine
        actions = defaultdict(lambda: [])
        for lot in machine.waiting_lots:
            actions[lot.actual_step.step_name].append(lot)
        self.mavg = self.mavg * 0.99 + len(actions) * 0.01
        if len(actions) > self.num_actions:
            self._machine.actions = r.random.sample(list(actions.values()), self.num_actions)
        else:
            self._machine.actions = list(actions.values())
            while len(self._machine.actions) < self.num_actions:
                self._machine.actions.append(None)
            r.random.shuffle(self._machine.actions)
        self._state = None
        return False

    @property
    def state(self):
        if self._state is None:
            m: Machine = self._machine
            t = self.instance.current_time
            self._state = [
                m.pms[0].timestamp - t if len(m.pms) > 0 else 999999,  # next maintenance
                m.utilized_time / m.setuped_time if m.setuped_time > 0 else 0,  # ratio of setup time / processing time
                (m.setuped_time + m.utilized_time) / t if t > 0 else 0,  # ratio of non idle time
                m.machine_class,  # type of machine
            ]
            from statistics import mean, median
            for action in self._machine.actions:
                if action is None:
                    self._state += [-1000] * len(self.state_components)
                else:
                    action: List[Lot]
                    free_since = [self.instance.current_time - l.free_since for l in action]
                    work_rem = [len(l.remaining_steps) for l in action]
                    cr = [l.cr(self.instance.current_time) for l in action]
                    priority = [l.priority for l in action]
                    l0 = action[0]

                    self._machine: Machine
                    action_type_state_lambdas = {
                        E.A.L4M.S.OPERATION_TYPE.NO_LOTS: lambda: len(action),
                        E.A.L4M.S.OPERATION_TYPE.NO_LOTS_PER_BATCH: lambda: len(action) / l0.actual_step.batch_max,
                        E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MEAN: lambda: mean(work_rem),
                        E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MEDIAN: lambda: median(work_rem),
                        E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MAX: lambda: max(work_rem),
                        E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MIN: lambda: min(work_rem),
                        E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MEAN: lambda: mean(free_since),
                        E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MEDIAN: lambda: median(free_since),
                        E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MAX: lambda: max(free_since),
                        E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MIN: lambda: min(free_since),
                        E.A.L4M.S.OPERATION_TYPE.PROCESSING_TIME.AVERAGE: lambda: l0.actual_step.processing_time.avg(),
                        E.A.L4M.S.OPERATION_TYPE.BATCH.MIN: lambda: l0.actual_step.batch_min,
                        E.A.L4M.S.OPERATION_TYPE.BATCH.MAX: lambda: l0.actual_step.batch_max,
                        E.A.L4M.S.OPERATION_TYPE.BATCH.FULLNESS: lambda: min(1, len(action) / l0.actual_step.batch_max),
                        E.A.L4M.S.OPERATION_TYPE.PRIORITY.MEAN: lambda: mean(priority),
                        E.A.L4M.S.OPERATION_TYPE.PRIORITY.MEDIAN: lambda: median(priority),
                        E.A.L4M.S.OPERATION_TYPE.PRIORITY.MAX: lambda: max(priority),
                        E.A.L4M.S.OPERATION_TYPE.PRIORITY.MIN: lambda: min(priority),
                        E.A.L4M.S.OPERATION_TYPE.CR.MEAN: lambda: mean(cr),
                        E.A.L4M.S.OPERATION_TYPE.CR.MEDIAN: lambda: median(cr),
                        E.A.L4M.S.OPERATION_TYPE.CR.MAX: lambda: max(cr),
                        E.A.L4M.S.OPERATION_TYPE.CR.MIN: lambda: min(cr),
                        E.A.L4M.S.OPERATION_TYPE.SETUP.NEEDED: lambda: 0 if l0.actual_step.setup_needed == '' or l0.actual_step.setup_needed == m.current_setup else 1,
                        E.A.L4M.S.OPERATION_TYPE.SETUP.MIN_RUNS_LEFT: lambda: 0 if self._machine.min_runs_left is None else self._machine.min_runs_left,
                        E.A.L4M.S.OPERATION_TYPE.SETUP.MIN_RUNS_OK: lambda: 1 if l0.actual_step.setup_needed == '' or l0.actual_step.setup_needed == self._machine.min_runs_setup else 0,
                        E.A.L4M.S.OPERATION_TYPE.SETUP.LAST_SETUP_TIME: lambda: self._machine.last_setup_time,
                        E.A.L4M.S.MACHINE.MAINTENANCE.NEXT: lambda: 0,
                        E.A.L4M.S.MACHINE.IDLE_RATIO: lambda: 1 - (
                                    self._machine.utilized_time / self.instance.current_time) if self._machine.utilized_time > 0 else 1,
                        E.A.L4M.S.MACHINE.SETUP_PROCESSING_RATIO: lambda: (
                                    self._machine.setuped_time / self._machine.utilized_time) if self._machine.utilized_time > 0 else 1,
                        E.A.L4M.S.MACHINE.MACHINE_CLASS: lambda: 0,
                    }
                    self._state += [
                        action_type_state_lambdas[s]()
                        for s in self.state_components
                    ]
        return self._state

    def render(self, mode="human"):
        pass
