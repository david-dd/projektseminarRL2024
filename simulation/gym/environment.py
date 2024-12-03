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
                 reward_type, action, state_components, greedy_instance,plugins=None, num_agents=1, ):
        self.num_agents = num_agents
        
        self.agent_states = [None] * num_agents  # List to hold states for each agent
        self.agent_instances = [None] * num_agents
        self.lots_done_per_agent = [None] * num_agents
        self.agent_rewards = [0] * num_agents  # Rewards for each agent
        self.agent_done = [False] * num_agents  # Done flags for each agent
        self.agent_actions = [None] * num_agents  # Actions for each agent
        
        
        
        self.did_reset = False
        self.files = read_all('datasets/' + dataset)
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
        self.reset()

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        self.seed_val = seed
        self.reset()

    def step(self, actions):
        self.did_reset = False
        self.actual_step += 1
        
        rewards = []
        dones = []
        infos = []
        
        for agent_idx in range(self.num_agents):
            
            action = actions[agent_idx]
            
            # apply_priority_rule(self._machine)
            waiting_lots = self.agent_instances[agent_idx]._machine.actions
            lot_index = action
            if lot_index < len(waiting_lots) and waiting_lots[lot_index] is not None:
                lot_group = waiting_lots[lot_index]
                lot = lot_group[0]
                lots = lot_group[:min(len(lot_group), lot.actual_step.batch_max)]
                violated_minruns = self._machine.min_runs_left is not None and self._machine.min_runs_setup == lot.actual_step.setup_needed
                self.agent_instances[agent_idx].dispatch(self.agent_instances[agent_idx]._machine, lots)
                if self.max_steps == 0:
                    done = self.next_step() or self.agent_instances[agent_idx].current_time > 3600 * 24 * self.days
                else:
                    done = self.next_step() or self.max_steps < self.actual_step
                reward = 0
                if self.reward_type in [1, 2]:      # determines what reward structure is used 
                    for i in range(self.lots_done_per_agent[agent_idx], len(self.agent_instances[agent_idx].done_lots)):
                        lot = self.agent_instances[agent_idx].done_lots[i]
                        reward += 1000              # reward for all done lots
                        if self.reward_type == 2:
                            reward += 1000 if lot.deadline_at >= lot.done_at else 0     # reward for lots done within deadline
                        else:
                            reward += 1000 if lot.deadline_at >= lot.done_at else -min(500, (
                                    lot.done_at - lot.deadline_at) / 3600)              # penalty for lots done after deadline 
                
                else:
                    pass
                if violated_minruns:
                    reward += -10
                self.lots_done_per_agent[agent_idx] = len(self.agent_instances[agent_idx].done_lots)
                # Accumulate results
                rewards.append(reward)
                dones.append(done)
                infos.append({})  # Additional information (if any)
                
                for plugin in self.plugins:
                    plugin.on_step_reward(reward)
                return self.agent_states, rewards, dones, infos
            else:
                for plugin in self.plugins:
                    plugin.on_step_reward(-100)
                return self.state, -100, self.max_steps < self.actual_step, {}

    def reset(self):
        if not self.did_reset:
            self.did_reset = True
            self.actual_step = 0

            for agent_idx in range(self.num_agents):
                if self.greedy_instance is not None:
                    # Create a copy of the greedy_instance for each agent
                    agent_instance = copy.deepcopy(self.greedy_instance)
                    self.lots_done_per_agent.append(len(agent_instance.done_lots))
                else:
                    # Create a fresh instance for each agent
                    self.lots_done_per_agent.append(0)
                    run_to = 3600 * 24 * self.days
                    agent_instance = FileInstance(self.files, run_to, True, self.plugins)
                
                # Store the instance for the agent
                self.agent_instances.append(agent_instance)

                # Initialize each agent's state
                Randomizer().random.seed(self.seed_val + agent_idx)
                self.agent_states.append(self.state)  # Assuming `state` gives the agent's current state

            self.seed_val += self.num_agents
            self.step_buffer()
            self.next_step()  # Assuming this prepares the environment for the first step
    
        return self.agent_states  # Return the state of all agents


    
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
        all_agents_done = True  # Assume all agents are done, and we check if any agent is still active
        
        # Iterate through each agent
        for agent_idx in range(self.num_agents):
            agent = self.agent_instances[agent_idx]
            
            found = False
            while not found:
                done = agent.next_decision_point()
                
                if done or agent.current_time > 3600 * 24 * self.days:  # Check for termination conditions
                    self.agent_done[agent_idx] = True  # Mark this agent as done
                    break  # No need to process further for this agent
                else:
                    self.agent_done[agent_idx] = False  # Ensure agent is not marked as done

                for machine in agent.usable_machines:
                    if self.station_group is None or \
                            f'[{machine.group}]' in self.station_group or \
                            f'<{machine.family}>' in self.station_group:
                        found = True
                        break  # Proceed if the conditions are met
                    
                    # If machine is valid, attempt to dispatch lots
                    machine, lots = get_lots_to_dispatch_by_machine(agent, machine=machine, ptuple_fcn=self.dispatcher)
                    if lots is None:
                        agent.usable_machines.remove(machine)  # Remove machine if no lots available
                    else:
                        agent.dispatch(machine, lots)  # Dispatch the lots to the machine
            
            # Now after this loop, check if this agent is still not done:
            if not self.agent_done[agent_idx]:
                all_agents_done = False  # If any agent is still working, mark as not done

            # Generate actions per machine
            actions = defaultdict(lambda: [])
            for machine in agent._machine.waiting_lots:
                actions[machine.actual_step.step_name].append(machine)
            
            # Compute a moving average of actions (if needed)
            self.mavg = self.mavg * 0.99 + len(actions) * 0.01
            
            # Adjust number of actions
            if len(actions) > self.num_actions:
                agent._machine.actions = r.random.sample(list(actions.values()), self.num_actions)
            else:
                agent._machine.actions = list(actions.values())
                while len(agent._machine.actions) < self.num_actions:
                    agent._machine.actions.append(None)
                r.random.shuffle(agent._machine.actions)

        self._state = None
        return all_agents_done  # Return whether all agents are done with their steps

    @property
    def state(self):
        # Assuming that the state will be computed for the current agent (each agent has its own state)
        all_states = []

        # Loop over each agent and calculate their state
        for agent_idx in range(self.num_agents):
            agent = self.agent_instances[agent_idx]
            
            if agent is None:
                print(f"Agent at index {agent_idx} is None. Skipping.")
                continue  # Skip this iteration and move to the next agent.
            
            m: Machine = agent._machine  # Use agent-specific machine
            t = agent.current_time  # Get the current time of the specific agent
            
            if self._state is None:  # Check if state is already calculated, if not, calculate it
                agent_state = [
                    m.pms[0].timestamp - t if len(m.pms) > 0 else 999999,  # Next maintenance time
                    m.utilized_time / m.setuped_time if m.setuped_time > 0 else 0,  # Ratio of setup time / processing time
                    (m.setuped_time + m.utilized_time) / t if t > 0 else 0,  # Ratio of non-idle time
                    m.machine_class,  # Type of machine
                ]

                from statistics import mean, median
                for action in self._machine.actions:
                    if action is None:
                        agent_state += [-1000] * len(self.state_components)  # Add placeholder values for None actions
                    else:
                        action: List[Lot]
                        free_since = [agent.current_time - l.free_since for l in action]  # Agent-specific time
                        work_rem = [len(l.remaining_steps) for l in action]
                        cr = [l.cr(agent.current_time) for l in action]
                        priority = [l.priority for l in action]
                        l0 = action[0]

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
                            E.A.L4M.S.MACHINE.IDLE_RATIO: lambda: 1 - (self._machine.utilized_time / agent.current_time) if self._machine.utilized_time > 0 else 1,
                            E.A.L4M.S.MACHINE.SETUP_PROCESSING_RATIO: lambda: (self._machine.setuped_time / self._machine.utilized_time) if self._machine.utilized_time > 0 else 1,
                            E.A.L4M.S.MACHINE.MACHINE_CLASS: lambda: 0,
                        }

                        agent_state += [
                            action_type_state_lambdas[s]()
                            for s in self.state_components
                        ]
                
                # Store the computed state for this agent
                all_states.append(agent_state)

        return all_states  # Return the states for all agents



    def render(self, mode="human"):
        pass
