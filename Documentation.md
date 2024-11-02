# important files 

run the simulation [main.py](main.py)
create a new experiment [exp_set_gen.py](exp_set_gen.py) creates a new config file to run the training procces
train a new model [rl_train.py](rl_train.py)
test a model [rl_test.py](rl_test.py)
reward function [environment.py](./simulation/gym/environment.py)


# train a new model 

## create a new experiment 

1. configure the env file
2. create the config.json file 

```shell
python3.10 exp_set_gen_with_env.py
```

## train the moddel 

```shell
python3.10 rl_train_with_env.py
```

## test the moddel

```shell
python3.10 rl_test_with_env.py --days = 365 --wandb
```


# Understanding Agent Decisions in PySCFabSim

In the `PySCFabSim` project, the agent's decisions are primarily related to dispatching lots to machines in a semiconductor manufacturing simulation. The agent's decisions are implemented in the `simulation/greedy.py` file and other related files. Here is a breakdown of how these decisions are made and implemented:

## Decision Points
The agent makes decisions at each decision point in the simulation. This is handled in the `next_decision_point` method of the `Instance` class. The decision points are where the agent decides which lots to dispatch to which machines.

## Dispatching Lots
The main decision the agent makes is which lots to dispatch to which machines. This is implemented in the `get_lots_to_dispatch_by_machine` and `get_lots_to_dispatch_by_lot` functions. These functions determine the best lots to dispatch based on the current state of the simulation.

## Dispatching Logic
The dispatching logic is implemented in the `dispatch` method of the `Instance` class. This method updates the state of the simulation based on the agent's decisions.

## Reward Structure
The reward structure for the agent is defined in the `DynamicSCFabSimulationEnvironment` class. The reward is calculated based on the performance of the dispatching decisions made by the agent.



## What is the Toolgroup Implant?

The toolgroup "Implant" refers to a set of machines used in the ion implantation process in semiconductor manufacturing. Ion implantation is a critical step where ions of a dopant material are accelerated and implanted into the silicon wafer to modify its electrical properties. This process is essential for creating regions of different conductivity in the semiconductor material, which are necessary for the functioning of transistors and other semiconductor devices.

