# Reproduction of experiments

## Install requirements (suggested to run in a new virtual environment)

```shell
pip3.10 install -r requirements.txt
```

### optional requirements

Wandb-Plugin:
```shell
pip install wandb
wandb login         # Follow login instructions
```

Chart-Plugin
```shell
pip install chart
```

## setup .env file 

coppy [env_template](./.env_template) and rename it to .env than add your path to the repository 

(vs code rightclick on empty space in directory -> copy path)

on windows you have to use dubble "\\"

## start project 

```shell
python3.10 main.py
```

### options

defaults for options are alredy set

```shell
--dataset SMT2020_HVLM  # or SMT2020_LVHM
--days                  # simulation days
--dispatcher FIFO/CR/RANDOM
--seed                  # number: 1/2/3
--wandb                 # to activate the plugin
--chart
--alg
```




## Test FIFO and CR dispatching startegies

```shell
./reproduce_dispatcher_experiments.sh
```

## Dataset

Our simulator uses the SMT2020 dataset. It is available on https://p2schedgen.fernuni-hagen.de/index.php/downloads/simulation

Kopp, Denny & Hassoun, Michael & Kalir, Adar & Mönch, Lars. (2020). SMT2020—A Semiconductor Manufacturing Testbed. IEEE Transactions on Semiconductor Manufacturing. PP. 1-1. 10.1109/TSM.2020.3001933. 

## Ausführung

### Greedy-Programm

& C:/Users/willi/AppData/Local/Programs/Python/Python311/python.exe "c:/Users/willi/OneDrive/Documents/Studium/Diplomarbeit/Programm + Datengrundlage/PySCFabSim-release/main.py"


### RL Agent

#### Training	

& C:/Users/willi/AppData/Local/Programs/Python/Python311/python.exe "c:/Users/willi/OneDrive/Documents/Studium/Diplomarbeit/Programm + Datengrundlage/PySCFabSim-release/main.py" experiments\0_ds_HVLM_a9_tp365_reward2_di_fifo_Di\config.json

#### Testing

 & C:/Users/willi/AppData/Local/Programs/Python/Python311/python.exe "c:/Users/willi/OneDrive/Documents/Studium/Diplomarbeit/Programm + Datengrundlage/PySCFabSim-release/rl_test.py"