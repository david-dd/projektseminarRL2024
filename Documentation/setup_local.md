# PySCFabSim - a mashine lerning powered semiconductor fab simulator for the SMT2020 Dataset

## Local install 

### Requirements 

- Python 3.9 / 3.10

### Download the Package

```shell
git clone https://github.com/david-dd/projektseminarRL2024.git
```

### Install dependencies 

install the Project requremnets 

```shell
cd projektseminarRL2024/
pip3.10 install -r requirements.txt
```

### optional requirements

#### Wandb-Plugin

Wandb allows you to monitor your Runs and compare them to eachother 

```shell
pip install wandb
wandb login
```

Follow login instructions


#### Chart-Plugin
```shell
pip install chart
```


### setup .env file 

coppy [env_template](./../.env_template) and rename it to .env than add your path to the repository 

```shell
cd projektseminarRL2024/
cp .env_template .env
```

open the `.env` File and follow the instructions there 

```shell
nano .env
```

## use the Project 

### Create a new experiment

to crate a new experiment we use [exp_set_gen_env.py](../exp_set_gen_env.py) 

the file usese the following variables from the .env file to create a new experiment 

`SYSTEM_PATH` is used for gym to properly source the enviorment.

`TOOLGROUP`, `TRAINING_PERIOD` and `REWARD` are used to define the experiment name and the setting that are being written to the `config.json`



```shell
python3.10 exp_set_gen_env.py

chmod g+w experiments
```

now the experiments folder sould have been created with a new experiemnt

the `config.json` file should look something like this 

```shell
{
  "name": "TF_BE_40_0_ds_HVLM_a9_tp630_reward4_fifo",
  "params": {
    "seed": 0,
    "dataset": "HVLM",
    "action_count": 9,
    "training_period": 630,
    "dispatcher": "fifo",
    "reward": 4,
    "station_group": "<TF_BE_40>"
  }
}
```

### Train a new Agent 

to train a new agent we use [rl_train_env.py](../rl_train_env.py) 

`SYSTEM_PATH` is used for gym to properly source the enviorment.

`EXPERIMENT_NAME` choses the experiment that is being used 

`USER_NUMBER` and `EXPERIMENT_NUMBER` are used to name the subfolder that has the trainig.weights and other files

`GREEDY_DAYS` and `TRAINING_STEPS` define how the agent is being trained 


```shell
python3.10 rl_train_env.py
```

after the training is done the new folder structure should look something like this 

```shell
.
└── experiments/
    ├── TF_BE_40_0_ds_HVLM_a9_tp630_reward1_fifo/
    │   ├── TF_BE_40_4_1_2024_11_28_09_46_23_336654_86615/
    │   │   ├── eval/
    │   │   │   ├── best_model/
    │   │   │   │   └── best_model.zip
    │   │   │   └── evaluations.npz
    │   │   ├── checkpoint/
    │   │   │   ├── checkpoint__100000_steps.zip
    │   │   │   └── ...
    │   │   ├── rewards_list.pkl
    │   │   ├── rewards_plot_TF_BE_40_4_1_2024_11_28_09_46_23_336654_86615.png
    │   │   └── trained.weights
    │   ├── TF_BE_40...
    │   └── config.json
    └── Implant_128...
```


### Evaluate the Agent 

to evaluate the agent we use [rl_test_env.py](../rl_test_env.py)

`SYSTEM_PATH` is used for gym to properly source the enviorment.

`EXPERIMENT_NAME_2` and `EXPERIMENT_SUBFOLDER` choses the experiment that is being used 

`TESTING_DAYS` the amount of days we evaluete the Agent over 


```shell
python3.10 rl_test_env.py
```

you can then view your results in wandb

