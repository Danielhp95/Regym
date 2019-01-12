# Generalized-RL-Self-Play-Framework


## Installation

This experiment uses [pipenv](https://pipenv.readthedocs.io/en/latest/) with a Python 3.6.5 to manage Python dependencies. The python environments used are all [OpenAI gym](https://gym.openai.com/) environments.

### Dependencies

1. [Guide to installing pipenv](https://pipenv.readthedocs.io/en/latest/install/#installing-pipenv)
2. [Intall Mujoco and mujoco-py](https://github.com/openai/mujoco-py) (This is an involved process)


### Installing this project

Clone this repository, `cd` into it and run the command:

```bash
pipenv install
```

If Mujoco and mujoco-py have been installed properly, you should now be able to run the experiment.

## Environments used

+ [Repeated game of rock paper scissors](https://github.com/Danielhp95/gym-rock-paper-scissors)
+ [Robotic sumo](https://github.com/openai/robosumo)

## Running the experiment in your local machine

The experiment is run through the `run.py` script in the `experiment/` module. The `run.py` script takes the following command line arguments:

```
    Usage:
      run [options]

    Options:
      --environment STRING                    OpenAI environment used to train agents on
      --experiment_id STRING                  Experimment id used to identify between different experiments
      --number_of_runs INTEGER                Number of runs used to calculate standard deviations for various metrics
      --checkpoint_at_iterations INTEGER...   Iteration numbers at which policies will be benchmarked against one another
      --benchmarking_episodes INTEGER         Number of head to head matches used to infer winrates between policies
      --self_play_training_schemes STRING...  Self play training schemes used to choose opponent agent policies during training
      --algorithms STRING...                  Algorithms used to learn a policy
      --fixed_agents STRING...                Fixed agents used to benchmark training policies against

```

## Modules

### `rl_algorithms/`

Module containing a set of classes representing various RL algorithms.

Currently implemented:
+ Tabular Q Learning.
+ DQN derivative (**not done**)


 The interface they must follow is:

```python
class Algorithm():

    def __init__(self, state_space_size, action_space_size, hashing_function, learning_rate, training):
        pass

    def handle_experience(self, state, action, reward, succesor_state):
        pass

    def take_action(self, state):
        pass

    def clone(self, training):
        pass
```


### `training_schemes/`

(Only self play for now) Different training schemes. Currently we have
+ Naive Self Play
+ Delta Distributional Self Play

### `multiagent_loops/`
(should probably rename to environment loops)

Module that defines environment loops. Some environments need agents to submit all actions simultaneously, such as Rock Paper Scissors, whilst others like Go need one action at a time.

### `experiment/`

