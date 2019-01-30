# Generalized-RL-Self-Play-Framework

This is the repository hosting the code used for the paper (AWESOME PAPER NAME).

Installation
Getting Started  (turn into links)
Running experiment
Results
Modules

## Installation

First, clone this repository:

```
git clone https://github.com/Danielhp95/Generalized-RL-Self-Play-Framework
cd Generalized-RL-Self-Play-Framework
(For integrating PPO ) git checkout develop-ppo-integration
```

### Dependencies

#### Python version

For this experiment we use Python `3.6`. Here's a [guide to installing Python `3.6` in Ubuntu 16.04](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/)

### Python Dependencies

For managing Python dependencies we use [pipenv](https://readthedocs.org/projects/pipenv/). Once you have cloned the repository, proceed to install the dependencies defined in the `Pipfile`

#### pipenv 
This can be done by calling `pipenv install` inside of the directory holding this repository.

```
pipenv install
```

#### conda

```
conda install --yes --file requirements.txt   (need to generate requirements.txt file first)
``` 

(Alex: write requirements.txt)
(Dani: update gym_rock_paper_scissors gym)

### Getting started:

This experiment aims at measuring the effect self-play training schemes on X different environments, and Y different algorithms and 
Z self-play training schemes. The effect is measured by looking at:

+ The winrates of every a self-play scheme and algorithm pair against all other pairs after a specific number of episodes
+ The evolution of these winrates over time.
+ The episodic reward of every individual training policy. 

The experiment trains in parallel a set of self-play schemes / algorithm pairs on a given environment, creating a separate processes for each SP scheme, algorithm environment. A list of  **benchmarking checkpoints** is specified, containing episode numbers. Once a training process, successfully simulates a number of episodes specified in **benchmarking checkpoints**, a benchmarking procedure begins, where it is  policies are frozen and benchmarked against one another. The benchmarking that takes place only fares against (could be interesting to benchmark old menageries vs new menageries)  

    Usage:
      run [options]

    Options:
      --environment STRING                    OpenAI environment used to train agents on
      --experiment_id STRING                  Experimment id used to identify between different experiments
      --number_of_runs INTEGER                Number of runs used to calculate standard deviations for various metrics
      --checkpoint_at_iterations INTEGER...   Iteration numbers at which agents will be benchmarked against one another
      --benchmarking_episodes INTEGER         Number of head to head matches used to infer winrates between agents
      --self_play_training_schemes STRING...  Self play training schemes used to choose opponent agent agents during training
      --algorithms STRING...                  Algorithms used to learn a agent
      --fixed_agents STRING...                Fixed agents used to benchmark training agents against

### Sending policies between processes
Currently: 
    DQN: 
        Saves pytorch model (weights and graph) into disk.
        Sends a python object (AgentHook) which contains the logic for (re)building an agent that has been saved.


## Modules

### `rl_algorithms/`

Module containing a set of classes representing various RL algorithms.

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
