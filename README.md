# Generalized-RL-Self-Play-Framework

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

