# Contributing: Adding a new algorithm / agent

This as an exhaustive list detailing *every single* step that needs to be done in order to add a new algorithm to be used in this RL framework. 

Please try to follow the conventions used in other implemented agent / algorithms. By adhering to the conventions presented here, any newly coded algorithm can be run against our test suite and benchmarked against other existing agents and algorithms.

### Overview

Adding a new agent / algorithm can be separated into the following tasks:

1. Implement an `Agent` class which, following a given interface, will collect the environment information required for the new algorithm to operate.
2. Implement the python code for the new algorithm which uses the environment information gathered by the `Agent` from step 1.
3. Create a new test in the test suite, invoking our algorithm agnostic test suite, to make sure that the new algorithm is compatible with a variety of OpenAI environments, and that it can solve these.
4. Updating the relevant python submodules to reflect the existance of the new agent.


## Adding a new agent / algorithm to `regym` 

Look at the existing [Tabular Q Learning](../regym/rl_algorithms/agents/tabular_q_learning_agent.py) and [REINFORCE](../regym/rl_algorithms/agents/reinforce_agent.py) implementations for "simple" examples of what constitutes an agent and algorithm in our framework.

Let's assume we want to implement the (fictitious) state-of-the-art AlgorithmX into our framework. Adding AlgorithmX requires the following:

### 1. Create file containing AlgorithmXAgent code

**1.1:** Create a file in the submodule `regym/rl_algorithms/agents` named `algorithmX_agent.py`

```python
# This file path: `regym/rl_algorithms/agents/algorithmX_agent.py`

from regym.rl_algorithms.algorithmx import AlgorithmX
class AlgorithmXAgent():

    def __init__(self, algorithm, other_args):
        self.training  = True
        self.algorithm = algorithm
        # ...

    def handle_experience(self, state, action, reward, succesor_state, done):
        should_train = ...
        if should_train: self.algorithm.train(...)
        pass

    def take_action(self, state):
        action = self.algorithm.model(state)
        # ...
        return action

    def clone(self, training):
        pass

def build_AlgorithmX_Agent(task, config, agent_name):
    algorithm = AlgorithmX(observation_dim=task.observation_dim,
                           action_dim=task.action_dim,
                           learning_rate=config['learning_rate']...)
    return AlgorithmXAgent(algorithm, ...)
```

Following the usual Reinforcement Learning fashion, the agent will be asked fo an action by calling the `take_action(...)` function. Operationally, calling this function is equivalent to sampling an action from the agent's policy. The `handle_experience(...)` function is called after the environment processes the agent(s) actions, and feeds a single "experience" to the agent to be processed. We recommend that the agent implementes the logic to call its underlying algorithm to update its policy inside of this function. Finally, the `clone(...)` function can be implemented to create a clone of the agent, useful in [self-play training](https://danielhp95.github.io/assets/pdfs/COG-2019-submission.pdf). A deep copy is encouraged, using the `copy.deepcopy` method from the `copy` built-in python module.

It's *very* important that the `build_AlgorithmX_Agent(task, config, agent_name)` maintains that function signature, as it gives us a common interface to generate agents with a variety of underlying RL algorithms.

It is inside of this function that you can inspect the `task` object parameter to decide how to initialize the agent and its underlying algorithm. The example above uses the `task.observation_dim` and `task.action_dim` as input to the `AlgorithmX`, which in turn will generate a neural network with an input / output dimension of the same value as the task's action / observation space. Depending on the `task.observation_type` and `task.action_type` you may want to change the topology of the underlying neural networks. For instance, for a `Discrete` action space, you may want to use a categorical distribution, but a `Continuous` action space may require to place a Gaussian distribution over each action dimension (look at [PPO](../regym/rl_algorithms/agents/ppo_agent.py) for an example of this).

**1.2:** Add to the existing `regym/rl_algorithms/agents/__init__.py` an import to both the new agent class and it's build function. In other words, add:

```python
from .algorithmX_agent import build_AlgorithmX_Agent, AlgorithmXAgent
```

### 2 Add a new algorithm 
**2.1** Add a new AlgorithmX module inside the existing `regym/rl_algorithms/` (e.g a create directory `regym/rl_algorithms/AlgorithmX/`)  

**2.2** The next part will be the most time consuming one. Here we need to code up AlgorithmX. We recommend to create an `AlgorithmX` class inside of the `regym.rl_algorithms.AlgorithmX` submodule. This class should hold a `model`, which will represent the policy (and perhaps also the value network, if `AlgorithmX` is a value based RL algorithm).

We have so far used [PyTorch](https://pytorch.org/) as a deep learning framework, and hence our models are `torch.nn.Module` objects. However, `regym` allows for any deep learning frameworks.

Finally, we recommend that, during `AlgorithmXAgent.handle_experience(...)` function, the function `AlgorithmX.train(...)` is called. The [REINFORCE](../regym/rl_algorithms/reinforce/reinforce.py) algorithm calls this function after a given number of finished episodes. [PPO](../regym/rl_algorithms/PPO/ppo.py) calls this function after a certain number of timesteps have elapsed. What this means is that each algorithm will trigger a policy update (a train operation) under different conditions. Regardless, this call always happens inside of the agent's `handle_experience(...)` function.

```python
# This file path: `regym/rl_algorithms/AlgorithmX/algorithmx.py`
class AlgorithmX():

    def __init__(self, observation_dim, action_dim, learning_rate):
        self.model = CoolNeuralNetwork(input_dim=observation_dim,
                                       output_dim=action_dim, lr=learning_rate)

    def train(self, args):
        pass
```

**2.3** Create `regym/rl_algorithms/AlgorithmX/__init__.py` containing:

```python
from .algorithmx import AlgorithmX
```

Now our `algorithmX_agent.py` module will be able to import `AlgorithmX`.

### 3 Use `regym`'s test suite. (TODO)

Future versions of `regym` will include an algorithm agnostic which will test whether the `build_X_Agent(...)` function is compatible with a variety of environments, whether it can learn to solve these and whether or not it's behaviour is deterministic under certain fixed seeds.

### 4 Ensure that the agent can be accessed through the `regym.rl_algorithms` submodule

Try launching the jupyter notebook [getting-started](./notebooks/getting-started.ipynb), but replace the imported `build_PPO_Agent(...)` function with the `build_AlgorithmX_Agent(...)` function you've implemented (remember to pass the correct hyperparameter dictionary!). Can your agent solve the CartPole environment?

If so. Well done! And thank you for contributing to `regym`!
