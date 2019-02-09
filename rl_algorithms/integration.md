## Needed interface: 

We need the following interface:

```python
    agent.handle_experience(state, individual_action, individual_reward, succ_state, done)
```

```python
    action = agent.take_action(state)
```

### PPO DeepRL

+ In BaseAgent, step uses pipes (only the BaseActor, so no worries!)

+ `PPO_agent.py` changes:
1. Implement the `handle_experience` and `take_action` to make PPO train.
2. Make sure we can run multiple training processes. 
3. AgentHooks interface. Abstracting that for all agents
4. Sending information through agenthooks.


For me, I will look into how to implement the training interface.


We only care about `PPO_agent`.
+ Abstract step function to:
1. Take `handle_experience` arguments. Inside `ppoagent.step` function
2. Give experience to be processed (maybe a optimize function). Middle of `ppoagent.step` function. Most of function
