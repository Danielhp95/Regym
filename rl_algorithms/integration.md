## Needed interface: 

We need the following interface:

```python
    agent.handle_experience(state, individual_action, individual_reward, succ_state, done)
```

```python
    action = agent.take_action(state)
```

### PPO DeepRL

+ In BaseAgent, step uses pipes.


We only care about `PPO_agent`.
+ Abstract step function to:
1. Take `handle_experience` arguments. Inside `ppoagent.step` function
2. Give experience to be processed (maybe a optimize function). Middle of `ppoagent.step` function. Most of function
+ Check how pipes are used to send actions to environment.


