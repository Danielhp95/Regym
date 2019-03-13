# Description:

The test to consider is ```ppo_test.py```. It can be launched with the usual pytest command.

The files to consider are:
```
f1: ../../multiagent_loops/simultaneous_action_rl_loop.py
f2: ../../rl_algorithms/agents/ppo_agent.py
f3: ../../rl_algorithms/PPO/ppo.py
```

Inside f1, the run_episode function is to be considered, and more specifically the ```inner_loop``` variable.
It controls whether the experiences are handled within the RL loop by the agents or on the outside.

Inside f2, there are two sets of ```handle_experience``` and ```take_action``` functions. One is commented out (top one), it is the one that is needed for the multi_actor approach to work (with an outer-loop handling of the experience). Let us call it the multi-actor pair. On the other hand, the one that is not commented out (bottome one), is the closest rendition of the original functions that we paircoded together. Let us call it the original pair. 

# Issues:

## Condition 1:

While using (inner_loop experience handling and original pair), the ppo_test runs without failing to beat the rock agent. But, when using (outer_loop experience handling and original pair), ppo fails to learn to consistently beat rock agent. This is the first issue. From the viewpoint of ppo, there should not be any differences between the two previous schemes, and yet the results of the test are different. I do not understand what is at play.

## Condition 2:
Next, since the goal is to make sure that the multi-actor approach is working, it would be good that the (outer-loop experiernce handling and multi-actor pair) scheme succeeds to beat rock consistenly. Yet, the multi-actor pair of functions with any experience handling procedure keeps failing to learn to beat rock.

 