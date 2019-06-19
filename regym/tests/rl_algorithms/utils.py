def can_act_in_environment(task, build_agent_function, config_dict, name, num_actions=5):
    env = task.env
    agent = build_agent_function(task, config_dict, name)
    for _ in range(num_actions):
        random_observation = env.observation_space.sample()
        action = agent.take_action(random_observation)
        assert env.action_space.contains(action)
