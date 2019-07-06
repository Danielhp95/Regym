import torch


class ImaginationCore():
    '''
    Missing docstring
    '''

    def __init__(self, distill_policy, environment_model):
        '''
        :param distill_policy: Policy used to take actions in the 'imagined' rollouts
        :param environment_model: Model of an environment used to obtain rewards and observations
                                  based on an input observation and action. A perfect model is not assumed.
        '''
        self.distill_policy     = distill_policy
        self.environment_model = environment_model

    def imagine_rollout(self, initial_observation, num_steps):
        '''
        :param initial_observation: Observation / state (shape: batch x input_shape) 
                                    from which the rollout stems.
        :param num_steps: int > 0, number of steps to roll into the future.
        :returns: Tuple (concatenated rollout observations, concatenated rollout rewards)
        '''
        state = initial_observation
        rollout_states  = []
        rollout_rewards = []
        for step in range(num_steps):
            # roll forward using self.distill_policy to act in self.environment_model:
            prediction = self.distill_policy(state)
            action = prediction['a']
            # batch x 1 
            next_state, reward = self.environment_model(state, action)
            # append a new state and reward into rollout_states and rollout_rewards:
            rollout_states.append(next_state.unsqueeze(0))
            rollout_rewards.append(reward.unsqueeze(0))

            state = next_state

        # rollout_length x batch x state_shape / reward_size
        return torch.cat(rollout_states, dim=0), torch.cat(rollout_rewards, dim=0)
