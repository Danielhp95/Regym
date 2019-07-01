import torch


class ImaginationCore():
    '''
    Missing docstring
    '''

    def __init__(self, distill_policy, environment_model):
        '''
        :param distill_policy: Policy used in perform 'imagined' rollouts
        :param environment_model: Model of an environment used to obtain rewards and observations
                                  based on an input observation and action. A perfect model is not assumed.
        '''
        self.distill_policy     = distill_policy
        self.environment_model = environment_model

    def imagine_rollout(self, initial_observation, num_steps):
        '''
        :param initial_observation: Observation / state from which forward planning will commence
        :param num_steps: int > 0, number of steps to roll into the future.
        :returns: Tuple (concatenated rollout observations, concatenated rollout rewards)
        '''
        rollout_states  = [torch.Tensor([1])]
        rollout_rewards = [torch.Tensor([1])]
        for step in range(num_steps):
            # TODO roll forward using self.distill_policy to act in self.environment_model to
            # append a new state and reward into rollout_states and rollout_rewards
            pass
        return torch.cat(rollout_states), torch.cat(rollout_rewards)
