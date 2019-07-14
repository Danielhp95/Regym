import torch
import gc


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

    def imagine_rollout(self, initial_observation, num_steps, first_action=None):
        '''
        :param initial_observation: Observation / state (shape: batch x input_shape) 
                                    from which the rollout stems.
        :param num_steps: int > 0, number of steps to roll into the future.
        :param first_action: action(s) to execute on the first imagined step (shape: batch x action_shape).
        :returns: Tuple (concatenated rollout observations, concatenated rollout rewards)
        '''
        batch_size = initial_observation.size(0)
        state = initial_observation
        rollout_states  = []
        rollout_actions  = []
        rollout_rewards = []
        for step in range(num_steps):
            # roll forward using self.distill_policy to act in self.environment_model:
            if step==0 and first_action is not None:
                action = first_action
            else:
                if self.environment_model.use_cuda: state=state.cuda()
                prediction = self.distill_policy(state)
                action = prediction['a']
            # batch x 1 

            '''
            next_states = []
            rewards = []
            state = state.cpu()
            action = action.cpu()
            bs = torch.zeros((1, *(state.size()[1:])))
            ba = torch.zeros((1, *(action.size()[1:])))
            if self.environment_model.use_cuda:
                bs = bs.cuda()
                ba = ba.cuda()
            for bidx in range(batch_size):
                bs.copy_(state[bidx].unsqueeze(0))
                ba.copy_(action[bidx].unsqueeze(0))
                next_state, reward = self.environment_model(bs, ba)
                next_states.append(next_state.cpu())
                rewards.append(reward.cpu())
                del next_state
                del reward
            torch.cuda.empty_cache()
            #if batch_size > 8:
            #    gc.collect()

            next_state = torch.cat(next_states, dim=0)
            reward = torch.cat(rewards, dim=0)
            '''

            next_state, reward = self.environment_model(state, action)
            next_state = next_state.cpu()
            reward = reward.cpu()
            action = action.cpu()
            
            # append a new state and reward into rollout_states and rollout_rewards:
            rollout_states.append(next_state.unsqueeze(0))
            rollout_actions.append(action.unsqueeze(0))
            rollout_rewards.append(reward.unsqueeze(0))

            state = next_state

        # rollout_length x batch x state_shape / reward_size
        return torch.cat(rollout_states, dim=0), torch.cat(rollout_actions, dim=0), torch.cat(rollout_rewards, dim=0)
