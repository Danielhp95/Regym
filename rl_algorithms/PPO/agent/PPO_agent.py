#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class PPOAgent(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        # Create network here (will be changed for Robosymo)
        # Categorical Neural net, takes input from env
        self.config = config
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.task = config.task_fn()
        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []
        # self.states = self.task.reset()
        # self.states = config.state_normalizer(self.states)

    def handle_experience(self, s, a, r, succ_s, done):
        pass

    '''
    TODO potential issues:
        - Numpy array format
        - Storing the prediction in the self.Storage
    '''
    def take_action(self, state):
        prediction = self.network(state)
        action = to_np(prediction['a'])
        return action

    def step(self):
        '''
        1. The step function interacts with the environment for an _entire_ episode.
            - Takes actions
            - Stores transition information (Experience).
        2. Calculates values to regress towards
        2. Bookeping
            - Increases total steps
        '''

        '''
        1. The step function interacts with the environment for an _entire_ episode.
        '''
        config = self.config
        storage = Storage(config.rollout_length) # Make storage global, flush on episode termination?
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, _ = self.task.step(to_np(prediction['a']))
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})
            states = next_states

        '''
        2. Calculates values to regress towards
        '''
        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        '''
        2. Optimization step
        '''
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

        '''
        3. Bookeping
        '''
        steps = config.rollout_length * config.num_workers
        self.total_steps += steps


def build_PPO_Agent(action_dimensions, state_dimensions, env):
    '''
    Build a Config (same as DeepRL) codebase.
    :param env: multiagent environment where agent will be trained.
    :returns: PPOAgent adapted to be trained on given environment
    '''
    config = Config()
    config.discount = 0.99
    config.use_gae = Tru
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    # config.rollout_length = 128 No longer necessary. DeepRL implementation used fixed rollout lenghts
    config.optimization_epochs = 10
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.log_interval = 128 * 5 * 10
    config.network_fn = lambda: CategoricalActorCriticNet(state_dimensions, action_dimensions, NatureConvBody())
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=3e-4, eps=1e-5)
    return PPOAgent(config)
