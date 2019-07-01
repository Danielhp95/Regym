import torch
import torch.optim as optim

from ..replay_buffers import Storage
from ..PPO import compute_loss as ppo_loss


class I2AAlgorithm():

    def __init__(self, imagination_core, model_free_network,
                 rollout_encoder, aggregator, actor_critic_head,
                 rollout_length, imagined_rollouts_per_step,
                 policies_update_horizon, environment_update_horizon,
                 environment_model_learning_rate, environment_model_adam_eps,
                 policies_adam_learning_rate, policies_adam_eps, use_cuda):
        self.use_cuda = use_cuda

        self.imagination_core = imagination_core
        self.rollout_length = rollout_length
        self.imagined_rollouts_per_step = imagined_rollouts_per_step
        self.rollout_encoder = rollout_encoder

        self.aggregator = aggregator

        self.model_free_network = model_free_network
        self.actor_critic_head = actor_critic_head

        self.storage = Storage(size=10) # TODO figure out storage mechanism

        self.policies_update_horizon    = policies_update_horizon
        self.environment_update_horizon = environment_update_horizon
        self.distill_optimizer = optim.Adam(imagination_core.distill_policy.parameters(),
                                           lr=policies_adam_learning_rate, eps=policies_adam_eps)

        #model_parameters = model_free_network.parameters() + rollout_encoder.parameters() + actor_critic_head.parameters()
        model_parameters = list(model_free_network.parameters()) + list(actor_critic_head.parameters())
        
        self.actor_critic_optimizer = optim.Adam(model_parameters,
                                                 lr=policies_adam_learning_rate,
                                                 eps=policies_adam_eps)
        '''
        self.environment_model_optimizer = optim.Adam(self.imagination_core.environment_model.parameters(),
                                                      lr=environment_model_learning_rate,
                                                      eps=environment_model_adam_eps)
        '''

    def take_action(self, state):
        # 1. Imagine state and reward for self.imagined_rollouts_per_step times
        imagined_states, imagined_rewards = self.imagination_core.imagine_rollout(state, self.rollout_length)
        # 2. encode them with RolloutEncoder and use aggregator to concatenate them together into imagination code
        imagination_code = None
        # 3. model free pass
        features = self.model_free_network(state)
        # 4. concatenate model free pass and imagination code
        # 5. Final fully connected layer which turns into action.
        imagination_code_features = features #torch.cat([imagination_code, features])
        prediction = self.actor_critic_head(imagination_code_features)
        return prediction

    def train_policies(self):
        # self.distill_optimizer.zero_grad()
        # self.actor_critic_optimizer.zero_grad()
        distill_loss       = self.compute_distill_policy_loss()
        actor_critic_loss = self.compute_actor_critic_policy_loss() # change name to compute model free lose (and pass PPO loss)
        # propagate loss backwards

    def train_environment_model(self):
        # self.environment_model_optimizer.zero_grad()
        model_loss = self.compute_environment_model_loss()
        # propagate loss backwards

    def compute_environment_model_loss(self): # TODO
        return None

    def compute_distill_policy_loss(self): # TODO
        # Note: this formula may only work with discrete actions?
        # Formula: cross_entropy_coefficient * softmax_probabilities(actor_critic_logit) * softmax_probabilities(distil_logit)).sum(1).mean()
        return None

    # NOTE: this loss will also propagate through the RolloutEncoder
    # Use PPO loss. Try and refactor PPOAlgorithm.optimize_model
    # into it's own function that we can call here?
    def compute_actor_critic_policy_loss(self): # TODO
        # loss = ppo_loss()
        return loss
