import torch.optim as optim

from ..replay_buffers import Storage


class I2AAlgorithm():

    def __init__(self, rollout_length, imagined_trajectories_per_step,
                 policies_update_horizon, environment_update_horizon,
                 environment_model_learning_rate, environment_model_adam_eps,
                 policies_adam_learning_rate, policies_adam_eps, use_cuda):
        self.use_cuda = use_cuda
        self.policies_update_horizon    = policies_update_horizon
        self.environment_update_horizon = environment_update_horizon

        self.storage = Storage(size=10) # TODO figure out storage mechanism
        # self.distil_optimizer       = optim.Adam(DISTIL_MODEL, lr=policies_adam_learning_rate, eps=policies_adam_eps)
        # self.actor_critic_optimizer = optim.Adam(ACTOR_CRITIC_MODEL, lr=policies_adam_learning_rate, eps=policies_adam_eps)
        # self.environment_model_optimizer = optim.Adam(ENVIRONMENT_MODEL, lr=environment_model_learning_rate, eps=environment_model_adam_eps)

    def train_policies(self):
        # self.distil_optimizer.zero_grad()
        # self.actor_critic_optimizer.zero_grad()
        distil_loss             = self.compute_distil_policy_loss()
        action_loss, value_loss = self.compute_actor_critic_policy_loss()
        # propagate loss backwards

    def train_environment_model(self):
        # self.environment_model_optimizer.zero_grad()
        model_loss = self.compute_environment_model_loss()
        # propagate loss backwards

    def compute_environment_model_loss(self):
        return None

    def compute_distil_policy_loss(self):
        return None

    def compute_actor_critic_policy_loss(self):
        return None, None
