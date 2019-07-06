from typing import Dict
import torch
import torch.nn as nn


class RolloutEncoder(nn.Module):
    '''
    Encoder which takes a rollout from an environmet model and reduces it
    to a rollout embedding in a latent feature space.
    Refer to section 3.2 of original paper: https://arxiv.org/pdf/1707.06203.pdf
    '''
    def __init__(self, input_shape, nbr_states_to_encode: int,
                 feature_encoder: nn.Module, rollout_feature_encoder: nn.Module,
                 kwargs: Dict[str, object]):
        '''
        :param input_shapes: dimensions of the input.
        :param nbr_states_to_encode: number of states to encode in the rollout embeddings, from the end of the rollout.
        :param feature_encoder: nn.Modules that encodes the observations into features.
        :param rollout_feature_encoder: recurrent nn.Modules that encodes
                                        a rollout of features into a rollout embedding.
        :param kwargs: keyword arguments.
        '''
        super(RolloutEncoder, self).__init__()
        self.input_shape = input_shape
        self.nbr_states_to_encode = nbr_states_to_encode
        self.kwargs = kwargs
        self.feature_encoder = feature_encoder
        self.rollout_feature_encoder = rollout_feature_encoder
        self.fc_rollout_embeddings = nn.Linear(self.kwargs['rollout_encoder_nbr_hidden_units'], self.kwargs['rollout_encoder_embedding_size'])

    def forward(self, states: torch.Tensor, rewards: torch.Tensor):
        '''
        Processes the :param states: and :param rewards: to generate
        latent rollout embeddings.
        :param states: (rollout_length x batch x depth x height x width )
                        Tensor of rollout episode states.
        :param rewards: (rollout_length x batch x reward_size)
                        Tensor of rollout episode reward.
        :returns: torch.Tensor latent embeddings of the :param states: and
                  :param rewards:
        '''

        batch_size = states.size(1)

        # batching all the states of all the rollouts:
        states2encode = states[-self.nbr_states_to_encode:].view(-1, *(self.input_shape))
        rewards = rewards[-self.nbr_states_to_encode:]

        features = self.feature_encoder(states2encode)
        # reformating into nbr_states_to_encode x batch x feature_dim:
        features = features.view(self.nbr_states_to_encode, batch_size, -1)

        feat_rewards = torch.cat([features, rewards], dim=2)
        # reversing the rollouts:
        reversed_feat_rewards = torch.cat([feat_rewards[i].unsqueeze(0) for i in range(feat_rewards.size(0)-1, -1, -1)], dim=0)

        # Forward pass:
        outputs, next_rnn_states = self.rollout_feature_encoder(reversed_feat_rewards)

        # rollout_length x batch_size x hidden_size
        rollout_embeddings = self.fc_rollout_embeddings(outputs[-1].view(batch_size, -1))
        # batch x rollout_encoder_embedding_size
        return rollout_embeddings
