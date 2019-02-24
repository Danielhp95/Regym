from copy import deepcopy
import torch.optim as optim
import numpy as np


class PPOAlgorithm():

    def __init__(self, kwargs):
        '''
        episode_interval_per_training:
        discount:
        use_gae:
        use_cuda:
        gae_tau:
        entropy_weight:
        gradient_clip:
        optimization_epochs:
        mini_batch_size:
        ppo_ratio_clip:
        learning_rate:
        adam_eps:
        model:
        replay_buffer:
        "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
        "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
        "use_cuda": boolean to specify whether to use CUDA.
        '''
        self.kwargs = deepcopy(kwargs)
        self.model = self.kwargs['model']
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()

        # self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['learning_rate'], eps=kwargs['adam_eps'])

    def optimize_model(self):
        raise NotImplementedError('Not yet implemented')
