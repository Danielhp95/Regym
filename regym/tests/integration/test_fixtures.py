import pytest
from regym.environments import parse_environment


@pytest.fixture
def ppo_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = True
    config['use_cuda'] = True
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 32
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 1024
    config['phi_arch'] = 'RNN'
    config['actor_arch'] = 'MLP'
    config['critic_arch'] = 'MLP'
    return config


@pytest.fixture
def i2a_config_dict():
    config = dict()
    config['nbr_frame_stacking'] = 4
    config['rollout_length'] = 3
    config['rollout_encoder_nbr_state_to_encode'] = 5
    config['reward_size'] = 1
    config['imagined_rollouts_per_step'] = None
    config['nbr_actor'] = 8
    
    # Assuming CNN task:
    config['preprocess_function'] = 'ResizeCNNPreprocessFunction'
    config['observation_resize_dim'] = 80
    # Assuming FC task:
    #config['preprocess_function'] = 'PreprocessFunction'
    
    config['use_cuda'] = True

    config['environment_model_learning_rate'] = 3.0e-4
    config['environment_model_adam_eps'] = 1.0e-5
    config['policies_adam_learning_rate'] = 3.0e-4
    config['policies_adam_eps'] = 1.0e-5

    # Model Training Algorithm hyperparameters:
    config['model_training_algorithm'] = 'PPO'
    # PPO hyperparameters:
    config['discount'] = 0.99
    config['use_gae'] = True
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 128
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5

    # Environment Model: Architecture description:
    config['environment_model_update_horizon'] = 1024
    config['environment_model_gradient_clip'] = 5
    config['environment_model_batch_size'] = 128
    # (Recurrent) Convolutional Architecture:
    '''
    config['environment_model_arch'] = 'CNN' #'CNN-GRU-RNN'
    config['environment_model_enc_channels'] = [32, 32, 32]
    config['environment_model_enc_kernels'] = [6, 4, 3]
    config['environment_model_enc_strides'] = [6, 2, 1]
    config['environment_model_enc_paddings'] = [0, 1, 1]
    config['environment_model_enc_feature_dim'] = 256
    config['environment_model_enc_hidden_units'] = (64,)    # recurrent cells
    config['environment_model_dec_channels'] = [32, 32, 32]
    config['environment_model_dec_kernels'] = [6, 4, 3]
    config['environment_model_dec_strides'] = [6, 2, 1]
    config['environment_model_dec_paddings'] = [0, 1, 1]
    config['environment_model_dec_feature_dim'] = 256
    config['environment_model_dec_hidden_units'] = (64,)    # recurrent cells
    '''
    # Sokoban Architecture:
    config['environment_model_arch'] = 'Sokoban'
    config['environment_model_channels'] = [32]
    # Fully-Connected Architecture:
    '''
    config['environment_model_arch'] = 'MLP'
    config['environment_model_enc_nbr_hidden_units'] = [512, 256, 128]
    config['environment_model_dec_nbr_hidden_units'] = [256, 512]
    '''
        

    # Rollout Encoder:
    # Recurrent Convolutional Architecture:
    config['rollout_encoder_model_arch'] = 'CNN-GRU-RNN'
    config['rollout_encoder_channels'] = [16, 32, 64]
    config['rollout_encoder_kernels'] = [8, 4, 3]
    config['rollout_encoder_strides'] = [8, 2, 1]
    config['rollout_encoder_paddings'] = [0, 1, 1]
    config['rollout_encoder_feature_dim'] = 512
    config['rollout_encoder_nbr_hidden_units'] = 256
    config['rollout_encoder_nbr_rnn_layers'] = 1
    config['rollout_encoder_embedding_size'] = 256
    # Recurrent Fully-Connected Architecture:
    '''
    config['rollout_encoder_model_arch'] = 'GRU-RNN'
    config['rollout_encoder_nbr_rnn_layers'] = 1
    config['rollout_encoder_embedding_size'] = 32
    '''
        

    # Distilled Policy:
    config['distill_policy_update_horizon'] = 2048
    config['distill_policy_gradient_clip'] = 5
    config['distill_policy_batch_size'] = 32
    # Convolutional architecture:
    config['distill_policy_arch'] = 'CNN'
    config['distill_policy_channels'] = [16, 32, 64]
    config['distill_policy_kernels'] = [8, 4, 3]
    config['distill_policy_strides'] = [8, 2, 1]
    config['distill_policy_paddings'] = [0, 1, 1]
    config['distill_policy_feature_dim'] = 512
    # Fully-Connected architecture:
    '''
    config['distill_policy_arch'] = 'MLP'
    config['distill_policy_nbr_hidden_units'] = [128]
    '''
    # Distilled Policy: Actor Head architecture:
    config['distill_policy_head_arch'] = 'MLP'
    config['distill_policy_head_nbr_hidden_units'] = (128,)


    # Model :
    config['model_update_horizon'] = 2048
    # Model-Free Path: Convolutional architecture:
    config['model_free_network_arch'] = 'CNN'
    config['model_free_network_channels'] = [32, 32, 64]
    config['model_free_network_kernels'] = [8, 4, 3]
    config['model_free_network_strides'] = [8, 2, 1]
    config['model_free_network_paddings'] = [0, 1, 1]
    config['model_free_network_feature_dim'] = 512
    # Model-Free Path: Fully Connected architecture description
    config['model_free_network_nbr_hidden_units'] = None
    # Actor Critic Head:
    config['achead_phi_arch'] = 'GRU-RNN'
    config['achead_phi_nbr_hidden_units'] = (256,)
    config['achead_actor_arch'] = 'MLP'
    config['achead_actor_nbr_hidden_units'] = (128,)
    config['achead_critic_arch'] = 'MLP'
    config['achead_critic_nbr_hidden_units'] = (128,)

    # Random Network Distillation:
    config['use_random_network_distillation'] = False
    config['intrinsic_discount'] = 0.99
    config['rnd_loss_int_ratio'] = 0.5
    config['rnd_feature_net_fc_arch_hidden_units'] = (128, 64)  #if arch is MLP
    config['rnd_feature_net_cnn_arch_feature_dim'] = 64         #if arch is CNN
    config['rnd_update_period_running_meanstd_int_reward'] = 1e5
    # Convolutional Architecture:
    config['rnd_arch_channels'] = [32, 32, 32]
    config['phi_arch_kernels'] = [6, 4, 3]
    config['phi_arch_strides'] = [6, 2, 1]
    config['phi_arch_paddings'] = [0, 1, 1]
    config['phi_arch_feature_dim'] = 256
    config['phi_arch_hidden_units'] = (64,)
    
    return config


@pytest.fixture
def i2a_rnd_config_dict():
    config = dict()
    
    config['nbr_frame_stacking'] = 4
    config['rollout_length'] = 3
    config['rollout_encoder_nbr_state_to_encode'] = 5
    config['reward_size'] = 1
    config['imagined_rollouts_per_step'] = 3 #None = num_actions
    config['nbr_actor'] = 4
    
    # Assuming CNN task:
    config['preprocess_function'] = 'ResizeCNNPreprocessFunction'
    config['observation_resize_dim'] = 80
    # Assuming FC task:
    #config['preprocess_function'] = 'PreprocessFunction'
    
    config['use_cuda'] = True

    config['environment_model_learning_rate'] = 3.0e-4
    config['environment_model_adam_eps'] = 1.0e-5
    config['policies_adam_learning_rate'] = 3.0e-4
    config['policies_adam_eps'] = 1.0e-5

    # Model Training Algorithm hyperparameters:
    config['model_training_algorithm'] = 'PPO'
    # PPO hyperparameters:
    config['discount'] = 0.99
    config['use_gae'] = True
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 4
    config['mini_batch_size'] = 128
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5

    # Environment Model: Architecture description:
    config['environment_model_update_horizon'] = 128
    config['environment_model_gradient_clip'] = 5
    config['environment_model_batch_size'] = 128
    config['environment_model_optimization_epochs'] = 4
    # (Recurrent) Convolutional Architecture:
    '''
    config['environment_model_arch'] = 'CNN' #'CNN-GRU-RNN'
    config['environment_model_enc_channels'] = [32, 32, 32]
    config['environment_model_enc_kernels'] = [6, 4, 3]
    config['environment_model_enc_strides'] = [6, 2, 1]
    config['environment_model_enc_paddings'] = [0, 1, 1]
    config['environment_model_enc_feature_dim'] = 256
    config['environment_model_enc_hidden_units'] = (64,)    # recurrent cells
    config['environment_model_dec_channels'] = [32, 32, 32]
    config['environment_model_dec_kernels'] = [6, 4, 3]
    config['environment_model_dec_strides'] = [6, 2, 1]
    config['environment_model_dec_paddings'] = [0, 1, 1]
    config['environment_model_dec_feature_dim'] = 256
    config['environment_model_dec_hidden_units'] = (64,)    # recurrent cells
    '''
    # Sokoban Architecture:
    '''
    config['environment_model_arch'] = 'Sokoban'
    config['environment_model_channels'] = [32]
    '''
    # Fully-Connected Architecture:
    config['environment_model_arch'] = 'MLP'
    config['environment_model_enc_nbr_hidden_units'] = (128,)
    config['environment_model_dec_nbr_hidden_units'] = (128,)
        

    # Rollout Encoder:
    # Recurrent Convolutional Architecture:
    '''
    config['rollout_encoder_model_arch'] = 'CNN-GRU-RNN'
    config['rollout_encoder_channels'] = [32, 32, 64]
    config['rollout_encoder_kernels'] = [8, 4, 3]
    config['rollout_encoder_strides'] = [4, 2, 1]
    config['rollout_encoder_paddings'] = [0, 1, 1]
    config['rollout_encoder_feature_dim'] = 512
    config['rollout_encoder_encoder_nbr_hidden_units'] = 256
    config['rollout_encoder_nbr_rnn_layers'] = 1
    config['rollout_encoder_embedding_size'] = 256
    '''
    # Recurrent Fully-Connected Architecture:
    config['rollout_encoder_model_arch'] = 'MLP-GRU-RNN'
    config['rollout_encoder_nbr_hidden_units'] = (256,)
    config['rollout_encoder_encoder_nbr_hidden_units'] = 128
    config['rollout_encoder_nbr_rnn_layers'] = 1
    config['rollout_encoder_embedding_size'] = 128
        

    # Distilled Policy:
    config['distill_policy_update_horizon'] = 512
    config['distill_policy_gradient_clip'] = 5
    config['distill_policy_batch_size'] = 32
    config['distill_policy_optimization_epochs'] = 4
    # Convolutional architecture:
    '''
    config['distill_policy_arch'] = 'CNN'
    config['distill_policy_channels'] = [32, 32, 64]
    config['distill_policy_kernels'] = [8, 4, 3]
    config['distill_policy_strides'] = [4, 2, 1]
    config['distill_policy_paddings'] = [0, 1, 1]
    config['distill_policy_feature_dim'] = 256
    '''
    # Fully-Connected architecture:
    config['distill_policy_arch'] = 'MLP'
    config['distill_policy_nbr_hidden_units'] = (128,)
    # Distilled Policy: Actor Head architecture:
    config['distill_policy_head_arch'] = 'MLP'#'GRU/LSTM-RNN'
    config['distill_policy_head_nbr_hidden_units'] = (128,)


    # Model :
    config['model_update_horizon'] = 128
    # Model-Free Path: Convolutional architecture:
    '''
    config['model_free_network_arch'] = 'CNN'
    '''
    config['model_free_network_channels'] = [32, 64, 64]
    config['model_free_network_kernels'] = [8, 4, 3]
    config['model_free_network_strides'] = [4, 2, 1]
    config['model_free_network_paddings'] = [0, 1, 1]
    config['model_free_network_feature_dim'] = 256
    # Model-Free Path: Fully Connected architecture description
    config['model_free_network_arch'] = 'MLP'
    config['model_free_network_nbr_hidden_units'] = (128,)
    # Actor Critic Head:
    config['achead_phi_arch'] = 'GRU-RNN'
    config['achead_phi_nbr_hidden_units'] = (128,)
    config['achead_actor_arch'] = 'MLP'
    config['achead_actor_nbr_hidden_units'] = (128,)
    config['achead_critic_arch'] = 'MLP'
    config['achead_critic_nbr_hidden_units'] = (128,)

    # Random Network Distillation:
    config['use_random_network_distillation'] = True
    config['intrinsic_discount'] = 0.99
    config['rnd_loss_int_ratio'] = 0.5
    config['rnd_obs_clip'] = 5
    config['rnd_non_episodic_int_r'] = True
    config['rnd_update_period_running_meanstd_int_reward'] = 1e5
    config['rnd_update_period_running_meanstd_obs'] = config['rnd_update_period_running_meanstd_int_reward']
    # RND Convolutional Architecture:
    config['rnd_arch'] = 'CNN'
    config['rnd_arch_channels'] = [32, 64, 64]
    config['rnd_arch_kernels'] = [8, 4, 3]
    config['rnd_arch_strides'] = [4, 2, 1]
    config['rnd_arch_paddings'] = [0, 1, 1]
    config['rnd_arch_feature_dim'] = 512
    # RND Fully-Connected Architecture:
    '''
    config['rnd_feature_net_fc_arch_hidden_units'] = (128, 64)
    '''

    # Latent Embedding:
    config['use_latent_embedding'] = True
    config['latent_emb_nbr_variables'] = 128
    # Latent Encoder:

    # Latent Encoder: Convolutional architecture:
    config['latent_encoder_arch'] = 'CNN'
    config['latent_encoder_channels'] = [32, 32, 64]
    config['latent_encoder_kernels'] = [8, 4, 3]
    config['latent_encoder_strides'] = [4, 2, 1]
    config['latent_encoder_paddings'] = [0, 1, 1]
    config['latent_encoder_feature_dim'] = config['latent_emb_nbr_variables']
    # Latent Decoder: Broadcast Convolutional architecture:
    '''
    config['latent_decoder_arch'] = 'BroadcastCNN'
    config['latent_decoder_channels'] = [64, 32, 16]
    config['latent_decoder_kernels'] = [4, 4, 3]
    config['latent_decoder_strides'] = [4, 2, 1]
    config['latent_decoder_paddings'] = [0, 1, 1]
    config['latent_decoder_feature_dim'] = config['latent_emb_nbr_variables']
    '''
    return config


@pytest.fixture
def ppo_rnd_config_dict_ma():
    config = dict()
    config['nbr_frame_stacking'] = 4
    config['observation_resize_dim'] = 80
    config['discount'] = 0.999
    config['use_gae'] = True
    config['use_cuda'] = True
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.001
    config['gradient_clip'] = 0.0
    config['optimization_epochs'] = 4
    config['mini_batch_size'] = 1024
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['nbr_actor'] = 32
    config['horizon'] = 128
    config['phi_arch'] = 'CNN-GRU-RNN'#'CNN'#
    config['actor_arch'] = 'MLP'
    config['critic_arch'] = 'MLP'
    
    # Phi Body: Convolutional Architecture:
    config['phi_arch_channels'] = [32, 64, 64]
    config['phi_arch_kernels'] = [8, 4, 3]
    config['phi_arch_strides'] = [4, 2, 1]
    config['phi_arch_paddings'] = [0, 1, 1]
    config['phi_arch_feature_dim'] = 256
    config['phi_arch_hidden_units'] = (256,)
    

    # Random Network Distillation:
    config['use_random_network_distillation'] = True
    config['intrinsic_discount'] = 0.99
    config['rnd_loss_int_ratio'] = 0.5
    config['rnd_obs_clip'] = 5
    config['rnd_non_episodic_int_r'] = True
    config['rnd_update_period_running_meanstd_int_reward'] = 1e5
    config['rnd_update_period_running_meanstd_obs'] = config['rnd_update_period_running_meanstd_int_reward']
    # RND Convolutional Architecture:
    config['rnd_arch'] = 'CNN'
    config['rnd_arch_channels'] = [32, 64, 64]
    config['rnd_arch_kernels'] = [8, 4, 3]
    config['rnd_arch_strides'] = [4, 2, 1]
    config['rnd_arch_paddings'] = [0, 1, 1]
    config['rnd_arch_feature_dim'] = 512
    # RND Fully-Connected Architecture:
    '''
    config['rnd_feature_net_fc_arch_hidden_units'] = (128, 64)
    '''
    return config


@pytest.fixture
def otc_task():
    import os
    here = os.path.abspath(os.path.dirname(__file__))
    return parse_environment(os.path.join(here, 'ObstacleTower/obstacletower'))

@pytest.fixture
def si_task():
    import os
    return parse_environment('SpaceInvaders-v0')
