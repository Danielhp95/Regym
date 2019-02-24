from .gym_rock_paper_scissors_agent import MixedStrategyAgent
from .tabular_q_learning_agent import build_TabularQ_Agent, TabularQLearningAgent
from .deep_q_network_agent import build_DQN_Agent, DeepQNetworkAgent
from .deep_deterministic_policy_gradient_agent import build_DDPG_Agent, DDPGAgent
from .ppo_agent import build_PPO_Agent

rockAgent     = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent')
paperAgent    = MixedStrategyAgent(support_vector=[0, 1, 0], name='PaperAgent')
scissorsAgent = MixedStrategyAgent(support_vector=[0, 0, 1], name='ScissorsAgent')
randomAgent   = MixedStrategyAgent(support_vector=[1/3, 1/3, 1/3], name='RandomAgent')
