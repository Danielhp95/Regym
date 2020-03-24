from .agent import Agent

# Useful basic agents
from .random_agent import build_Random_Agent, RandomAgent
from .mixed_strategy_agent import MixedStrategyAgent

# Tabular agents
from .tabular_q_learning_agent import build_TabularQ_Agent, TabularQLearningAgent

# DQN based agents
from .deep_q_network_agent import build_DQN_Agent, DeepQNetworkAgent

# Policy gradient agents
from .ppo_agent import build_PPO_Agent, PPOAgent
from .reinforce_agent import build_Reinforce_Agent, ReinforceAgent
from .a2c_agent import build_A2C_Agent, A2CAgent

from .mcts_agent import build_MCTS_Agent, MCTSAgent

rockAgent     = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent')
paperAgent    = MixedStrategyAgent(support_vector=[0, 1, 0], name='PaperAgent')
scissorsAgent = MixedStrategyAgent(support_vector=[0, 0, 1], name='ScissorsAgent')
randomAgent   = MixedStrategyAgent(support_vector=[1/3, 1/3, 1/3], name='RandomAgent')
