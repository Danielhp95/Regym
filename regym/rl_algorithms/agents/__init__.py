from .agent import Agent

# Useful basic agents
from .deterministic_agent import build_Deterministic_Agent, DeterministicAgent
from .mixed_strategy_agent import MixedStrategyAgent
from .random_agent import build_Random_Agent, RandomAgent
from .human_agent import build_Human_Agent, HumanAgent
from .neural_net_agent import build_NeuralNet_Agent, NeuralNetAgent

# Tabular agents
from .tabular_q_learning_agent import build_TabularQ_Agent, TabularQLearningAgent

# DQN based agents
from .deep_q_network_agent import build_DQN_Agent, DeepQNetworkAgent

# Policy gradient agents
from .ppo_agent import build_PPO_Agent, PPOAgent
from .reinforce_agent import build_Reinforce_Agent, ReinforceAgent
from .a2c_agent import build_A2C_Agent, A2CAgent

# SAC agents
from .soft_actor_critic_agent import build_SAC_Agent, SoftActorCriticAgent

# Search based agents
from .mcts_agent import build_MCTS_Agent, MCTSAgent
from .expert_iteration_agent import build_ExpertIteration_Agent, ExpertIterationAgent

rockAgent     = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent')
paperAgent    = MixedStrategyAgent(support_vector=[0, 1, 0], name='PaperAgent')
scissorsAgent = MixedStrategyAgent(support_vector=[0, 0, 1], name='ScissorsAgent')
randomAgent   = MixedStrategyAgent(support_vector=[1/3, 1/3, 1/3], name='RandomAgent')
