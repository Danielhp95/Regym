from collections import namedtuple
from functools import partial
import numpy as np

from . import naive_self_play as naive
from . import delta_limit_uniform_distributional_self_play as delta_limit_dis

from .delta_distributional_self_play import DeltaDistributionalSelfPlay
from .psro import PSRONashResponse


SelfPlayTrainingScheme = namedtuple('SelfPlayTrainingScheme', 'opponent_sampling_distribution curator name')
NaiveSelfPlay               = SelfPlayTrainingScheme(naive.opponent_sampling_distribution,
                                                     naive.curator, 'NaiveSP')

DeltaLimitUniformSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, distribution=np.random.choice),
                                              delta_limit_dis.curator, 'DeltaLimitUniformSP')

FullHistoryLimitSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, delta=0.0, distribution=np.random.choice),
                                             delta_limit_dis.curator, 'FullHistoryLimitSP')

HalfHistoryLimitSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, delta=0.5, distribution=np.random.choice),
                                             delta_limit_dis.curator, 'HalfHistoryLimitSP')

LastQuarterHistoryLimitSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, delta=0.75, distribution=np.random.choice),
                                                    delta_limit_dis.curator, 'LastQuarterHistoryLimitSP')

EmptySelfPlay = SelfPlayTrainingScheme(opponent_sampling_distribution=None, curator=None, name='EmptySelfPlay')
