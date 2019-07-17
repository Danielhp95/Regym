from collections import namedtuple
from functools import partial
import numpy as np

from . import naive_self_play as naive
from . import delta_distributional_self_play as delta_dis
from . import delta_limit_uniform_distributional_self_play as delta_limit_dis


SelfPlayTrainingScheme = namedtuple('SelfPlayTrainingScheme', 'opponent_sampling_distribution curator name')
NaiveSelfPlay               = SelfPlayTrainingScheme(naive.opponent_sampling_distribution,
                                                     naive.curator, 'NaiveSP')

DeltaDistributionalSelfPlay = SelfPlayTrainingScheme(delta_dis.opponent_sampling_distribution,
                                                     delta_dis.curator, 'DeltaDistributionalSP')

DeltaUniformSelfPlay = SelfPlayTrainingScheme(partial(delta_dis.opponent_sampling_distribution, distribution=np.random.choice),
                                              delta_dis.curator, 'DeltaUniformSP')

FullHistorySelfPlay = SelfPlayTrainingScheme(partial(delta_dis.opponent_sampling_distribution, delta=0.0, distribution=np.random.choice),
                                             delta_dis.curator, 'FullHistorySP')

HalfHistorySelfPlay = SelfPlayTrainingScheme(partial(delta_dis.opponent_sampling_distribution, delta=0.5, distribution=np.random.choice),
                                             delta_dis.curator, 'HalfHistorySP')

LastQuarterHistorySelfPlay = SelfPlayTrainingScheme(partial(delta_dis.opponent_sampling_distribution, delta=0.75, distribution=np.random.choice),
                                                    delta_dis.curator, 'LastQuarterHistorySP')

DeltaLimitUniformSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, distribution=np.random.choice),
                                              delta_limit_dis.curator, 'DeltaLimitUniformSP')

FullHistoryLimitSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, delta=0.0, distribution=np.random.choice),
                                             delta_limit_dis.curator, 'FullHistoryLimitSP')

HalfHistoryLimitSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, delta=0.5, distribution=np.random.choice),
                                             delta_limit_dis.curator, 'HalfHistoryLimitSP')

LastQuarterHistoryLimitSelfPlay = SelfPlayTrainingScheme(partial(delta_limit_dis.opponent_sampling_distribution, delta=0.75, distribution=np.random.choice),
                                                    delta_limit_dis.curator, 'LastQuarterHistoryLimitSP')


EmptySelfPlay = SelfPlayTrainingScheme(opponent_sampling_distribution=None, curator=None, name='EmptySelfPlay')

