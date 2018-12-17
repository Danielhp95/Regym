from collections import namedtuple
from functools import partial
import random

from . import naive_self_play as naive
from . import delta_distributional_self_play as delta_dis


SelfPlayTrainingScheme = namedtuple('SelfPlayTrainingScheme', 'opponent_sampling_distribution curator name')
NaiveSelfPlay               = SelfPlayTrainingScheme(naive.opponent_sampling_distribution,
                                                     naive.curator, 'NaiveSP')
DeltaDistributionalSelfPlay = SelfPlayTrainingScheme(delta_dis.opponent_sampling_distribution,
                                                     delta_dis.curator, 'DeltaDistributionalSP')

DeltaUniformSelfPlay = SelfPlayTrainingScheme(partial(delta_dis.opponent_sampling_distribution, distribution=random.choice),
                                              delta_dis.curator, 'DeltaUniformSelfPlay')

FullHistorySelfPlay = SelfPlayTrainingScheme(partial(delta_dis.opponent_sampling_distribution, delta=0.0, distribution=random.choice),
                                             delta_dis.curator, 'FullHistorySP')

HalfHistorySelfPlay = SelfPlayTrainingScheme(partial(delta_dis.opponent_sampling_distribution, delta=0.5, distribution=random.choice),
                                             delta_dis.curator, 'HalfHistorySP')

EmptySelfPlay = SelfPlayTrainingScheme(opponent_sampling_distribution=None, curator=None, name='EmptySelfPlay')
