from collections import namedtuple
from functools import partial
import numpy as np

from . import naive_self_play as naive

from .delta_distributional_self_play import DeltaDistributionalSelfPlay
from .delta_limit_uniform_distributional_self_play import DeltaLimitDistributionalSelfPlay

from .psro import PSRONashResponse


SelfPlayTrainingScheme = namedtuple('SelfPlayTrainingScheme', 'opponent_sampling_distribution curator name')
NaiveSelfPlay          = SelfPlayTrainingScheme(naive.opponent_sampling_distribution,
                                                naive.curator, 'NaiveSP')
EmptySelfPlay = SelfPlayTrainingScheme(opponent_sampling_distribution=None, curator=None, name='EmptySelfPlay')
