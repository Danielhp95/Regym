from collections import namedtuple

EXP = namedtuple('EXP', ('state','action','next_state', 'reward','done') )
EXPPER = namedtuple('EXPPER', ('idx','priority','state','action','next_state', 'reward','done') )
