import os
import sys
import math
sys.path.append(os.path.abspath('..'))

from rl_algorithms import AgentHook


class Menagerie(object):
    def __init__(self, subfolder_splits=[1e5, 1e4, 1e3], selfplay_scheme_name=None, training_agent_name=None, menagerie_path=None):
        self.menagerie = []
        self.subfolder_splits = subfolder_splits
        self.selfplay_scheme_name = selfplay_scheme_name
        self.training_agent_name = training_agent_name
        self.menagerie_path = menagerie_path
        
    def _generate_agent_menagerie_path(self):
        assert( self.menagerie_path is not None)
        assert( self.selfplay_scheme_name is not None)
        assert( self.training_agent_name is not None)
        agent_menagerie_path = '{}/{}-{}'.format(self.menagerie_path, self.selfplay_scheme_name, self.training_agent_name)
        if not os.path.exists(self.menagerie_path):
            os.mkdir(self.menagerie_path)
        if not os.path.exists(agent_menagerie_path):
            os.mkdir(agent_menagerie_path)
        return agent_menagerie_path

    def _generate_save_dir(self, iteration):
        floor_ranges = [ int(iteration // int(split) ) for split in self.subfolder_splits]
        power10s = [ int(math.log10(split)) for split in self.subfolder_splits]
        subfolder_paths = ["{}-{}e{}".format( floor_range, floor_range+1, power10) for floor_range, power10 in zip(floor_ranges,power10s) ]
        subfolder_path = ""
        for subpath in subfolder_paths:
            subfolder_path = os.path.join(subfolder_path, subpath)
        save_dir = subfolder_path
        return save_dir

    def set_selfplay_scheme_name(self, name):
        self.selfplay_scheme_name = name 

    def set_training_agent_name(self, name):
        self.training_agent_name = name 
    
    def set_menagerie_path(self, path):
        self.menagerie_path = path

    def __getitem__(self, key):
        if isinstance( key, slice ):
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance( key, int ):
            #Handle negative indices
            if key < 0 : 
                key += len( self )
            if key < 0 or key >= len( self ) :
                raise IndexError("The index (%d) is out of range."%key)
            return self.menagerie[key] 
        else:
            raise TypeError("Invalid argument type.")

    def __call__(self, agent, iteration):
        agent_menagerie_path = self._generate_agent_menagerie_path()
        save_dir = self._generate_save_dir(iteration)
        save_path = os.path.join(agent_menagerie_path, save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        agent_save_path = f'{save_path}/{iteration}_iterations.pt'
        agent_hook = AgentHook(agent.clone(training=False), save_path=agent_save_path)
        self.menagerie.append( agent_hook)

        return self

    def __len__(self):
        return len(self.menagerie)
