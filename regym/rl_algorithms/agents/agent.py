from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, name):
        self.requires_environment_model = False
        self.name = name
        self.training = True
        self.handled_experiences = 0

    @abstractmethod
    def take_action(self, state):
        pass

    @abstractmethod
    def handle_experience(self, s, a, r, succ_s, done=False):
        self.handled_experiences += 1

    @abstractmethod
    def clone(self):
        pass
