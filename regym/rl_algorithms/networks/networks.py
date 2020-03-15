import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 3e-1


def init_weights(size):
    v = 1. / np.sqrt(size[0])
    return torch.Tensor(size).uniform_(-v, v)


def LeakyReLU(x):
    return F.leaky_relu(x, 0.1)


class ActorNN(nn.Module) :
    def __init__(self,state_dim=3,action_dim=2,action_scaler=1.0,HER=False,actfn=LeakyReLU, use_cuda=False ) :
        super(ActorNN,self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scaler = action_scaler

        self.HER = HER
        if self.HER :
            self.state_dim *= 2

        self.actfn = actfn
        #Features :
        self.featx = nn.Linear(self.state_dim,400)
        self.featx.weight.data = init_weights(self.featx.weight.data.size())

        # Actor network :
        self.actor1 = nn.Linear(400,300)
        self.actor1.weight.data.uniform_(-EPS,EPS)
        self.actor2 = nn.Linear(300,self.action_dim)
        self.actor2.weight.data.uniform_(-EPS,EPS)

        self.use_cuda = use_cuda
        if self.use_cuda :
            self = self.cuda()


    def features(self,x) :
        fx = self.actfn( self.featx( x) )
        # batch x 400
        return fx

    def forward(self, x) :
        fx = self.features(x)
        # batch x 400
        out = self.actfn( self.actor1( fx ) )
        # batch x 300
        out = self.actor2( fx )
        # batch x self.action_dim

        #scale the actions :
        unscaled = F.tanh(xx)
        scaled = unscaled * self.action_scaler
        return scaled

    def clone(self):
        cloned = ActorNN(state_dim=self.state_dim,action_dim=self.action_dim,action_scaler=self.action_scaler,CNN=self.CNN,HER=self.HER,actfn=self.actfn, use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned


class CriticNN(nn.Module) :
    def __init__(self,state_dim=3,action_dim=2,HER=False,actfn=LeakyReLU, use_cuda=False  ) :
        super(CriticNN,self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.HER = HER
        if self.HER :
            self.state_dim *= 2

        self.actfn = actfn

        #Features :
        self.featx = nn.Linear(self.state_dim,400)
        self.featx.weight.data = init_weights(self.featx.weight.data.size())

        # Critic network :
        ## state value path :
        self.critic1 = nn.Linear(400+self.action_dim,300)
        self.critic1.weight.data = init_weights(self.critic1.weight.data.size())

        self.critic2 = nn.Linear(300,1)
        self.critic2.weight.data.uniform_(-EPS*1e-1,EPS*1e-1)

        self.use_cuda = use_cuda
        if self.use_cuda :
            self = self.cuda()

    def features(self,x) :
        fx = self.actfn( self.featx(x) )
        # batch x 400

        return fx

    def forward(self, x,a) :
        fx = self.features(x)
        # batch x 400
        concat = torch.cat([ fx, a], dim=1)
        # batch x 400+self.action_dim
        out = self.actfn( self.critic1( concat ) )
        # batch x 300
        out = self.critic2(out)
        # batch x 1
        return out

    def clone(self):
        cloned = CriticNN(state_dim=self.state_dim,action_dim=self.action_dim,CNN=self.CNN,HER=self.HER,actfn=self.actfn, use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned
