import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 3e-1


def init_weights(size):
    v = 1. / np.sqrt(size[0])
    return torch.Tensor(size).uniform_(-v, v)


def hard_update(fromm, to):
    for fp, tp in zip(fromm.parameters(), to.parameters()):
        fp.data.copy_(tp.data)


def soft_update(fromm, to, tau):
    for fp, tp in zip(fromm.parameters(), to.parameters()):
        fp.data.copy_((1.0-tau)*fp.data + tau*tp.data)


def LeakyReLU(x):
    return F.leaky_relu(x, 0.1)


class DQN(nn.Module) :
    def __init__(self,state_dim=3, nbr_actions=2,actfn=LeakyReLU, use_cuda=False ) :
        super(DQN,self).__init__()

        self.state_dim = state_dim
        self.nbr_actions = nbr_actions
        self.use_cuda = use_cuda

        self.actfn = actfn

        self.f1 = nn.Linear(self.state_dim, 1024)
        self.f2 = nn.Linear(1024, 256)
        self.f3 = nn.Linear(256,64)

        self.qsa = nn.Linear(64,self.nbr_actions)

        if self.use_cuda:
            self = self.cuda()


    def clone(self):
        cloned = DQN(state_dim=self.state_dim,nbr_actions=self.nbr_actions,actfn=self.actfn,use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned

    def forward(self, x) :
        x = self.actfn( self.f1(x) )
        x = self.actfn( self.f2(x) )
        fx = self.actfn( self.f3(x) )

        qsa = self.qsa(fx)

        return qsa



class DuelingDQN(nn.Module) :
    def __init__(self,state_dim=3, nbr_actions=2,actfn=LeakyReLU, use_cuda=False ) :
        super(DuelingDQN,self).__init__()

        self.state_dim = state_dim
        self.nbr_actions = nbr_actions
        self.use_cuda = use_cuda

        self.actfn = actfn
        self.f1 = nn.Linear(self.state_dim, 1024)
        self.f2 = nn.Linear(1024, 256)
        self.f3 = nn.Linear(256,64)

        self.value = nn.Linear(64,1)
        self.advantage = nn.Linear(64,self.nbr_actions)

        if self.use_cuda:
            self = self.cuda()

    def clone(self):
        cloned = DuelingDQN(state_dim=self.state_dim,nbr_actions=self.nbr_actions,actfn=self.actfn,use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned

    def forward(self, x) :
        x = self.actfn( self.f1(x) )
        x = self.actfn( self.f2(x) )
        fx = self.actfn( self.f3(x) )

        v = self.value(fx)
        va = torch.cat( [ v for i in range(self.nbr_actions) ], dim=1)
        a = self.advantage(fx)

        suma = torch.mean(a,dim=1,keepdim=True)
        suma = torch.cat( [ suma for i in range(self.nbr_actions) ], dim=1)

        x = va+a-suma

        return x


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
        unscaled = torch.tanh(xx)
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
