from copy import deepcopy
import multiprocessing
import numpy as np
import torch

from regym.rl_algorithms.servers import neural_net_server
from regym.rl_algorithms.servers.neural_net_server import NeuralNetServerHandler


def test_can_process_single_observation():
    client_connection1, server_connection1 = multiprocessing.Pipe()
    net = generate_dummy_neural_net(weight=1.)

    server = multiprocessing.Process(
            target=neural_net_server,
            args=(deepcopy(net), [server_connection1]))

    observation_1 = np.array([0]) 
    client_connection1.send((observation_1, None))

    server.start()

    expected_response_1 = {'output': torch.Tensor([0])}
    assert expected_response_1 == client_connection1.recv()
    server.terminate()
    

def test_can_process_batch_observation_and_respond_individually():
    client_connection1, server_connection1 = multiprocessing.Pipe()
    client_connection2, server_connection2 = multiprocessing.Pipe()

    net = generate_dummy_neural_net(weight=1.)

    server = multiprocessing.Process(
            target=neural_net_server,
            args=(deepcopy(net), [server_connection1, server_connection2]))

    observation_1 = np.array([0]) 
    observation_2 = np.array([1]) 


    client_connection1.send((observation_1, None))
    client_connection2.send((observation_2, None))

    server.start()

    expected_response_1 = {'output': torch.Tensor([0])}
    expected_response_2 = {'output': torch.Tensor([1])}

    assert expected_response_1 == client_connection1.recv()
    assert expected_response_2 == client_connection2.recv()
    server.terminate()


def test_can_update_the_neural_net_in_the_server():
    net1 = generate_dummy_neural_net(weight=0.)
    net2 = generate_dummy_neural_net(weight=1.)

    observation = np.array([1])

    expected_response_1 = {'output': torch.Tensor([0])}
    expected_response_2 = {'output': torch.Tensor([1])}

    server_handler = NeuralNetServerHandler(num_connections=1,
                                            net=net1)

    server_handler.client_connections[0].send((observation, None))
    actual_response = server_handler.client_connections[0].recv()
    assert expected_response_1 == actual_response 

    server_handler.update_neural_net(net2)

    server_handler.client_connections[0].send((observation, None))
    actual_response = server_handler.client_connections[0].recv()
    assert expected_response_2 == actual_response 

    server_handler.close_server()


def generate_dummy_neural_net(weight):
    class DummyNet(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.linear = torch.nn.Linear(in_features=1, out_features=1, bias=False)
            self.linear.weight.data = torch.Tensor([[weight]])
        def forward(self, x, legal_actions=None):
            return {'output': self.linear(x)}

    return DummyNet(weight)
