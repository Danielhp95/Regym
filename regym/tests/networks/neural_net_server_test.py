from typing import List
from copy import deepcopy
import multiprocessing
import time

import pytest
import numpy as np
import torch

from regym.networks.servers import neural_net_server
from regym.networks.servers.neural_net_server import NeuralNetServerHandler


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


#@pytest.mark.skipif(not torch.cuda.is_available(),
#                    reason="Requires a gpu and cuda to be available")
def test_server_is_faster_on_gpu():
    pr = cProfile.Profile()
    pr.enable()
    cpu_time = _test_server_speed(device='cpu')
    pr.disable()
    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    print(ps.print_stats())
    #if filename != '': ps.dump_stats(filename)
    #gpu_time = _test_server_speed(device='cpu')
    #assert gpu_time < cpu_time


def _test_server_speed(device, init_dim=32, num_connections=10,
                       num_requests=10):
    net = generate_timing_neural_net(dims=[init_dim,32,32,32])
    server_handler = NeuralNetServerHandler(num_connections=num_connections,
                                            net=net, device=device,
                                            max_requests=num_requests * num_connections)
    total_time = 0
    for _ in range(num_requests):
        for connection_i in range(num_connections):
            observation = torch.rand(size=(1, init_dim))
            server_handler.client_connections[connection_i].send((observation, None))
        responses = [server_handler.client_connections[connection_i].recv()
                     for connection_i in range(num_connections)]
        total_time += sum([x['time'] for x in responses])
    return total_time.item()


def generate_dummy_neural_net(weight):
    class DummyNet(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.linear = torch.nn.Linear(in_features=1, out_features=1, bias=False)
            self.linear.weight.data = torch.Tensor([[weight]])
        def forward(self, x, legal_actions=None):
            return {'output': self.linear(x)}
    return DummyNet(weight)


def generate_timing_neural_net(dims: List[int]):
    class TimingDummyNet(torch.nn.Module):
        def __init__(self, dims: List[int]):
            super().__init__()
            self.layers = torch.nn.Sequential(
                *[torch.nn.Linear(in_features=h_in, out_features=h_out, bias=True)
                 for h_in, h_out in zip(dims, dims[1:])])
        def forward(self, x, legal_actions=None):
            start = time.time()
            self.layers(x)
            total_time = time.time() - start
            return {'time': torch.Tensor([total_time] * x.shape[0])}
    return TimingDummyNet(dims)
