import os
from typing import Callable, Dict, List
from torch import multiprocessing
from copy import deepcopy
import torch

from regym.networks.preprocessing import batch_vector_observation


class NeuralNetServerHandler:

    def __init__(self, num_connections: int,
                 net: torch.nn.Module,
                 pre_processing_fn: Callable = batch_vector_observation,
                 device: str = 'cpu',
                 niceness: int = -5):
        '''
        NOTE: net will be deepcopied

        :param num_connections: TODO
        :param net: TODO
        :param pre_processing_fn: TODO
        :param device: TODO
        :param server_niceness: TODO (document -20:+19 range)
        '''
        self.num_connections = num_connections
        self.device = device
        self.pre_processing_fn = pre_processing_fn

        self.server_connections, self.client_connections = [], []
        for _ in range(num_connections):
            server_connection, client_connection = multiprocessing.Pipe()
            self.server_connections.append(server_connection)
            self.client_connections.append(client_connection)

        # Required, otherwise forking method won't work
        net.share_memory()

        self.server = multiprocessing.Process(
                target=neural_net_server,
                args=(deepcopy(net), self.server_connections,
                      self.pre_processing_fn, self.device, niceness),
                name='neural_network_server',
                daemon=True)  # We want the server to terminate
                              # when the main script terminates
        self.server.start()


    def update_neural_net(self, net):
        # Delete server, create new server with a deepcopy of :param: net
        self.server.terminate()
        self.server = multiprocessing.Process(
                target=neural_net_server,
                args=(deepcopy(net), self.server_connections,
                      self.pre_processing_fn, self.device))
        self.server.start()

    def close_server(self):
        self.server.terminate()

    # TODO: __repr__ function


def neural_net_server(net: torch.nn.Module,
                      connections: List[multiprocessing.Pipe],
                      pre_processing_fn: Callable = batch_vector_observation,
                      device: str = 'cpu',
                      niceness: int = -5):
    """
    Server style function which continuously polls :params: parent_connections
    for observations (inputs) to be fed to torch.nn.Module :param: net.
    The neural net will be loaded onto :param: device (i.e, cpu, gpu:0)

    ASSUMPTION: Requests want individual observations, never batches.

    NOTE: The :param: connections expect input of the form:
    Tuple[Any, List[int]], a tuple of (observation, legal_actions).
    Currently this server DOES NOT handle other input gracefully.

    :param net: TODO
    :param pre_processing_fn: TODO
    :param connections: TODO
    :param device: TODO
    :param niceness: TODO
    """
    # Sets process niceness to :param: niceness.
    #parent_niceness = os.nice(0)
    #os.nice(niceness - parent_niceness)

    net.to(device)
    pipes_to_serve, observations, legal_actions = [], [], []
    while True:
        for conn in connections:
            if conn.poll():
                pipes_to_serve.append(conn)
                request = conn.recv()
                observations.append(request[0])
                legal_actions.append(request[1])
        if observations:
            pre_processed_obs = pre_processing_fn(observations)
            prediction = net(pre_processed_obs, legal_actions=legal_actions)

            responses = _generate_responses(len(pipes_to_serve), prediction)

            _send_responses(pipes_to_serve, responses)

            pipes_to_serve.clear()
            observations.clear()
            legal_actions.clear()


def _generate_responses(num_pipes_to_serve: int,
                        prediction: Dict[str, torch.Tensor]) \
                        -> List[Dict[str, torch.Tensor]]:
    '''Generates a response for each pipe that sent a request to the server'''
    responses = [{} for _ in range(num_pipes_to_serve)]
    for k, v in prediction.items():
        for i in range(num_pipes_to_serve):
            responses[i][k] = v[i].cpu().detach()
    return responses


def _send_responses(pipes_to_serve: List[multiprocessing.Pipe],
                    responses: List[Dict[str, torch.Tensor]]):
    for i in range(len(pipes_to_serve)): pipes_to_serve[i].send(responses[i])
