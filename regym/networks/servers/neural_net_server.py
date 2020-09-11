from typing import Callable, Dict, List, Tuple
import os
from copy import deepcopy
import math

from torch import multiprocessing
import torch
import textwrap

from regym.networks.preprocessing import batch_vector_observation


class NeuralNetServerHandler:

    def __init__(self, num_connections: int,
                 net: torch.nn.Module,
                 pre_processing_fn: Callable = batch_vector_observation,
                 device: str = 'cpu',
                 niceness: int = -5,
                 max_requests: float = math.inf):
        '''
        NOTE: net will be deepcopied

        :param num_connections: TODO
        :param net: TODO
        :param pre_processing_fn: TODO
        :param device: TODO
        :param server_niceness: TODO (document -20:+19 range)
        :param max_requests: Maximum number of requests handled by the server.
                             Meant for debugging purposes.
        '''
        self.num_connections = num_connections
        self.device = device
        self.pre_processing_fn = pre_processing_fn
        self.net_representation = str(net)

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

    def __repr__(self):
        server_info = f"Connections: {self.num_connections}\tDevice: {self.device}\tprepocessing_fn: {self.pre_processing_fn}"
        net_str = f"{textwrap.indent(self.net_representation, '    ')}"
        return f"NeuralNetServer:\n" + server_info + "\n" + net_str


def neural_net_server(net: torch.nn.Module,
                      connections: List[multiprocessing.Pipe],
                      pre_processing_fn: Callable = batch_vector_observation,
                      device: str = 'cpu',
                      niceness: int = -5,
                      max_requests: float = math.inf):
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
    :param max_requests: Maximum number of requests handled by the server.
                         Meant for debugging purposes.
    """
    # Sets process niceness to :param: niceness.
    #parent_niceness = os.nice(0)
    #os.nice(niceness - parent_niceness)

    net.to(device)
    processed_requests = 0
    while True or processed_requests >= max_requests:
        (observations, legal_actions,
        connections_to_serve) = _poll_connections(connections)
        if observations:
            processed_requests += len(observations)
            pre_processed_obs = pre_processing_fn(observations).to(device)
            prediction = net(pre_processed_obs, legal_actions=legal_actions)

            responses = _generate_responses(len(connections_to_serve), prediction)

            _send_responses(connections_to_serve, responses)


def _poll_connections(connections) -> Tuple:
    observations, legal_actions, connections_to_serve = [], [], []
    for conn in connections:
        if conn.poll():
            connections_to_serve.append(conn)
            request = conn.recv()
            observations.append(request[0])
            legal_actions.append(request[1])
    return observations, legal_actions, connections_to_serve


def _generate_responses(num_pipes_to_serve: int,
                        prediction: Dict[str, torch.Tensor]) \
                        -> List[Dict[str, torch.Tensor]]:
    '''Generates a response for each pipe that sent a request to the server'''
    responses = [{} for _ in range(num_pipes_to_serve)]
    for k, v in prediction.items():
        for i in range(num_pipes_to_serve):
            # TODO: figure out if we really need to put predictions on cpu
            # might be better to leave them in the original devicw for loss calculations

            # Detaching is necessary for torch.Tensor(s) to be sent over processes
            responses[i][k] = v[i].detach()  # .cpu()
    return responses


def _send_responses(pipes_to_serve: List[multiprocessing.Pipe],
                    responses: List[Dict[str, torch.Tensor]]):
    for i in range(len(pipes_to_serve)): pipes_to_serve[i].send(responses[i])
