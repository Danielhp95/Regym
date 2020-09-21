from typing import Callable, Dict, List, Tuple, Any
import os
from copy import deepcopy
import math
import select
from functools import partial

from torch import multiprocessing
import torch
import textwrap

from regym.networks.preprocessing import batch_vector_observation


class NeuralNetServerHandler:
    r'''
    A NeuralNetServerHandler is a wrapper around a Process which holds a
    neural network that processes requests by:
        - Gathering input observations from incoming connections:
        - Pre-processes input observations and puts them into a single batch
        - Batch is forward-passed through neural network
        - Each datapoint in output batch is sent back through the connection
          that requested it.

    This approach centralizes the use of a neural network, instead of making
    each process use it's own neural net. This is useful for algorithms like
    Expert Iteration, where multiple MCTS processes require to processes
    forward passes from the same neural network (to compute action priors
    during the expansion phase).

    NOTE: The underlying server Process is started upon initialization, but has to
    be closed either manually or by exiting the python interpreter that
    generated the NeuralNetServerHandler (python finishing its run).

    Usage:
    1 - Creation
    A neural net needs to be specified, alongside the number
    of connections exposed by the server, connections are multiprocessing.Pipe
    objects.
    >>> net: nn.Module  # Existing torch based neural network
    >>> server_handler = NeuralNetServer(num_connections=2, net=net)
    >>> server_handler.client_connections  # type: list, lenght: 2

    2 - Sending requests
    Sending a request involves sending a 2-element tuple
    (observations, legal_actions), legal_actions CAN be `None`.
    Send this tuple through a connection using the aforementioned
    multiprocessing.Pipe interface.
    >>> legal_actions: List[int]   # Available actions at a given env state
    >>> observation: torch.Tensor  # Environment state / observation
    >>> server_handler.client_connections[0].send((observation, legal_actions)

    3 - Receiving requests
    The server will receive the request and process it, depositing the result
    in the same connection (Pipe) that requested it.
    >>> processed_request = server_handler.client_connections[0].recv()

    4 - Close server
    >>> server_handler.close_server()
    '''

    def __init__(self, num_connections: int,
                 net: torch.nn.Module,
                 pre_processing_fn: Callable = batch_vector_observation,
                 device: str = 'cpu',
                 max_requests: float = math.inf):


        '''
        NOTE: net will be deepcopied

        :param num_connections: Number of multiprocessing.Pipe objects to be
                                created.
        :param net: Neural network that will process incoming tensors
        :param pre_processing_fn: Function that will be ued on incoming tensors,
                                  its output will be fed onto :param: net.
        :param device: Device where :param: net will be mounted unto.
        :param max_requests: Maximum number of requests handled by the server.
                             Meant for debugging purposes.
        '''
        self.num_connections = num_connections
        self.device = device
        self.pre_processing_fn = pre_processing_fn
        self.net_representation = str(net)
        self.max_requests = max_requests

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
                      self.pre_processing_fn, self.device, max_requests),
                name='neural_network_server',
                daemon=True)  # We want the server to terminate
                              # when the main script terminates
        self.server.start()

    def update_neural_net(self, net):
        '''
        Destroys the previous server and creates a new one with :param: net,
        effectively swapping the existing neural net with :param: net
        '''
        self.server.terminate()
        self.server = multiprocessing.Process(
                target=neural_net_server,
                args=(deepcopy(net), self.server_connections,
                      self.pre_processing_fn, self.device))
        self.server.start()

    def close_server(self):
        ''' Terminates the underlying Process containing the neural net '''
        self.server.terminate()

    def __repr__(self):
        server_info = f"Connections: {self.num_connections}\tDevice: {self.device}\tprepocessing_fn: {self.pre_processing_fn}"
        net_str = f"{textwrap.indent(self.net_representation, '    ')}"
        return f"NeuralNetServer:\n" + server_info + "\n" + net_str


def neural_net_server(net: torch.nn.Module,
                      connections: List[multiprocessing.Pipe],
                      pre_processing_fn: Callable = batch_vector_observation,
                      device: str = 'cpu',
                      max_requests: float = math.inf):
    """
    Server style function which continuously polls :param: parent_connections
    for observations (inputs) to be fed to torch.nn.Module :param: net.
    The neural net will be loaded onto :param: device (i.e, cpu, gpu:0).
    The :param: connections are polled for incoming observations, and
    together as a list they are processed via :param: pre_processing_fn.

    ASSUMPTION: Requests want individual observations, never batches.

    NOTE: The :param: connections expect input of the form:
    Tuple[Any, List[int]], a tuple of (observation, legal_actions).
    Currently this server DOES NOT handle other input gracefully.

    :param net: Neural network that will process incoming tensors
    :param pre_processing_fn: Function that will be ued on incoming tensors,
                              its output will be fed onto :param: net.
    :param connections: List of multiprocessing.Pipes from where input to
                        :param: net will come from. Results from :param: net
                        will also be returned through these connections.
    :param device: Device where :param: net will be mounted unto.
    :param max_requests: Maximum number of requests handled by the server.
                         Meant for debugging purposes.
    """
    net.to(device)
    processed_requests = 0

    poll_funcs = _create_polling_functions(connections)

    while True and (processed_requests < max_requests):
        (observations, legal_actions,
        connections_to_serve) = _poll_connections(connections, poll_funcs)
        if connections_to_serve:
            processed_requests += len(observations)
            pre_processed_obs = pre_processing_fn(observations).to(device)
            prediction = net(pre_processed_obs, legal_actions=legal_actions)

            responses = _generate_responses(len(connections_to_serve), prediction)

            _send_responses(connections_to_serve, responses)


def _poll_connections(connections: List[multiprocessing.Pipe],
                      poll_funcs: List[Callable]) \
                      -> Tuple[Any, Any, List[multiprocessing.Pipe]]:
    '''
    Gathers a set of requests of shape (observation, legal_actions) by
    polling all :param: connections via :param: poll_funcs.

    :returns: List of requested (observations) and corresponding (legal_actions)
              together with the corresponding connections which asked for them.
    '''
    observations, legal_actions, connections_to_serve = [], [], []
    for i in range(len(connections)):
        if poll_funcs[i](0):  # 0 corresponds to timeout
            connections_to_serve.append(connections[i])
            request = connections[i].recv()
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
    ''' Sends :param: responses to correspondig :param: pipes_to_serve '''
    for i in range(len(pipes_to_serve)): pipes_to_serve[i].send(responses[i])


def _create_polling_functions(connections: List[multiprocessing.Pipe])\
                              -> List[Callable]:
    '''
    Still the bottleneck D:. But way faster than before!

    The method multiprocessing.Pipe.poll() is prohibitevely slow. To the point
    that polling Pipes were the bottleneck of the neuralnet server. Instead of
    using the multiprocessing.Pipe.poll(), we use the lower-level, OS specified
    polling() method exposed from the file descriptor attached to each Pipe in
    :param: connections.
    For extra info:
    https://docs.python.org/3/library/select.html?highlight=select#polling-objects

    :param connections: List of connections from which the neural net server
                        will receive processing requests
    :returns: List of polling functions that (quickly) check if there's pending
              I/O on their corresponding pipes
    '''
    poll_funcs = []
    for conn in connections:
        pollin = select.poll()
        pollin.register(conn, select.POLLIN)
        poll_funcs.append(pollin.poll)
    return poll_funcs
