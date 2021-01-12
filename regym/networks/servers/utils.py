from typing import List
from multiprocessing.connection import Connection

import numpy as np


def request_prediction_from_server(observation,
                                   legal_actions: List[int],
                                   connection: Connection,
                                   key: str,
                                   *args, **kwargs) -> np.ndarray:
    r'''
    Requests :param: key from a prediction from a NeuralNetServerHandler
    via :param: connection. Giving as input :param: observation
    and :param: legal_actions.

    Ultimately the final available keys that can be parsed from the
    prediction can be found in the class of the neural net hosted in the
    server. Normal keys:
        - 'probs': \pi(. | observation). Shape (n, )
        - 'V': State value prediction. V(observation). shape (1,)
        - 'policy_0': \pi_{opponent}(. | observation).
                      Prediction of opponent policy. Shape: (n, )
    '''
    connection.send((observation, legal_actions))
    prediction = connection.recv()
    if key not in prediction.keys():
        raise KeyError(f'Tried to access key \'{key}\' on a prediction '
                       f'requested from server with keys {prediction.keys()}.')
    return prediction[key].squeeze(0).numpy()
