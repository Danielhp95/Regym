from typing import List, Dict, Any, Union, Callable
from functools import partial
import numpy as np
from copy import deepcopy


def apply_data_augmentation_to_experiences(experiences: List, env_ids: List[int], data_augmnentation_fn: Callable) -> List:
    '''
    Augments every experience in :param: experience according to :param: data_augmnentation_fn.

    TODO: explain hack: We are first putting augmented data and then experiences
    NOTE: There are some algorithms which trigger a specific effect upon handling
    a terminal experience (done flag set to True).

    :params:
        - experiences: List of experiences to be augmented
    :returns:
        - Augmented List of experiences
    '''
    augmented_data, augmented_env_ids = [], []
    for (o, a, r, succ_o, done, extra_info), env_i in zip(experiences, env_ids):
        augmented_exps = data_augmnentation_fn(o, a, r, succ_o, done, extra_info)  # This could be used for any other function
        # Because more than one datapoint might be added
        for new_exp in augmented_exps:
            augmented_data += [new_exp]
            augmented_env_ids += [env_i]
    return (augmented_data + experiences, augmented_env_ids + env_ids)


def generate_horizontal_symmetry(o, a, r, succ_o, done, extra_info: Dict[str, Any],
                                 flip_obs_on_dim: int=0) -> List:
    '''
    Data augmentation which horizontally flips observations :param: o and
    :param: succ_o. Elements in dictionary :param: extra_info are also flipped
    accordingly.

    NOTE! This is currently kinda hardcoded to be tailored for ExpertIterationAgent

    :params:
        - o: Successor observation, will be flipped
        - a: Action. TODO: add changes
        - r: Reward. Unchanged by this symmetry.
        - succ_o: Successor observation, will be flipped.
        - done: Whether the episode has finished or not
        - extra_info: Dictionary containing the current predictions for all agents
        - flip_obs_on_dim: Dimension over which to flip a dimension.
                           Depending on the format of the observation
                           this might be 0 (i.e vector observations)
                           or 1 (observations with different channels)
    '''
    sym_o = np.flip(o, flip_obs_on_dim)
    sym_succ_o = np.flip(succ_o, flip_obs_on_dim)
    # TODO: Action should _technically_ be flipped too. For this, we should have
    # access to the environment's action space. Which currently feels more effort than it's worth
    sym_a = a
    sym_r = r  # Symmetry does not influence reward
    sym_done = False
    sym_extra_info = deepcopy(extra_info)  # Maybe this operation is too heavy
    for key in extra_info.keys():  # Go through all of the agents
        if 'child_visitations' in extra_info[key]:
            # The value in `child_visitation` is a torch.Tensor
            sym_extra_info[key]['child_visitations'] = extra_info[key]['child_visitations'].flip(0)
        if 'probs' in extra_info[key]: sym_extra_info[key]['probs'] = np.flip(extra_info[key]['probs'])
        if 's' in extra_info[key]: sym_extra_info[key]['s'] = np.flip(extra_info[key]['s'], flip_obs_on_dim)
    return [(sym_o, sym_a, sym_r, sym_succ_o, sym_done, sym_extra_info)]


def _parse_data_augmentation_fn_from_string(fn_name: str) -> Callable:
    if fn_name == 'generate_horizontal_symmetry': return generate_horizontal_symmetry
    else:
        raise ValueError('Could not parse data augmentation function '
                         f'from name {fn_name}')

def parse_data_augmentation_fn(fn_name_or_dict: Union[str, Dict[str, Any]]) -> Callable:
    '''
    Parses :param: fn_name to see if there is a data augmentation function
    with the same name. :param: fn_name_or_dict can either be a function name
    that can be processed from this function or a Dictionary, with one entry being
    "name: <name of function>" and the rest of elements being parameters to partially
    apply to the "name" function.

    TODO: better error handling / messaging
    '''
    if isinstance(fn_name_or_dict, str): return _parse_data_augmentation_fn_from_string(fn_name_or_dict)
    elif isinstance(fn_name_or_dict, Dict):
        function = _parse_data_augmentation_fn_from_string(fn_name_or_dict['name'])
        fn_name_or_dict.pop('name')
        return partial(function, **fn_name_or_dict)
    else:
        raise ValueError('Could not parse data augmentation function '
                         f'from input parameter: {fn_name_or_dict}')
