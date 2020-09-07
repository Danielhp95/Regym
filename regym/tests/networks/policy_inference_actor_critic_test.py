import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from regym.networks.bodies import FCBody

from regym.networks.generic_losses import cross_entropy_loss
from regym.networks.heads import PolicyInferenceActorCriticNet


def test_can_learn_three_different_policies():
    num_policies = 2
    num_actions = 3
    target_policy_1 = torch.FloatTensor([[1., 0., 0.]])
    target_policy_2 = torch.FloatTensor([[1/3, 1/3, 1/3]])

    feature_extractor = FCBody(state_dim=3, hidden_units=(3,), gate=nn.functional.leaky_relu)
    policy_inference_body = FCBody(state_dim=3, hidden_units=(3,), gate=nn.functional.leaky_relu)
    actor_critic_body = FCBody(state_dim=3, hidden_units=(3,), gate=nn.functional.leaky_relu)

    model = PolicyInferenceActorCriticNet(
                 num_policies=num_policies,
                 num_actions=num_actions,
                 feature_extractor=feature_extractor,
                 policy_inference_body=policy_inference_body,
                 actor_critic_body=actor_critic_body)

    train_model(model, target_policy_1, target_policy_2)
    _test_model(model, target_policy_1, target_policy_2)


def train_model(model, target_policy_1, target_policy_2):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    training_steps = 5000
    progress_bar = tqdm(range(training_steps))
    for i in progress_bar:
        input_tensor = torch.rand(size=(1, 3))
        prediction = model(input_tensor)

        from torch.nn.functional import kl_div
        cross_entropy_loss_1 = -1. * kl_div(prediction['policy_0']['probs'], target_policy_1) #cross_entropy_loss(prediction['policy_0']['probs'], target_policy_1.unsqueeze(0))
        cross_entropy_loss_2 = -1. * kl_div(prediction['policy_1']['probs'], target_policy_2) #cross_entropy_loss(prediction['policy_1']['probs'], target_policy_2.unsqueeze(0))

        total_loss = cross_entropy_loss_1 + cross_entropy_loss_2
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        progress_bar.set_description(
            'L1: {}\tL2: {}'.format(
            cross_entropy_loss_1, cross_entropy_loss_2))


def _test_model(model, target_policy_1, target_policy_2):
    test_steps = 100
    for _ in range(test_steps):
        input_tensor = torch.rand(size=(1, 3))
        prediction = model(input_tensor)

        pred_1 = prediction['policy_0']['probs'].detach().numpy()
        pred_2 = prediction['policy_1']['probs'].detach().numpy()

        np.testing.assert_array_almost_equal(pred_1, target_policy_1.numpy(), decimal=1)
        np.testing.assert_array_almost_equal(pred_2, target_policy_2.numpy(), decimal=1)
