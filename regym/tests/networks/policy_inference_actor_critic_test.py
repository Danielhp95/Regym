import numpy as np
import torch
import torch.nn as nn

from regym.networks.bodies import FCBody

from regym.networks.generic_losses import cross_entropy_loss
from regym.networks.heads import PolicyInferenceActorCriticNet


def test_can_learn_three_different_policies():
    num_policies = 3 
    num_actions = 3
    target_policy_1 = torch.FloatTensor([1., 0., 0.])
    target_policy_2 = torch.FloatTensor([1./3, 1./3, 1./3])
    target_policy_3 = torch.FloatTensor([0., 0., 1.])

    #target_policy_1 = torch.FloatTensor([1./3, 1./3, 1./3])
    #target_policy_2 = torch.FloatTensor([1./3, 1./3, 1./3])
    #target_policy_3 = torch.FloatTensor([1./3, 1./3, 1./3])

    feature_extractor = FCBody(state_dim=3, hidden_units=(3,), gate=nn.functional.leaky_relu)
    policy_inference_body = FCBody(state_dim=3, hidden_units=(3,), gate=nn.functional.leaky_relu)
    actor_critic_body = FCBody(state_dim=3, hidden_units=(3,), gate=nn.functional.leaky_relu)

    model = PolicyInferenceActorCriticNet(
                 num_policies=num_policies,
                 num_actions=num_actions,
                 feature_extractor=feature_extractor,
                 policy_inference_body=policy_inference_body,
                 actor_critic_body=actor_critic_body)

    train_model(model, target_policy_1, target_policy_2, target_policy_3)
    _test_model(model, target_policy_1, target_policy_2, target_policy_3)


def train_model(model, target_policy_1, target_policy_2, target_policy_3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_tensor = torch.FloatTensor([[1., 1., 1.]])

    training_steps = 10000
    for i in range(training_steps):
        prediction = model(input_tensor)

        #if i > 400:
        #   for m in model.policy_inference_heads.modules():
        #       import ipdb; ipdb.set_trace()
        #       pass
        cross_entropy_loss_1, kl_div_1 = cross_entropy_loss(prediction['policy_0']['probs'], target_policy_1.unsqueeze(0))
        cross_entropy_loss_2, kl_div_2 = cross_entropy_loss(prediction['policy_1']['probs'], target_policy_2.unsqueeze(0))
        cross_entropy_loss_3, kl_div_3 = cross_entropy_loss(prediction['policy_2']['probs'], target_policy_3.unsqueeze(0))

        
        #print(i, '/', training_steps)
        #print('Loss 1: ', cross_entropy_loss_1)
        #print('Loss 2: ', cross_entropy_loss_2)
        #print('Pred  : ', prediction['policy_1']['probs'])
        #print('Loss 3: ', cross_entropy_loss_3)
        #print('')

        # Name a more iconic trio
        total_loss = cross_entropy_loss_1 + cross_entropy_loss_3 + cross_entropy_loss_2
        optimizer.zero_grad()
        from regym.util.nn_debugging import plot_gradient_flow
        total_loss.backward()
        #import ipdb; ipdb.set_trace()
        #plot_gradient_flow(model.policy_inference_heads.named_parameters())
        optimizer.step()


def _test_model(model, target_policy_1, target_policy_2, target_policy_3):
    input_tensor = torch.FloatTensor([[1., 1., 1.]])
    test_steps = 100
    for _ in range(test_steps):
        prediction = model(input_tensor)

        pred_1 = prediction['policy_0']['probs'].detach().squeeze(0).numpy()
        pred_2 = prediction['policy_1']['probs'].detach().squeeze(0).numpy()
        pred_3 = prediction['policy_2']['probs'].detach().squeeze(0).numpy()

        np.testing.assert_array_almost_equal(pred_1, target_policy_1.numpy(), decimal=3)
        np.testing.assert_array_almost_equal(pred_2, target_policy_2.numpy(), decimal=3)
        np.testing.assert_array_almost_equal(pred_3, target_policy_3.numpy(), decimal=3)
