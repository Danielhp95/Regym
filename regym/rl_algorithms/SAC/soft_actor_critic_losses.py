import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

summary_writer = None


def compute_q_critic_loss(o, a, r, o2, not_done,
                          q1, q2, q1_targ, q2_targ, pi,
                          alpha: float, gamma: float,
                          iteration_count: int) -> torch.Tensor:
    ''' TODO '''
    q1_pred = q1(o)
    q2_pred = q2(o)
    q1_a = q1_pred['Q'].gather(dim=1, index=a.unsqueeze(1))
    q2_a = q2_pred['Q'].gather(dim=1, index=a.unsqueeze(1))

    soft_bellman_backup = compute_soft_bellman_backup(
            r, o2, not_done, q1_targ, q2_targ, pi, alpha, gamma)

    # MSE loss against bellman backup
    loss_q1 = ((q1_a - soft_bellman_backup)**2).mean()
    loss_q2 = ((q2_a - soft_bellman_backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    if summary_writer:
        summary_writer.add_scalar('Training/Paper_Q1_value', q1_pred['Q'][1].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/Mean_Q1_values', q1_pred['Q'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/Mean_Q2_values', q2_pred['Q'].cpu().mean().item(), iteration_count)

        summary_writer.add_scalar('Training/Loss_Q_values', loss_q.item(), iteration_count)
        summary_writer.add_scalar('Training/Loss_Q1', loss_q1.item(), iteration_count)
        summary_writer.add_scalar('Training/Loss_Q2', loss_q2.item(), iteration_count)
    return loss_q


def compute_soft_bellman_backup(r, o2, not_done: torch.Tensor,
                                q1_targ, q2_targ, pi: nn.Module,
                                alpha: float , gamma: float) -> torch.Tensor:
    backup: torch.Tensor
    with torch.no_grad():
        # Target actions from *current* polocy
        pi_pred = pi(o2)
        a2 = pi_pred['a']
        log_pi_a2 = pi_pred['probs'].log().gather(dim=1, index=a2)

        q1_pi_targ_debug = q1_targ(o2)['Q']
        q2_pi_targ_debug = q2_targ(o2)['Q']
        q1_pi_targ = q1_targ(o2)['Q'].gather(dim=1, index=a2)
        q2_pi_targ = q2_targ(o2)['Q'].gather(dim=1, index=a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

        # debug
        # print('Mean policy: ', pi_pred['probs'].mean(dim=0), 'Mean Q: ', ((q1_pi_targ_debug + q2_pi_targ_debug) / 2).mean(dim=0))
        backup = r + gamma * not_done * (q_pi_targ - alpha * log_pi_a2)
    return backup


def compute_pi_actor_loss(o: torch.Tensor,
                          q1: nn.Module, q2: nn.Module,
                          pi: nn.Module,
                          alpha: float,
                          iteration_count: int) -> torch.Tensor:
    ''' TODO '''
    pi_pred = pi(o)
    q1_pred = q1(o)
    q2_pred = q2(o)

    a = pi_pred['a']
    log_pi = pi_pred['log_probs'].gather(dim=1, index=a)

    q1_pi = q1_pred['Q'].gather(dim=1, index=a)
    q2_pi = q2_pred['Q'].gather(dim=1, index=a)
    q_pi = torch.min(q1_pi, q2_pi)

    loss_pi = (alpha * log_pi - q_pi).mean()
    if summary_writer:
        summary_writer.add_scalar('Training/Loss_policy', loss_pi.mean().item(), iteration_count)
    return loss_pi
