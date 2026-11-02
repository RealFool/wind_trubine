from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Adam

from RL.models.actor_mlp import ActorMLP
from RL.models.critic_mlp import CriticMLP


@dataclass
class TD3Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class TD3Agent:
    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any], device: torch.device) -> None:
        self.device = device
        hidden_dims = config.get('hidden_dims', [256, 256, 128])
        actor_lr = float(config.get('actor_lr', 1e-4))
        critic_lr = float(config.get('critic_lr', 1e-3))
        self.gamma = float(config.get('gamma', 0.99))
        self.tau = float(config.get('tau', 0.005))
        self.policy_noise = float(config.get('policy_noise', 0.2))
        self.noise_clip = float(config.get('noise_clip', 0.5))
        self.policy_delay = int(config.get('policy_delay', 2))
        self.td3_bc_enabled = bool(config.get('td3_bc_enabled', False))
        self.td3_bc_alpha = float(config.get('td3_bc_alpha', 2.5))
        self.td3_bc_weight = float(config.get('td3_bc_weight', 1.0))
        self.td3_bc_loss_type = str(config.get('td3_bc_loss_type', 'smooth_l1')).lower()
        self.td3_bc_loss_scale = float(config.get('td3_bc_loss_scale', 1.0))
        self.td3_bc_target_ratio = float(config.get('td3_bc_target_ratio', 0.0))
        self.actor_residual_target_scale = float(config.get('actor_residual_target_scale', 1.0))
        if (
            not math.isfinite(self.actor_residual_target_scale)
            or not 0.0 < self.actor_residual_target_scale <= 1.0
        ):
            raise ValueError('agent.actor_residual_target_scale must be finite and in (0, 1]')
        self.td3_bc_max_effective_weight = float(config.get('td3_bc_max_effective_weight', 1.0e6))
        self.max_target_q_abs_mean = float(config.get('max_target_q_abs_mean', 0.0))
        self.max_actor_q_abs_mean = float(config.get('max_actor_q_abs_mean', 0.0))
        self.critic_loss_type = str(config.get('critic_loss_type', 'mse')).lower()
        self.critic_loss_delta = float(config.get('critic_loss_delta', 1.0))
        self.critic_grad_norm = float(config.get('critic_grad_norm', 10.0))
        self.update_step = 0

        self.actor = ActorMLP(obs_dim, action_dim, hidden_dims).to(device)
        self.actor_target = ActorMLP(obs_dim, action_dim, hidden_dims).to(device)
        self.critic1 = CriticMLP(obs_dim, action_dim, hidden_dims).to(device)
        self.critic2 = CriticMLP(obs_dim, action_dim, hidden_dims).to(device)
        self.critic1_target = CriticMLP(obs_dim, action_dim, hidden_dims).to(device)
        self.critic2_target = CriticMLP(obs_dim, action_dim, hidden_dims).to(device)
        self.actor_execution = ActorMLP(obs_dim, action_dim, hidden_dims).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_execution.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr)

    def set_actor_obs_normalizer(self, obs_median: torch.Tensor, obs_iqr: torch.Tensor) -> None:
        self.actor.set_obs_normalizer(obs_median.to(self.device), obs_iqr.to(self.device))
        self.actor_target.set_obs_normalizer(obs_median.to(self.device), obs_iqr.to(self.device))
        self.actor_execution.set_obs_normalizer(obs_median.to(self.device), obs_iqr.to(self.device))

    def set_critic_obs_normalizer(self, obs_median: torch.Tensor, obs_iqr: torch.Tensor) -> None:
        for critic in (self.critic1, self.critic2, self.critic1_target, self.critic2_target):
            critic.set_obs_normalizer(obs_median.to(self.device), obs_iqr.to(self.device))

    def select_action(self, obs: torch.Tensor, noise_std: float = 0.0, deterministic: bool = False) -> torch.Tensor:
        if not self.actor_execution_valid:
            return torch.zeros(
                (obs.shape[0], self.actor.backbone[-1].out_features),
                dtype=obs.dtype,
                device=self.device,
            )
        self.actor_execution.eval()
        with torch.no_grad():
            action = self._actor_output(self.actor_execution, obs.to(self.device))
        if not deterministic and noise_std > 0.0:
            action = action + noise_std * torch.randn_like(action)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return action.clamp(-1.0, 1.0)

    def _actor_output(self, actor: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(actor, ActorMLP):
            return actor(obs, self.actor_residual_target_scale)
        return actor(obs)

    @staticmethod
    def _module_is_finite(module: nn.Module, *, include_gradients: bool = False) -> bool:
        for parameter in module.parameters():
            if not torch.isfinite(parameter).all():
                return False
            if include_gradients and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                return False
        for buffer in module.buffers():
            if not torch.isfinite(buffer).all():
                return False
        return True

    @property
    def actor_execution_valid(self) -> bool:
        return self._module_is_finite(self.actor_execution)

    def _publish_actor_execution(self) -> bool:
        if not self._module_is_finite(self.actor):
            return False
        self.actor_execution.load_state_dict(self.actor.state_dict())
        self.actor_execution.eval()
        return True

    def fixed_offline_batch_is_healthy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> bool:
        """Check the fixed recovery batch without mutating model or optimizer state."""
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            return False
        if not all(self._module_is_finite(module) for module in (self.actor, self.critic1, self.critic2)):
            return False
        with torch.no_grad():
            actor_actions = self._actor_output(self.actor, observations)
            critic1_values = self.critic1(observations, actions)
            critic2_values = self.critic2(observations, actions)
        return all(torch.isfinite(values).all() for values in (
            actor_actions,
            critic1_values,
            critic2_values,
        ))

    def bc_update(self, obs: torch.Tensor, target_action: torch.Tensor, loss_type: str = 'mse') -> dict[str, float]:
        pred = self._actor_output(self.actor, obs.to(self.device))
        target = target_action.to(self.device)
        if loss_type == 'smooth_l1':
            loss = nn.functional.smooth_l1_loss(pred, target)
        else:
            loss = nn.functional.mse_loss(pred, target)
        mae = torch.abs(pred - target).mean()
        torque_mae = torch.abs(pred[:, 0] - target[:, 0]).mean()
        pitch_mae = torch.abs(pred[:, 1] - target[:, 1]).mean()
        if not all(math.isfinite(float(value.item())) for value in (loss, mae, torque_mae, pitch_mae)):
            return {'bc_update_skipped_nonfinite': 1.0}

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not self._module_is_finite(self.actor, include_gradients=True):
            self.actor_optimizer.zero_grad(set_to_none=True)
            return {'bc_update_skipped_nonfinite_gradient': 1.0}
        actor_snapshot = copy.deepcopy(self.actor.state_dict())
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        if not self._publish_actor_execution():
            self.actor.load_state_dict(actor_snapshot)
            return {'bc_update_skipped_nonfinite_parameter': 1.0}
        return {
            'bc_loss': float(loss.item()),
            'bc_mae': float(mae.item()),
            'bc_mse': float(loss.item()),
            'bc_torque_mae': float(torque_mae.item()),
            'bc_pitch_mae': float(pitch_mae.item()),
        }

    def set_critic_requires_grad(self, requires_grad: bool) -> None:
        critics = [self.critic1, self.critic2, self.critic1_target, self.critic2_target]
        for module in critics:
            for param in module.parameters():
                param.requires_grad = requires_grad

    def _compute_bc_loss(self, pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
        if loss_type == 'smooth_l1':
            return nn.functional.smooth_l1_loss(pred, target)
        return nn.functional.mse_loss(pred, target)

    def _compute_critic_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.critic_loss_type == 'smooth_l1':
            return nn.functional.smooth_l1_loss(prediction, target, beta=self.critic_loss_delta)
        return nn.functional.mse_loss(prediction, target)

    def update(
        self,
        batch: TD3Batch,
        bc_obs: torch.Tensor | None = None,
        bc_actions: torch.Tensor | None = None,
        bc_weight_override: float | None = None,
        update_actor: bool = True,
        actor_update_preflight: Callable[[dict[str, float]], bool] | None = None,
    ) -> dict[str, float]:
        self.update_step += 1
        actor_update_slot_due = self.update_step % self.policy_delay == 0

        def report_critic_result(metrics: dict[str, float]) -> dict[str, float]:
            metrics.setdefault('actor_update_slot_due', float(actor_update_slot_due))
            if actor_update_preflight is not None:
                actor_update_preflight(metrics)
            return metrics

        obs = batch.obs.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        dones = batch.dones.to(self.device)
        if not all(torch.isfinite(tensor).all() for tensor in (obs, actions, rewards, next_obs, dones)):
            return report_critic_result({'update_skipped_nonfinite_batch': 1.0})

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (
                self._actor_output(self.actor_target, next_obs) + noise
            ).clamp(-self.actor_residual_target_scale, self.actor_residual_target_scale)
            next_actions = torch.nan_to_num(next_actions, nan=0.0, posinf=1.0, neginf=-1.0)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            target = rewards + self.gamma * (1.0 - dones) * target_q
            target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)

        target_q_abs_mean = target_q.abs().mean()
        if self.max_target_q_abs_mean > 0.0 and float(target_q_abs_mean.item()) > self.max_target_q_abs_mean:
            return report_critic_result({
                'update_skipped_q_guard': 1.0,
                'target_q_mean': float(target.mean().item()),
                'target_q_abs_mean': float(target_q_abs_mean.item()),
            })

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic1_loss = self._compute_critic_loss(q1, target)
        critic2_loss = self._compute_critic_loss(q2, target)
        if not all(math.isfinite(float(value.item())) for value in (critic1_loss, critic2_loss, q1.mean(), q2.mean(), target.mean())):
            return report_critic_result({'update_skipped_nonfinite_loss': 1.0})

        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        if not self._module_is_finite(self.critic1, include_gradients=True):
            self.critic1_optimizer.zero_grad(set_to_none=True)
            return report_critic_result({'update_skipped_nonfinite_gradient': 1.0})
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=self.critic_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        if not self._module_is_finite(self.critic2, include_gradients=True):
            self.critic2_optimizer.zero_grad(set_to_none=True)
            return report_critic_result({'update_skipped_nonfinite_gradient': 1.0})
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=self.critic_grad_norm)
        self.critic2_optimizer.step()
        if not self._module_is_finite(self.critic1) or not self._module_is_finite(self.critic2):
            return report_critic_result({'update_skipped_nonfinite_parameter': 1.0})

        metrics = {
            'actor_update_slot_due': float(actor_update_slot_due),
            'critic1_loss': float(critic1_loss.item()),
            'critic2_loss': float(critic2_loss.item()),
            'q1_mean': float(q1.mean().item()),
            'q2_mean': float(q2.mean().item()),
            'target_q_mean': float(target.mean().item()),
            'target_q_abs_mean': float(target_q_abs_mean.item()),
            'critic1_td_error_abs_mean': float((q1 - target).abs().mean().item()),
            'critic2_td_error_abs_mean': float((q2 - target).abs().mean().item()),
            'critic_q_abs_gap_mean': float((q1 - q2).abs().mean().item()),
            'critic_obs_scaled_abs_mean': float(self.critic1.normalize_obs(obs).abs().mean().item()),
            'critic_obs_scaled_abs_max': float(self.critic1.normalize_obs(obs).abs().max().item()),
        }

        self.soft_update_critics()

        if actor_update_preflight is not None:
            update_actor = bool(update_actor and actor_update_preflight(metrics))

        if not update_actor:
            metrics['actor_update_deferred'] = 1.0
            return metrics

        if actor_update_slot_due:
            metrics['actor_update_attempted'] = 1.0
            actor_actions = self._actor_output(self.actor, obs)
            actor_q = self.critic1(obs, actor_actions)
            actor_rl_loss = -actor_q.mean()
            actor_loss = actor_rl_loss
            metrics['actor_rl_loss'] = float(actor_rl_loss.item())
            metrics['actor_q_mean'] = float(actor_q.mean().item())
            metrics['actor_q_abs_mean'] = float(actor_q.abs().mean().item())
            if self.max_actor_q_abs_mean > 0.0 and float(actor_q.abs().mean().item()) > self.max_actor_q_abs_mean:
                metrics['actor_update_skipped_q_guard'] = 1.0
                return metrics
            if self.td3_bc_enabled and bc_obs is not None and bc_actions is not None:
                bc_obs = bc_obs.to(self.device)
                bc_actions = bc_actions.to(self.device)
                if torch.isfinite(bc_obs).all() and torch.isfinite(bc_actions).all():
                    bc_pred = self._actor_output(self.actor, bc_obs)
                    bc_loss = self._compute_bc_loss(bc_pred, bc_actions, self.td3_bc_loss_type)
                    q_scale = actor_q.abs().mean().detach().clamp_min(1e-6)
                    lambda_coef = self.td3_bc_alpha / q_scale
                    bc_weight = self.td3_bc_weight if bc_weight_override is None else float(bc_weight_override)
                    rl_term = -lambda_coef * actor_q.mean()
                    effective_bc_weight = bc_weight * self.td3_bc_loss_scale
                    if self.td3_bc_target_ratio > 0.0:
                        required_weight = self.td3_bc_target_ratio * rl_term.detach().abs() / bc_loss.detach().clamp_min(1e-8)
                        effective_bc_weight = max(effective_bc_weight, float(required_weight.item()))
                    effective_bc_weight = min(effective_bc_weight, self.td3_bc_max_effective_weight)
                    bc_term = effective_bc_weight * bc_loss
                    actor_loss = rl_term + bc_term
                    metrics['actor_bc_loss'] = float(bc_loss.item())
                    metrics['actor_bc_lambda'] = float(lambda_coef.item())
                    metrics['actor_bc_weight'] = float(bc_weight)
                    metrics['actor_bc_effective_weight'] = float(effective_bc_weight)
                    metrics['actor_rl_term'] = float(rl_term.item())
                    metrics['actor_bc_term'] = float(bc_term.item())
                    metrics['actor_bc_to_rl_ratio'] = float(bc_term.detach().abs().item() / rl_term.detach().abs().clamp_min(1e-8).item())
            if not math.isfinite(float(actor_loss.item())):
                metrics['actor_update_skipped_nonfinite'] = 1.0
                return metrics
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if not self._module_is_finite(self.actor, include_gradients=True):
                self.actor_optimizer.zero_grad(set_to_none=True)
                metrics['actor_update_skipped_nonfinite_gradient'] = 1.0
                return metrics
            actor_snapshot = copy.deepcopy(self.actor.state_dict())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()
            if not self._publish_actor_execution():
                self.actor.load_state_dict(actor_snapshot)
                metrics['actor_update_skipped_nonfinite_parameter'] = 1.0
                return metrics
            self.soft_update_actor()
            metrics['actor_loss'] = float(actor_loss.item())
        return metrics

    def soft_update(self) -> None:
        self.soft_update_actor()
        self.soft_update_critics()

    def soft_update_actor(self) -> None:
        self._soft_update_module(self.actor, self.actor_target)

    def soft_update_critics(self) -> None:
        self._soft_update_module(self.critic1, self.critic1_target)
        self._soft_update_module(self.critic2, self.critic2_target)

    def _soft_update_module(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_execution': self.actor_execution.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'update_step': self.update_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict['actor'], strict=False)
        self.actor_target.load_state_dict(state_dict['actor_target'], strict=False)
        actor_execution_state = state_dict.get('actor_execution')
        if isinstance(actor_execution_state, dict):
            self.actor_execution.load_state_dict(actor_execution_state, strict=False)
        else:
            self._publish_actor_execution()
        self.critic1.load_state_dict(state_dict['critic1'], strict=False)
        self.critic2.load_state_dict(state_dict['critic2'], strict=False)
        self.critic1_target.load_state_dict(state_dict['critic1_target'], strict=False)
        self.critic2_target.load_state_dict(state_dict['critic2_target'], strict=False)
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(state_dict['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(state_dict['critic2_optimizer'])
        self.update_step = int(state_dict.get('update_step', 0))
