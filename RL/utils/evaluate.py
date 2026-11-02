from __future__ import annotations

import numpy as np
import torch

from RL.utils.metrics import summarize_metrics
from RL.utils.safe_residual_control import safe_action_metrics


def evaluate_policy(agent, env, episodes: int, logger=None, prefix: str = 'eval', controller=None) -> dict[str, float]:
    episode_rewards: list[float] = []
    episode_infos: list[dict[str, float]] = []
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        if controller is not None:
            controller.reset_condition(obs, previous_action_real=getattr(env, 'last_sent_action', None))
        done = False
        total_reward = 0.0
        infos: list[dict[str, float]] = []
        invalid_episode = False
        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = agent.select_action(obs_tensor, deterministic=True)
            residual_action = action.squeeze(0).cpu().numpy()
            if controller is not None:
                safe_action = controller.action_from_policy(residual_action, obs, allow_policy=True)
                action_for_env = safe_action.action_norm
                action_real_for_env = safe_action.action_real
            else:
                safe_action = None
                action_for_env = residual_action
                action_real_for_env = None
            if action_real_for_env is not None and bool(getattr(env, 'supports_physical_action', False)):
                step_result = env.step(action_for_env, action_real=action_real_for_env)
            else:
                step_result = env.step(action_for_env)
            if safe_action is not None:
                sent_action = np.asarray(
                    [
                        step_result.info.get('action_torque', safe_action.action_real[0]),
                        step_result.info.get('action_pitch', safe_action.action_real[1]),
                    ],
                    dtype=np.float64,
                )
                step_result.info.update(
                    safe_action_metrics(safe_action, sent_action_real=sent_action)
                )
                step_result.info['control_ood_score'] = safe_action.ood_score
                unsafe = controller.observe_step(float(step_result.info.get('power', 0.0)))
                step_result.info['control_unsafe_stop'] = float(unsafe)
                if unsafe:
                    done = True
            obs = step_result.obs
            done = done or step_result.done
            if not np.isfinite(step_result.reward):
                invalid_episode = True
                break
            total_reward += step_result.reward
            infos.append(step_result.info)
            if step_result.info.get('env_invalid_prediction', 0.0) > 0.0 or step_result.info.get('env_invalid_reward', 0.0) > 0.0:
                invalid_episode = True
                break
        summary = summarize_metrics(infos)
        episode_is_invalid = invalid_episode or not np.isfinite(total_reward)
        summary['episode_invalid'] = float(episode_is_invalid)
        if not episode_is_invalid:
            summary['episode_reward'] = total_reward
            episode_rewards.append(total_reward)
        episode_infos.append(summary)
        if logger is not None:
            logger.log_episode(episode, {f'{prefix}_{k}': v for k, v in summary.items()})
    merged = summarize_metrics(episode_infos)
    reward_values = np.asarray(episode_rewards, dtype=np.float64)
    merged['reward_mean'] = float(np.mean(reward_values)) if reward_values.size else 0.0
    merged['reward_median'] = float(np.median(reward_values)) if reward_values.size else 0.0
    merged['reward_std'] = float(np.std(reward_values)) if reward_values.size else 0.0
    merged['reward_valid_episode_count'] = float(reward_values.size)
    return merged
