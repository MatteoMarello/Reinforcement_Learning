"""Policy-gradient agents for the FAIML Reinforcement Learning project.

This module intentionally implements the Phase 2 algorithms from scratch:

* REINFORCE without baseline
* REINFORCE with a scalar/constant baseline
* one-step Actor-Critic policy gradient

The code is written for continuous-control Gymnasium environments such as
``Hopper-v4``.  The actor is a diagonal Gaussian policy followed by a tanh
squashing transformation so that sampled actions respect the environment action
bounds.  The log-probability includes the tanh change-of-variables correction,
which keeps the policy-gradient objective consistent with the executed action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


EPS = 1e-6


def discount_rewards(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted Monte-Carlo returns for one trajectory.

    Args:
        rewards: Tensor with shape ``[T]`` containing rewards of one episode.
        gamma: Discount factor.

    Returns:
        Tensor ``G`` with shape ``[T]`` where
        ``G[t] = rewards[t] + gamma * rewards[t+1] + ...``.
    """

    if rewards.ndim != 1:
        rewards = rewards.view(-1)

    discounted = torch.zeros_like(rewards)
    running_return = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(rewards.numel())):
        running_return = rewards[t] + gamma * running_return
        discounted[t] = running_return

    return discounted


@dataclass(frozen=True)
class AgentConfig:
    """Hyperparameters controlling the policy-gradient update."""

    algorithm: str = "reinforce"
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    value_loss_coef: float = 0.5
    grad_clip_norm: Optional[float] = 1.0
    normalize_advantages: bool = False
    baseline_value: Optional[float] = None
    baseline_momentum: float = 0.9


class Policy(torch.nn.Module):
    """Diagonal Gaussian actor plus state-value critic.

    The actor samples an unconstrained Gaussian action ``u`` and maps it to the
    valid environment action interval with ``a = bias + scale * tanh(u)``.
    Hopper has bounds ``[-1, 1]`` on all three action dimensions, but the class
    is kept generic for other bounded continuous-control environments.
    """

    def __init__(
        self,
        state_space: int,
        action_space: int,
        hidden_size: int = 64,
        action_low: Optional[Sequence[float]] = None,
        action_high: Optional[Sequence[float]] = None,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.state_space = int(state_space)
        self.action_space = int(action_space)
        self.hidden_size = int(hidden_size)

        self.actor_body = torch.nn.Sequential(
            torch.nn.Linear(self.state_space, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Tanh(),
        )
        self.actor_mean = torch.nn.Linear(self.hidden_size, self.action_space)
        self.log_std = torch.nn.Parameter(
            torch.full((self.action_space,), float(log_std_init), dtype=torch.float32)
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.state_space, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, 1),
        )

        low = np.full(self.action_space, -1.0, dtype=np.float32) if action_low is None else np.asarray(action_low, dtype=np.float32)
        high = np.full(self.action_space, 1.0, dtype=np.float32) if action_high is None else np.asarray(action_high, dtype=np.float32)

        if low.shape != (self.action_space,) or high.shape != (self.action_space,):
            raise ValueError(
                "action_low and action_high must have shape "
                f"({self.action_space},), got {low.shape} and {high.shape}."
            )
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("This implementation expects finite continuous action bounds.")
        if not np.all(high > low):
            raise ValueError("Each action upper bound must be greater than its lower bound.")

        self.register_buffer("action_low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high, dtype=torch.float32))
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

        self.init_weights()

    def init_weights(self) -> None:
        """Stable small-weight initialization for policy-gradient training."""

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        torch.nn.init.zeros_(self.actor_mean.bias)

    def actor_parameters(self) -> Iterable[torch.nn.Parameter]:
        yield from self.actor_body.parameters()
        yield from self.actor_mean.parameters()
        yield self.log_std

    def critic_parameters(self) -> Iterable[torch.nn.Parameter]:
        yield from self.critic.parameters()

    def _distribution(self, states: torch.Tensor) -> Normal:
        features = self.actor_body(states)
        mean = self.actor_mean(features)
        std = torch.exp(torch.clamp(self.log_std, min=-20.0, max=2.0))
        return Normal(mean, std)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states)

    def forward(self, states: torch.Tensor) -> Normal:
        """Return the unconstrained Gaussian distribution used by the actor."""

        return self._distribution(states)

    def sample_action(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample or deterministically select bounded actions.

        Args:
            states: Tensor with shape ``[batch, state_dim]`` or ``[state_dim]``.
            deterministic: If true, use the mean action; otherwise sample from
                the stochastic policy.

        Returns:
            ``(actions, log_probs)``.  ``log_probs`` has shape ``[batch]`` and is
            ``None`` in deterministic mode.
        """

        squeeze_batch = False
        if states.ndim == 1:
            states = states.unsqueeze(0)
            squeeze_batch = True

        dist = self._distribution(states)
        raw_action = dist.mean if deterministic else dist.sample()
        squashed_action = torch.tanh(raw_action)
        action = self.action_bias + self.action_scale * squashed_action

        log_prob = None
        if not deterministic:
            # Change of variables for a = bias + scale * tanh(u).
            correction = torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + EPS)
            log_prob = (dist.log_prob(raw_action) - correction).sum(dim=-1)

        if squeeze_batch:
            action = action.squeeze(0)
            if log_prob is not None:
                log_prob = log_prob.squeeze(0)

        return action, log_prob

    def get_initialization_kwargs(self) -> Dict[str, Any]:
        return {
            "state_space": self.state_space,
            "action_space": self.action_space,
            "hidden_size": self.hidden_size,
            "action_low": self.action_low.detach().cpu().numpy().tolist(),
            "action_high": self.action_high.detach().cpu().numpy().tolist(),
        }


class Agent:
    """Small from-scratch policy-gradient agent for Hopper-v4."""

    VALID_ALGORITHMS = {"reinforce", "reinforce_baseline", "actor_critic"}

    def __init__(
        self,
        policy: Policy,
        config: Optional[AgentConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or AgentConfig()
        if self.config.algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{self.config.algorithm}'. "
                f"Choose one of {sorted(self.VALID_ALGORITHMS)}."
            )
        if not (0.0 <= self.config.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if not (0.0 <= self.config.baseline_momentum < 1.0):
            raise ValueError("baseline_momentum must be in [0, 1).")

        self.train_device = torch.device(device)
        self.policy = policy.to(self.train_device)

        if self.config.algorithm == "actor_critic":
            self.optimizer = torch.optim.Adam(
                [
                    {"params": list(self.policy.actor_parameters()), "lr": self.config.actor_lr},
                    {"params": list(self.policy.critic_parameters()), "lr": self.config.critic_lr},
                ]
            )
        else:
            self.optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=self.config.actor_lr)

        self.running_baseline: Optional[float] = None
        if self.config.baseline_value is not None:
            self.running_baseline = float(self.config.baseline_value)

        self.states: List[np.ndarray] = []
        self.next_states: List[np.ndarray] = []
        self.action_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    @property
    def gamma(self) -> float:
        return self.config.gamma

    def reset_memory(self) -> None:
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def get_action(
        self,
        state: np.ndarray,
        evaluation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compatibility wrapper used by the original template."""

        return self.act(state=state, evaluation=evaluation)

    def act(
        self,
        state: np.ndarray,
        evaluation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.train_device)

        if evaluation:
            with torch.no_grad():
                return self.policy.sample_action(state_tensor, deterministic=True)

        return self.policy.sample_action(state_tensor, deterministic=False)

    def store_outcome(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action_log_prob: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def _trajectory_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.rewards:
            raise RuntimeError("Cannot update policy: no trajectory has been stored.")

        states = torch.as_tensor(np.asarray(self.states), dtype=torch.float32, device=self.train_device)
        next_states = torch.as_tensor(np.asarray(self.next_states), dtype=torch.float32, device=self.train_device)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.train_device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.train_device)
        action_log_probs = torch.stack(self.action_log_probs).to(self.train_device).view(-1)
        return states, next_states, action_log_probs, rewards, dones

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= 1:
            return values
        return (values - values.mean()) / (values.std(unbiased=False) + EPS)

    def _apply_gradient_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None and self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

    def update_policy(self) -> Dict[str, float]:
        """Update the policy using the selected algorithm and clear memory."""

        states, next_states, action_log_probs, rewards, dones = self._trajectory_tensors()
        self.reset_memory()

        if self.config.algorithm in {"reinforce", "reinforce_baseline"}:
            stats = self._update_reinforce(action_log_probs=action_log_probs, rewards=rewards)
        elif self.config.algorithm == "actor_critic":
            stats = self._update_actor_critic(
                states=states,
                next_states=next_states,
                action_log_probs=action_log_probs,
                rewards=rewards,
                dones=dones,
            )
        else:  # defensive guard; __init__ already validates this.
            raise RuntimeError(f"Unsupported algorithm: {self.config.algorithm}")

        return stats

    def _update_reinforce(self, action_log_probs: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        returns = discount_rewards(rewards, self.config.gamma)
        baseline = 0.0

        if self.config.algorithm == "reinforce_baseline":
            if self.config.baseline_value is not None:
                baseline = float(self.config.baseline_value)
            elif self.running_baseline is not None:
                baseline = float(self.running_baseline)
            # If no value has been observed yet, the first update uses baseline 0.

        advantages = returns - baseline
        if self.config.normalize_advantages:
            advantages = self._normalize(advantages)

        policy_loss = -(action_log_probs * advantages.detach()).mean()
        self._apply_gradient_step(policy_loss)

        if self.config.algorithm == "reinforce_baseline" and self.config.baseline_value is None:
            observed_baseline = float(returns.mean().detach().cpu().item())
            if self.running_baseline is None:
                self.running_baseline = observed_baseline
            else:
                m = self.config.baseline_momentum
                self.running_baseline = m * self.running_baseline + (1.0 - m) * observed_baseline

        return {
            "loss": float(policy_loss.detach().cpu().item()),
            "actor_loss": float(policy_loss.detach().cpu().item()),
            "critic_loss": 0.0,
            "mean_discounted_return": float(returns.mean().detach().cpu().item()),
            "baseline": float(baseline),
        }

    def _update_actor_critic(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        values = self.policy.value(states).view(-1)

        with torch.no_grad():
            next_values = self.policy.value(next_states).view(-1)
            td_targets = rewards + self.config.gamma * next_values * (1.0 - dones)
            advantages = td_targets - values.detach()

        actor_advantages = self._normalize(advantages) if self.config.normalize_advantages else advantages
        actor_loss = -(action_log_probs * actor_advantages.detach()).mean()
        critic_loss = F.mse_loss(values, td_targets)
        loss = actor_loss + self.config.value_loss_coef * critic_loss

        self._apply_gradient_step(loss)

        return {
            "loss": float(loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "mean_discounted_return": float(discount_rewards(rewards, self.config.gamma).mean().detach().cpu().item()),
            "baseline": 0.0,
        }

    def save(self, path: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "policy_kwargs": self.policy.get_initialization_kwargs(),
            "agent_config": asdict(self.config),
            "running_baseline": self.running_baseline,
            "extra": extra or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "Agent":
        checkpoint = torch.load(path, map_location=device)
        policy = Policy(**checkpoint["policy_kwargs"])
        config = AgentConfig(**checkpoint["agent_config"])
        agent = cls(policy=policy, config=config, device=device)
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.running_baseline = checkpoint.get("running_baseline")
        return agent
