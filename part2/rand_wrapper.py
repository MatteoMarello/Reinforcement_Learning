"""Domain randomization wrappers for the PandaPush task.

The course project simulates a sim-to-real gap by changing only one dynamics
parameter: the mass of the pushed cube.  The source environment starts with a
1 kg cube, while the target environment starts with a 5 kg cube.  This wrapper
implements two training-time strategies:

* UDR: Uniform Domain Randomization over a fixed mass interval.
* ADR: Automatic Domain Randomization with a simple performance-driven
  curriculum.  ADR starts near the nominal source mass and expands/contracts
  the sampled interval depending on recent success at the boundaries.

The wrapper intentionally does not need to know the target mass.  The upper
and lower global bounds are hyperparameters chosen by the experimenter.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class RandomizationState:
    mode: str
    nominal_mass: float
    current_mass: float
    mass_min: float
    mass_max: float
    global_mass_min: float
    global_mass_max: float
    last_sample_type: str
    episodes_seen: int
    boundary_window: int
    lower_boundary_success_rate: Optional[float]
    upper_boundary_success_rate: Optional[float]


class RandomizationWrapper(gym.Wrapper):
    """Randomize the PandaPush cube mass at reset time.

    Parameters
    ----------
    env:
        A PandaPush-v3 environment.
    mass_range:
        Global interval allowed for randomization.  For UDR this is the fixed
        sampling interval.  For ADR it is a hard safety interval that the
        adaptive curriculum cannot exceed.
    mode:
        One of {"none", "udr", "adr"}.
    seed:
        Optional independent RNG seed for mass sampling.
    adr_initial_delta:
        Initial half-width of the ADR interval around the nominal mass.
    adr_step_size:
        Amount by which a boundary is expanded or contracted.
    adr_boundary_prob:
        Probability of sampling exactly one boundary in ADR.  Boundary samples
        are used to estimate whether the curriculum can safely expand.
    adr_window_size:
        Number of recent boundary episodes used to compute success rates.
    adr_high_threshold:
        Expand a boundary when the corresponding success rate is at least this.
    adr_low_threshold:
        Contract a boundary when the corresponding success rate is at most this.
    adr_min_range_width:
        Minimum width kept for the adaptive range.
    verbose:
        If True, print sampled masses and ADR updates.
    """

    valid_modes = {"none", "udr", "adr"}

    def __init__(
        self,
        env: gym.Env,
        mass_range: Tuple[float, float] = (0.5, 8.0),
        mode: str = "none",
        seed: Optional[int] = None,
        adr_initial_delta: float = 0.1,
        adr_step_size: float = 0.25,
        adr_boundary_prob: float = 0.5,
        adr_window_size: int = 10,
        adr_high_threshold: float = 0.8,
        adr_low_threshold: float = 0.2,
        adr_min_range_width: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__(env)
        if mode not in self.valid_modes:
            raise ValueError(f"Unknown randomization mode '{mode}'. Valid modes: {sorted(self.valid_modes)}")
        if mass_range[0] <= 0 or mass_range[1] <= 0 or mass_range[0] >= mass_range[1]:
            raise ValueError("mass_range must be a positive interval (low, high) with low < high.")

        self.mode = mode
        self.mass_min_limit = float(mass_range[0])
        self.mass_max_limit = float(mass_range[1])
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(seed)

        self.nominal_mass = self._infer_current_mass(default=1.0)
        self.current_mass = self.nominal_mass

        if self.mode == "adr":
            self.mass_min = max(self.mass_min_limit, self.nominal_mass - abs(float(adr_initial_delta)))
            self.mass_max = min(self.mass_max_limit, self.nominal_mass + abs(float(adr_initial_delta)))
        elif self.mode == "udr":
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
        else:
            self.mass_min = self.nominal_mass
            self.mass_max = self.nominal_mass

        self.adr_step_size = abs(float(adr_step_size))
        self.adr_boundary_prob = float(np.clip(adr_boundary_prob, 0.0, 1.0))
        self.adr_window_size = int(max(1, adr_window_size))
        self.adr_high_threshold = float(adr_high_threshold)
        self.adr_low_threshold = float(adr_low_threshold)
        self.adr_min_range_width = float(max(1e-6, adr_min_range_width))

        self.lower_boundary_results: Deque[float] = deque(maxlen=self.adr_window_size)
        self.upper_boundary_results: Deque[float] = deque(maxlen=self.adr_window_size)
        self.last_sample_type = "none"
        self.episodes_seen = 0

    # ------------------------------------------------------------------
    # PyBullet mass access helpers
    # ------------------------------------------------------------------
    def _object_body_id(self) -> int:
        sim = self.env.unwrapped.task.sim
        return sim._bodies_idx["object"]

    def _infer_current_mass(self, default: float = 1.0) -> float:
        task = getattr(self.env.unwrapped, "task", None)
        if task is not None and hasattr(task, "current_mass"):
            try:
                return float(task.current_mass)
            except Exception:
                pass
        try:
            sim = self.env.unwrapped.task.sim
            body_id = sim._bodies_idx["object"]
            return float(sim.physics_client.getDynamicsInfo(body_id, -1)[0])
        except Exception:
            return float(default)

    def _set_object_mass(self, new_mass: float) -> None:
        new_mass = float(new_mass)
        sim = self.env.unwrapped.task.sim
        object_body_id = self._object_body_id()
        sim.physics_client.changeDynamics(
            bodyUniqueId=object_body_id,
            linkIndex=-1,
            mass=new_mass,
        )
        if hasattr(self.env.unwrapped.task, "current_mass"):
            self.env.unwrapped.task.current_mass = new_mass
        self.current_mass = new_mass

    # ------------------------------------------------------------------
    # Sampling logic
    # ------------------------------------------------------------------
    def _sample_mass(self) -> Optional[float]:
        if self.mode == "none":
            self.last_sample_type = "none"
            return None

        if self.mode == "udr":
            self.last_sample_type = "uniform"
            return float(self.rng.uniform(self.mass_min, self.mass_max))

        # ADR: mostly sample inside the current interval, but force boundary
        # samples often enough to measure performance at the edges.
        if self.rng.random() < self.adr_boundary_prob:
            if self.rng.random() < 0.5:
                self.last_sample_type = "lower_boundary"
                return float(self.mass_min)
            self.last_sample_type = "upper_boundary"
            return float(self.mass_max)

        self.last_sample_type = "interior"
        return float(self.rng.uniform(self.mass_min, self.mass_max))

    def _success_from_info(self, info: Dict[str, Any]) -> float:
        if isinstance(info, dict) and "is_success" in info:
            return float(info["is_success"])
        return 0.0

    def _maybe_update_adr_range(self, success: float) -> None:
        if self.mode != "adr":
            return

        if self.last_sample_type == "lower_boundary":
            self.lower_boundary_results.append(success)
            self._update_boundary("lower")
        elif self.last_sample_type == "upper_boundary":
            self.upper_boundary_results.append(success)
            self._update_boundary("upper")

    def _update_boundary(self, side: str) -> None:
        history = self.lower_boundary_results if side == "lower" else self.upper_boundary_results
        if len(history) < self.adr_window_size:
            return

        success_rate = float(np.mean(history))
        old_min, old_max = self.mass_min, self.mass_max

        if side == "lower":
            if success_rate >= self.adr_high_threshold:
                self.mass_min = max(self.mass_min_limit, self.mass_min - self.adr_step_size)
                history.clear()
            elif success_rate <= self.adr_low_threshold:
                candidate = min(self.nominal_mass, self.mass_min + self.adr_step_size)
                self.mass_min = min(candidate, self.mass_max - self.adr_min_range_width)
                history.clear()
        else:
            if success_rate >= self.adr_high_threshold:
                self.mass_max = min(self.mass_max_limit, self.mass_max + self.adr_step_size)
                history.clear()
            elif success_rate <= self.adr_low_threshold:
                candidate = max(self.nominal_mass, self.mass_max - self.adr_step_size)
                self.mass_max = max(candidate, self.mass_min + self.adr_min_range_width)
                history.clear()

        if self.verbose and (old_min != self.mass_min or old_max != self.mass_max):
            print(
                f"[ADR] side={side} success_rate={success_rate:.2f} "
                f"range [{old_min:.3f}, {old_max:.3f}] -> [{self.mass_min:.3f}, {self.mass_max:.3f}]"
            )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, **kwargs: Any):
        new_mass = self._sample_mass()
        if new_mass is not None:
            self._set_object_mass(new_mass)
            if self.verbose:
                print(
                    f"[{self.mode}] mass={new_mass:.3f} "
                    f"range=[{self.mass_min:.3f}, {self.mass_max:.3f}] "
                    f"sample={self.last_sample_type}"
                )
        else:
            self.current_mass = self._infer_current_mass(default=self.nominal_mass)

        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info.update(self.get_randomization_state(prefix="dr/"))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info.update(self.get_randomization_state(prefix="dr/"))

        if terminated or truncated:
            self.episodes_seen += 1
            success = self._success_from_info(info)
            self._maybe_update_adr_range(success)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Introspection for logging callbacks
    # ------------------------------------------------------------------
    def get_randomization_state(self, prefix: str = "") -> Dict[str, Any]:
        state = RandomizationState(
            mode=self.mode,
            nominal_mass=float(self.nominal_mass),
            current_mass=float(self.current_mass),
            mass_min=float(self.mass_min),
            mass_max=float(self.mass_max),
            global_mass_min=float(self.mass_min_limit),
            global_mass_max=float(self.mass_max_limit),
            last_sample_type=str(self.last_sample_type),
            episodes_seen=int(self.episodes_seen),
            boundary_window=int(self.adr_window_size),
            lower_boundary_success_rate=(
                float(np.mean(self.lower_boundary_results)) if self.lower_boundary_results else None
            ),
            upper_boundary_success_rate=(
                float(np.mean(self.upper_boundary_results)) if self.upper_boundary_results else None
            ),
        )
        result = asdict(state)
        if prefix:
            return {f"{prefix}{key}": value for key, value in result.items()}
        return result
