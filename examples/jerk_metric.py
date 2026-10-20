# ruff: noqa
"""PRISM-style jerk metric — SHARED by the DROID and LIBERO eval clients.

Signal type (velocity vs position) is a flag, because it decides how many finite
differences to take. Jerk is the 3rd derivative of position:

    signal="velocity"  -> order=2     (signal is already a 1st derivative)
        * DROID arm  = joint velocity
        * LIBERO arm = delta end-effector pose  (== 1st difference of position)
    signal="position"  -> order=3
        * proprioception = joint_position / eef_pos

Only the arm dims are tracked; the gripper (an absolute open/close command, not a
velocity) is excluded from the jerk computation.

Per step, per dim:   j_t[d] = |Delta^order(signal)_t[d]| / dt**order   (mean-abs).

WARMUP: the first `order` steps of each episode do not have enough samples to take
the difference, so `update()` returns None and they are NEVER counted in any average
(this is the "don't count the first two zero actions" behaviour, order=2). Gate
per-step printouts on `n_counted > 0` so warmup steps are not displayed either.

Three running levels, each PER DIM and MEANED OVER DIMS:
    current      : this step
    running avg  : mean over steps within the current episode   (level-1 running avg)
    across       : mean over episodes                           (level-2 running avg)
"""

import numpy as np

_ORDER = {"velocity": 2, "position": 3}


class JerkMeter:
    def __init__(self, name: str, signal: str, n_dims: int, dt: float = 1.0, labels=None):
        if signal not in _ORDER:
            raise ValueError(f"signal must be one of {list(_ORDER)}, got {signal!r}")
        self.name = name
        self.signal = signal
        self.order = _ORDER[signal]
        self.n_dims = int(n_dims)
        self.dt = float(dt)
        self.labels = list(labels) if labels is not None else [f"j{i}" for i in range(self.n_dims)]
        if len(self.labels) != self.n_dims:
            raise ValueError(f"labels length {len(self.labels)} != n_dims {self.n_dims}")

        # across-episode accumulators (per dim): all rollouts + successful-only
        self._all_sum = np.zeros(self.n_dims)
        self._all_cnt = 0
        self._succ_sum = np.zeros(self.n_dims)
        self._succ_cnt = 0
        self.episode_records: list[dict] = []
        self._reset_episode()

    # ------------------------------------------------------------------ episode
    def _reset_episode(self):
        self._buf: list[np.ndarray] = []       # last (order+1) signal vectors
        self._ep_sum = np.zeros(self.n_dims)    # sum of per-step per-dim jerk
        self._ep_cnt = 0                        # COUNTED steps (warmup excluded)
        self._ep_max = np.zeros(self.n_dims)
        self._cur = np.zeros(self.n_dims)

    def start_episode(self):
        self._reset_episode()

    def update(self, vec) -> np.ndarray | None:
        """Feed one executed step's signal. Returns per-dim jerk, or None during the
        first `order` (warmup) steps, which are never counted in any average."""
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)[: self.n_dims]
        self._buf.append(vec)
        if len(self._buf) > self.order + 1:
            self._buf.pop(0)
        if len(self._buf) < self.order + 1:
            return None  # warmup: not enough samples yet -> not counted
        diff = np.diff(np.stack(self._buf, axis=0), n=self.order, axis=0)[-1]
        j = np.abs(diff) / (self.dt ** self.order)
        self._cur = j
        self._ep_sum += j
        self._ep_cnt += 1
        self._ep_max = np.maximum(self._ep_max, j)
        return j

    @property
    def n_counted(self) -> int:
        return self._ep_cnt

    # current -------------------------------------------------------------
    @property
    def current(self) -> np.ndarray:
        return self._cur

    @property
    def current_mean(self) -> float:
        return float(self._cur.mean())

    # within-episode running average -------------------------------------
    @property
    def episode_avg(self) -> np.ndarray:
        return self._ep_sum / self._ep_cnt if self._ep_cnt else np.zeros(self.n_dims)

    @property
    def episode_avg_mean(self) -> float:
        return float(self.episode_avg.mean())

    # across-episode running average -------------------------------------
    @property
    def across_avg(self) -> np.ndarray:
        return self._all_sum / self._all_cnt if self._all_cnt else np.zeros(self.n_dims)

    @property
    def across_avg_mean(self) -> float:
        return float(self.across_avg.mean())

    @property
    def across_avg_success(self) -> np.ndarray:
        return self._succ_sum / self._succ_cnt if self._succ_cnt else np.zeros(self.n_dims)

    def end_episode(self, success=None) -> dict:
        """Finalize the episode and fold its per-dim mean into the across-episode running
        averages (all rollouts, and successful-only when success >= 0.5)."""
        ep = self.episode_avg.copy()
        rec = {
            "n_steps": int(self._ep_cnt),
            "jerk_mean": float(ep.mean()),
            "jerk_per_joint": ep.tolist(),
            "jerk_max": float(self._ep_max.max()) if self._ep_cnt else 0.0,
            "jerk_max_per_joint": self._ep_max.tolist(),
            "success": None if success is None else float(success),
        }
        if self._ep_cnt > 0:
            self._all_sum += ep
            self._all_cnt += 1
            if success is not None and float(success) >= 0.5:
                self._succ_sum += ep
                self._succ_cnt += 1
        self.episode_records.append(rec)
        return rec

    def summary(self) -> dict:
        """Across-episode aggregate for the whole session (the level-2 running average)."""
        return {
            "signal": self.signal,
            "order": self.order,
            "labels": list(self.labels),
            "n_episodes": int(self._all_cnt),
            "jerk_mean": self.across_avg_mean,
            "jerk_per_joint": self.across_avg.tolist(),
            "n_episodes_success": int(self._succ_cnt),
            "jerk_mean_success": (float(self.across_avg_success.mean()) if self._succ_cnt else None),
            "jerk_per_joint_success": (self.across_avg_success.tolist() if self._succ_cnt else None),
        }

    # ---------------------------------------------------------------- display
    def perjoint_str(self, vec, prec: int = 4) -> str:
        """'<label> value ...' using this meter's dim labels (e.g. j0 .. jN grip)."""
        vec = np.asarray(vec, dtype=float).reshape(-1)
        return "  ".join(f"{lbl} {v:.{prec}f}" for lbl, v in zip(self.labels, vec))

    def fmt_row(self, kind: str, vec, mean: float, prec: int = 4) -> str:
        """One aligned line:
        '<name> <kind>  mean over joints X  |  per joint  <label val ...>'."""
        return (
            f"  {self.name:<7} {kind:<11}  mean over joints {mean:.{prec}f}"
            f"  |  per joint  {self.perjoint_str(vec, prec)}"
        )

    def flat_row(self, prefix: str, rec: dict) -> dict:
        """CSV columns: <prefix>_jerk_mean/_max/_n_steps, then one per labelled dim."""
        row = {
            f"{prefix}_jerk_mean": rec["jerk_mean"],
            f"{prefix}_jerk_max": rec["jerk_max"],
            f"{prefix}_n_steps": rec["n_steps"],
        }
        for lbl, v in zip(self.labels, rec["jerk_per_joint"]):
            row[f"{prefix}_jerk_{lbl}"] = v
        return row


def render_step_block(step, meters, prec: int = 4) -> str:
    """Readable multi-line block: step header, then per meter a `current` row and a
    `running avg` row, each with the mean-over-joints column and per-joint values."""
    lines = [f"── step {step} " + "─" * 30]
    for m in meters:
        lines.append(m.fmt_row("current", m.current, m.current_mean, prec))
        lines.append(m.fmt_row("running avg", m.episode_avg, m.episode_avg_mean, prec))
    return "\n".join(lines)
