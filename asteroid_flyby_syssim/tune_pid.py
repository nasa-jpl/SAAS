"""PID tuning utility for asteroid flyby attitude control.

This utility runs many short flyby simulations and optimizes the controller
PID gains to reduce pointing error. It reuses the main simulation config
schema and dynamics/component nodes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.optimize import differential_evolution

from syssim.core import InputPort, Node, NodeSystem

from .flyby_sim import (
    FlybyRunConfig,
    NodeAttitudeController,
    NodeAttitudeDynamics,
    NodeCenterPointingGuidance,
    NodeHyperbolicDynamics,
    NodeReactionWheel,
    NodeTorqueAllocator,
    NodeWheelAggregator,
    TETRAHEDRAL_WHEEL_AXES,
    _hyperbolic_state_from_params,
)
from .asteroid_gravity import NodeAsteroidGravity


class NodeTuneRecorderInputs(NamedTuple):
    q_err: InputPort
    tau_cmd_body: InputPort


class NodeTuneRecorder(Node):
    """Lightweight recorder for tuning objective metrics."""

    def __init__(self, **kwargs):
        self._i = NodeTuneRecorderInputs(
            InputPort("q_err", self),
            InputPort("tau_cmd_body", self),
        )
        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        self.t: list[float] = []
        self.q_err: list[np.ndarray] = []
        self.tau_cmd: list[np.ndarray] = []

    def update(self, sim_time: float):
        q_err = self._i.q_err.read()
        tau_cmd = self._i.tau_cmd_body.read()
        if q_err is None or tau_cmd is None:
            return
        self.t.append(sim_time)
        self.q_err.append(np.array(q_err, dtype=float))
        self.tau_cmd.append(np.array(tau_cmd, dtype=float))

    @property
    def i(self):
        return self._i

    @property
    def o(self):
        return ()


@dataclass
class GainBounds:
    kp_min: float = 0.005
    kp_max: float = 0.30
    kd_min: float = 0.01
    kd_max: float = 1.20
    ki_min: float = 0.0
    ki_max: float = 0.08


@dataclass
class ObjectiveWeights:
    rms_error: float = 1.0
    max_error: float = 0.35
    final_error: float = 0.5
    torque_effort: float = 0.05


@dataclass
class TunePidArgs:
    run: FlybyRunConfig = field(default_factory=FlybyRunConfig)
    bounds: GainBounds = field(default_factory=GainBounds)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    tune_duration_s: float = 1200.0
    maxiter: int = 20
    popsize: int = 8
    seed: int = 13

    output_dir: str = "./outputs"
    run_name: str = "pid_tuning"


def _build_tuning_system(cfg: FlybyRunConfig) -> tuple[NodeSystem, NodeTuneRecorder]:
    gravity = NodeAsteroidGravity(
        asteroid=cfg.asteroid.asteroid,
        lmax=cfg.asteroid.gravity_lmax,
        name="gravity",
    )
    mu = gravity._gravity_model.gm

    r0, v0 = _hyperbolic_state_from_params(
        mu=mu,
        periapsis_radius_m=cfg.flyby.periapsis_radius_m,
        external_angle=cfg.flyby.external_angle_deg,
        true_anomaly_deg=cfg.flyby.true_anomaly0_deg,
        inbound_ra_deg=cfg.flyby.inbound_ra_deg,
        inbound_dec_deg=cfg.flyby.inbound_dec_deg,
        bplane_angle_deg=cfg.flyby.bplane_angle_deg,
    )

    translational = NodeHyperbolicDynamics(np.concatenate([r0, v0]), mu=mu, name="translational")
    guidance = NodeCenterPointingGuidance(name="guidance")
    controller = NodeAttitudeController(
        kp=cfg.controller.kp,
        kd=cfg.controller.kd,
        ki=cfg.controller.ki,
        integral_limit=cfg.controller.integral_limit,
        name="controller",
    )

    allocator = NodeTorqueAllocator(name="allocator")
    wheels = [
        NodeReactionWheel(
            i,
            cfg.rwa,
            TETRAHEDRAL_WHEEL_AXES[i],
            name=f"wheel_{i}",
        )
        for i in range(4)
    ]
    aggregator = NodeWheelAggregator(name="aggregator")

    attitude = NodeAttitudeDynamics(
        inertia_kgm2=cfg.spacecraft.inertia_kgm2,
        x0=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        name="attitude",
    )

    recorder = NodeTuneRecorder(name="tune_recorder")

    system = NodeSystem()
    for n in [gravity, translational, guidance, controller, allocator] + wheels + [aggregator, attitude, recorder]:
        system.add_node(n)

    translational.o.position >> gravity.i.position
    gravity.o.gravity_accel >> translational.i.gravity_accel

    translational.o.position >> guidance.i.position

    guidance.o.q_cmd >> controller.i.q_cmd
    guidance.o.w_cmd >> controller.i.w_cmd
    attitude.o.q >> controller.i.q
    attitude.o.w >> controller.i.w
    aggregator.o.h_rw_total >> controller.i.h_rw

    controller.o.tau_cmd_body >> allocator.i.tau_cmd_body

    for i in range(4):
        allocator.o.tau_cmd_wheel >> wheels[i].i.tau_cmd

    wheels[0].o.h_rw >> aggregator.i.h_rw_0
    wheels[0].o.tau_rw >> aggregator.i.tau_rw_0
    wheels[1].o.h_rw >> aggregator.i.h_rw_1
    wheels[1].o.tau_rw >> aggregator.i.tau_rw_1
    wheels[2].o.h_rw >> aggregator.i.h_rw_2
    wheels[2].o.tau_rw >> aggregator.i.tau_rw_2
    wheels[3].o.h_rw >> aggregator.i.h_rw_3
    wheels[3].o.tau_rw >> aggregator.i.tau_rw_3

    aggregator.o.tau_rw_total >> attitude.i.tau_body
    aggregator.o.h_rw_total >> attitude.i.h_rw_body

    controller.o.q_err >> recorder.i.q_err
    controller.o.tau_cmd_body >> recorder.i.tau_cmd_body

    return system, recorder


def _evaluate_once(cfg: FlybyRunConfig, weights: ObjectiveWeights) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    system, recorder = _build_tuning_system(cfg)
    with TemporaryDirectory(prefix="pid_tune_") as tmp:
        system.simulate(
            t_f=cfg.sim.duration_s,
            dt=cfg.sim.dt_s,
            save_dir=tmp,
            sim_name="pid_tuning_eval",
        )

    if len(recorder.q_err) == 0:
        penalty = 1e6
        return penalty, {
            "qerr_rms": penalty,
            "qerr_peak": penalty,
            "qerr_final": penalty,
            "tau_rms": penalty,
            "objective": penalty,
        }, {
            "t": np.array([]),
            "qerr_mag": np.array([]),
            "tau_mag": np.array([]),
        }

    t = np.array(recorder.t, dtype=float)
    q_err = np.array(recorder.q_err, dtype=float)
    tau_cmd = np.array(recorder.tau_cmd, dtype=float)

    qerr_mag = np.linalg.norm(q_err, axis=1)
    tau_mag = np.linalg.norm(tau_cmd, axis=1)

    qerr_rms = float(np.sqrt(np.mean(qerr_mag**2)))
    qerr_peak = float(np.max(qerr_mag))
    qerr_final = float(qerr_mag[-1])
    tau_rms = float(np.sqrt(np.mean(tau_mag**2)))

    objective = (
        weights.rms_error * qerr_rms
        + weights.max_error * qerr_peak
        + weights.final_error * qerr_final
        + weights.torque_effort * tau_rms
    )

    metrics = {
        "qerr_rms": qerr_rms,
        "qerr_peak": qerr_peak,
        "qerr_final": qerr_final,
        "tau_rms": tau_rms,
        "objective": float(objective),
    }
    traces = {
        "t": t,
        "qerr_mag": qerr_mag,
        "tau_mag": tau_mag,
    }
    return float(objective), metrics, traces


def _save_results(
    output_dir: Path,
    trials: list[dict[str, float]],
    best_cfg: FlybyRunConfig,
    best_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    baseline_traces: dict[str, np.ndarray],
    best_traces: dict[str, np.ndarray],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_path = output_dir / "pid_trials.csv"
    if trials:
        keys = list(trials[0].keys())
        with trials_path.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for tr in trials:
                f.write(",".join(str(tr[k]) for k in keys) + "\n")

    summary_path = output_dir / "pid_best_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Best PID gains\n")
        f.write(f"kp={best_cfg.controller.kp}\n")
        f.write(f"kd={best_cfg.controller.kd}\n")
        f.write(f"ki={best_cfg.controller.ki}\n\n")
        f.write("Baseline metrics\n")
        for k, v in baseline_metrics.items():
            f.write(f"{k}={v}\n")
        f.write("\nBest metrics\n")
        for k, v in best_metrics.items():
            f.write(f"{k}={v}\n")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(baseline_traces["t"], baseline_traces["qerr_mag"], label="baseline", linewidth=1.5)
    axs[0].plot(best_traces["t"], best_traces["qerr_mag"], label="tuned", linewidth=1.5)
    axs[0].set_ylabel("|q_err| [-]")
    axs[0].set_title("Pointing Error Magnitude")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].plot(baseline_traces["t"], baseline_traces["tau_mag"], label="baseline", linewidth=1.5)
    axs[1].plot(best_traces["t"], best_traces["tau_mag"], label="tuned", linewidth=1.5)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("|tau_cmd| [N m]")
    axs[1].set_title("Controller Torque Command Magnitude")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pid_tuning_comparison.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = tyro.cli(TunePidArgs)

    base_cfg = deepcopy(args.run)
    base_cfg.output.render_video = False
    base_cfg.sim.duration_s = args.tune_duration_s

    trials: list[dict[str, float]] = []

    baseline_obj, baseline_metrics, baseline_traces = _evaluate_once(base_cfg, args.weights)
    print("Baseline controller")
    print(
        f"  kp={base_cfg.controller.kp:.6f} kd={base_cfg.controller.kd:.6f} ki={base_cfg.controller.ki:.6f}"
    )
    print(
        "  objective={:.6f} qerr_rms={:.6f} qerr_peak={:.6f} qerr_final={:.6f} tau_rms={:.6f}".format(
            baseline_obj,
            baseline_metrics["qerr_rms"],
            baseline_metrics["qerr_peak"],
            baseline_metrics["qerr_final"],
            baseline_metrics["tau_rms"],
        )
    )

    bounds = [
        (args.bounds.kp_min, args.bounds.kp_max),
        (args.bounds.kd_min, args.bounds.kd_max),
        (args.bounds.ki_min, args.bounds.ki_max),
    ]

    def objective(x: np.ndarray) -> float:
        cfg = deepcopy(base_cfg)
        cfg.controller.kp = float(x[0])
        cfg.controller.kd = float(x[1])
        cfg.controller.ki = float(x[2])

        try:
            obj, metrics, _ = _evaluate_once(cfg, args.weights)
        except Exception:
            obj = 1e6
            metrics = {
                "qerr_rms": 1e6,
                "qerr_peak": 1e6,
                "qerr_final": 1e6,
                "tau_rms": 1e6,
            }

        trials.append(
            {
                "kp": cfg.controller.kp,
                "kd": cfg.controller.kd,
                "ki": cfg.controller.ki,
                "objective": float(obj),
                "qerr_rms": float(metrics["qerr_rms"]),
                "qerr_peak": float(metrics["qerr_peak"]),
                "qerr_final": float(metrics["qerr_final"]),
                "tau_rms": float(metrics["tau_rms"]),
            }
        )
        print(
            "trial {:04d}: kp={:.6f} kd={:.6f} ki={:.6f} obj={:.6f}".format(
                len(trials),
                cfg.controller.kp,
                cfg.controller.kd,
                cfg.controller.ki,
                obj,
            )
        )
        return float(obj)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=True,
        updating="deferred",
        workers=10,
    )

    tuned_cfg = deepcopy(base_cfg)
    tuned_cfg.controller.kp = float(result.x[0])
    tuned_cfg.controller.kd = float(result.x[1])
    tuned_cfg.controller.ki = float(result.x[2])

    best_obj, best_metrics, best_traces = _evaluate_once(tuned_cfg, args.weights)

    out_dir = Path(args.output_dir) / args.run_name
    _save_results(
        output_dir=out_dir,
        trials=trials,
        best_cfg=tuned_cfg,
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        baseline_traces=baseline_traces,
        best_traces=best_traces,
    )

    print("\nBest tuned controller")
    print(f"  kp={tuned_cfg.controller.kp:.6f}")
    print(f"  kd={tuned_cfg.controller.kd:.6f}")
    print(f"  ki={tuned_cfg.controller.ki:.6f}")
    print(
        "  objective={:.6f} qerr_rms={:.6f} qerr_peak={:.6f} qerr_final={:.6f} tau_rms={:.6f}".format(
            best_obj,
            best_metrics["qerr_rms"],
            best_metrics["qerr_peak"],
            best_metrics["qerr_final"],
            best_metrics["tau_rms"],
        )
    )
    print(f"\nSaved tuning results in: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
