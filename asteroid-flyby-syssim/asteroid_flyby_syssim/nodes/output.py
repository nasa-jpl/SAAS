"""Output recorder and animation nodes for asteroid flyby simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from syssim.core import EmptySpec, InputPort, Node, input_port


@dataclass
class NodeTrajectoryAnimatorInputs:
    """Input ports for trajectory animation."""

    position: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft position relative to asteroid center [m]."""
    look_vec: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Camera boresight unit vector in inertial coordinates."""


class NodeTrajectoryAnimator(Node[NodeTrajectoryAnimatorInputs, EmptySpec, EmptySpec, EmptySpec]):
    """Collect trajectory samples and save a 3D look-vector GIF on finalize."""

    Inputs = NodeTrajectoryAnimatorInputs

    def __init__(self, output_path: Path, fps: float, **kwargs):
        """Initialize trajectory animation collection.

        Parameters
        ----------
        output_path : pathlib.Path
            GIF path written during finalization.
        fps : float
            Output animation frame rate [frames/s].
        **kwargs
            Additional keyword arguments forwarded to ``Node``.
        """
        self._output_path = Path(output_path)
        self._fps = fps
        super().__init__(**kwargs)
        self._i = self.i

    def initialize(self):
        """Clear collected trajectory and look-vector samples."""
        self._t: list[float] = []
        self._r: list[np.ndarray] = []
        self._look: list[np.ndarray] = []

    def update(self, sim_time: float):
        """Record the current trajectory sample when inputs are available.

        Parameters
        ----------
        sim_time : float
            Current simulation time [s].
        """
        r = self.i.position.read().value
        look = self.i.look_vec.read().value
        if r is None or look is None:
            return
        self._t.append(sim_time)
        self._r.append(np.array(r, dtype=float))
        self._look.append(np.array(look, dtype=float))

    def finalize(self, fault_history=None):
        """Save collected trajectory samples as a GIF.

        Parameters
        ----------
        fault_history : object, optional
            Fault history forwarded by the syssim runtime.
        """
        if not self._r:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        r = np.array(self._r)
        look = np.array(self._look)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        lim = max(np.max(np.linalg.norm(r, axis=1)) * 1.1, 1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Flyby Trajectory and Camera Look Vector")
        ax.scatter([0.0], [0.0], [0.0], s=180, c="tab:brown", label="Asteroid Center")

        (traj_line,) = ax.plot([], [], [], c="tab:blue", lw=1.5, label="Spacecraft Trajectory")
        point = ax.scatter([], [], [], c="tab:red", s=20)
        quiv = None

        def _update(frame_idx: int):
            """Update the trajectory animation artists for one frame.

            Parameters
            ----------
            frame_idx : int
                Index of the frame to draw.

            Returns
            -------
            tuple
                Matplotlib artists updated for the current frame.
            """
            nonlocal quiv
            rr = r[: frame_idx + 1]
            traj_line.set_data(rr[:, 0], rr[:, 1])
            traj_line.set_3d_properties(rr[:, 2])
            cur = r[frame_idx]
            point._offsets3d = ([cur[0]], [cur[1]], [cur[2]])
            if quiv is not None:
                quiv.remove()
            arrow_scale = max(np.linalg.norm(cur) * 0.2, 1.0)
            lv = look[frame_idx] * arrow_scale
            quiv = ax.quiver(cur[0], cur[1], cur[2], lv[0], lv[1], lv[2], color="tab:green", linewidth=2)
            return traj_line, point

        ani = animation.FuncAnimation(fig, _update, frames=len(r), interval=1000.0 / max(self._fps, 1.0), blit=False)
        writer = animation.PillowWriter(fps=max(self._fps, 1.0))
        ani.save(self._output_path, writer=writer)
        plt.close(fig)


@dataclass
class NodeDataRecorderInputs:
    """Input ports for flyby diagnostic time-series recording."""

    position: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft position relative to asteroid center [m]."""
    velocity: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft velocity relative to asteroid center [m/s]."""
    q: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(4,))
    """Spacecraft attitude quaternion [w, x, y, z]."""
    w: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spacecraft body angular velocity [rad/s]."""
    q_err: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Attitude error vector from the controller."""
    tau_cmd_body: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Commanded body torque [N m]."""
    gravity_error: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(3,))
    """Spherical-harmonic minus point-mass gravity acceleration [m/s^2]."""
    wheel_speed_0: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(1,))
    """Wheel 0 angular speed [rad/s]."""
    wheel_speed_1: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(1,))
    """Wheel 1 angular speed [rad/s]."""
    wheel_speed_2: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(1,))
    """Wheel 2 angular speed [rad/s]."""
    wheel_speed_3: InputPort[np.ndarray] = input_port(np.ndarray, dtype=float, shape=(1,))
    """Wheel 3 angular speed [rad/s]."""
    wheel_torque_0: InputPort[float] = input_port(float)
    """Wheel 0 effective scalar torque [N m]."""
    wheel_torque_1: InputPort[float] = input_port(float)
    """Wheel 1 effective scalar torque [N m]."""
    wheel_torque_2: InputPort[float] = input_port(float)
    """Wheel 2 effective scalar torque [N m]."""
    wheel_torque_3: InputPort[float] = input_port(float)
    """Wheel 3 effective scalar torque [N m]."""


class NodeDataRecorder(Node[NodeDataRecorderInputs, EmptySpec, EmptySpec, EmptySpec]):
    """Record flyby state, control, gravity, and wheel diagnostics to CSV/NPZ."""

    Inputs = NodeDataRecorderInputs

    def __init__(self, output_prefix: Path, **kwargs):
        """Initialize time-series data recording.

        Parameters
        ----------
        output_prefix : pathlib.Path
            Output path prefix used for CSV, NPZ, and plot artifacts.
        **kwargs
            Additional keyword arguments forwarded to ``Node``.
        """
        self._output_prefix = Path(output_prefix)
        super().__init__(**kwargs)
        self._i = self.i

    def initialize(self):
        """Clear recorded diagnostic rows before simulation."""
        self._rows: list[dict[str, float]] = []

    def update(self, sim_time: float):
        """Record one diagnostic row when all inputs are available.

        Parameters
        ----------
        sim_time : float
            Current simulation time [s].
        """
        values = {name: getattr(self.i, name).read().value for name in self.i.__dataclass_fields__}
        if any(value is None for value in values.values()):
            return
        pos = values["position"]
        vel = values["velocity"]
        q = values["q"]
        w = values["w"]
        q_err = values["q_err"]
        tau_cmd = values["tau_cmd_body"]
        g_err = values["gravity_error"]
        ws = [values[f"wheel_speed_{idx}"] for idx in range(4)]
        wt = [values[f"wheel_torque_{idx}"] for idx in range(4)]
        self._rows.append(
            {
                "t": sim_time,
                "rx": float(pos[0]),
                "ry": float(pos[1]),
                "rz": float(pos[2]),
                "vx": float(vel[0]),
                "vy": float(vel[1]),
                "vz": float(vel[2]),
                "qw": float(q[0]),
                "qx": float(q[1]),
                "qy": float(q[2]),
                "qz": float(q[3]),
                "wx": float(w[0]),
                "wy": float(w[1]),
                "wz": float(w[2]),
                "qerrx": float(q_err[0]),
                "qerry": float(q_err[1]),
                "qerrz": float(q_err[2]),
                "tau_cmd_x": float(tau_cmd[0]),
                "tau_cmd_y": float(tau_cmd[1]),
                "tau_cmd_z": float(tau_cmd[2]),
                "g_err_x": float(g_err[0]),
                "g_err_y": float(g_err[1]),
                "g_err_z": float(g_err[2]),
                "ws0": float(ws[0][0]),
                "ws1": float(ws[1][0]),
                "ws2": float(ws[2][0]),
                "ws3": float(ws[3][0]),
                "wt0": float(wt[0]),
                "wt1": float(wt[1]),
                "wt2": float(wt[2]),
                "wt3": float(wt[3]),
            }
        )

    def _plot_components_with_magnitude(self, t, components, labels, title, ylabel, output_path):
        """Plot vector components and magnitude to a file.

        Parameters
        ----------
        t : np.ndarray
            Time samples [s].
        components : np.ndarray
            Vector component samples with shape ``(N, 3)``.
        labels : tuple[str, str, str]
            Labels for the three component traces.
        title : str
            Figure title.
        ylabel : str
            Y-axis label.
        output_path : pathlib.Path
            Path where the PNG figure is written.
        """
        mag = np.linalg.norm(components, axis=1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, components[:, 0], label=labels[0])
        ax.plot(t, components[:, 1], label=labels[1])
        ax.plot(t, components[:, 2], label=labels[2])
        ax.plot(t, mag, "k--", linewidth=1.5, label="magnitude")
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _save_diagnostic_plots(self, arrs: dict[str, np.ndarray]):
        """Write diagnostic plots from recorded time-series arrays.

        Parameters
        ----------
        arrs : dict[str, np.ndarray]
            Recorded columns keyed by CSV/NPZ field name.
        """
        t = arrs["t"]
        wheel_speeds = np.column_stack([arrs["ws0"], arrs["ws1"], arrs["ws2"], arrs["ws3"]])
        wheel_torques = np.column_stack([arrs["wt0"], arrs["wt1"], arrs["wt2"], arrs["wt3"]])

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for idx in range(4):
            axs[0].plot(t, wheel_speeds[:, idx], label=f"wheel {idx}")
        axs[0].plot(t, np.linalg.norm(wheel_speeds, axis=1), "k--", linewidth=1.5, label="magnitude")
        axs[0].set_ylabel("Speed [rad/s]")
        axs[0].set_title("Reaction Wheel Speeds")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="best")
        for idx in range(4):
            axs[1].plot(t, wheel_torques[:, idx], label=f"wheel {idx}")
        axs[1].plot(t, np.linalg.norm(wheel_torques, axis=1), "k--", linewidth=1.5, label="magnitude")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Torque [N m]")
        axs[1].set_title("Reaction Wheel Torques")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="best")
        fig.tight_layout()
        fig.savefig(self._output_prefix.parent / "reaction_wheels.png", dpi=150)
        plt.close(fig)

        self._plot_components_with_magnitude(
            t,
            np.column_stack([arrs["tau_cmd_x"], arrs["tau_cmd_y"], arrs["tau_cmd_z"]]),
            ("tau_cmd_x", "tau_cmd_y", "tau_cmd_z"),
            "Controller Torque Command",
            "Torque [N m]",
            self._output_prefix.parent / "controller_torque_command.png",
        )
        self._plot_components_with_magnitude(
            t,
            np.column_stack([arrs["qerrx"], arrs["qerry"], arrs["qerrz"]]),
            ("qerr_x", "qerr_y", "qerr_z"),
            "Controller Pointing Error",
            "Quaternion Error [-]",
            self._output_prefix.parent / "controller_pointing_error.png",
        )
        self._plot_components_with_magnitude(
            t,
            np.column_stack([arrs["g_err_x"], arrs["g_err_y"], arrs["g_err_z"]]),
            ("g_err_x", "g_err_y", "g_err_z"),
            "Gravity Model Error vs Keplerian",
            "Acceleration Error [m/s^2]",
            self._output_prefix.parent / "gravity_error.png",
        )

    def finalize(self, fault_history=None):
        """Write recorded diagnostics to CSV, NPZ, and plot files.

        Parameters
        ----------
        fault_history : object, optional
            Fault history forwarded by the syssim runtime.
        """
        if not self._rows:
            return
        self._output_prefix.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._rows[0].keys())
        csv_path = self._output_prefix.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8") as stream:
            stream.write(",".join(keys) + "\n")
            for row in self._rows:
                stream.write(",".join(str(row[key]) for key in keys) + "\n")
        arrs = {key: np.array([row[key] for row in self._rows], dtype=float) for key in keys}
        np.savez(self._output_prefix.with_suffix(".npz"), **arrs)
        self._save_diagnostic_plots(arrs)


@dataclass
class NodeFrameCollectorInputs:
    """Input ports for rendered camera frame collection."""

    image: InputPort[np.ndarray] = input_port(np.ndarray)
    """Rendered RGB image frame."""
    mask: InputPort[np.ndarray] = input_port(np.ndarray)
    """Rendered asteroid mask frame."""


class NodeFrameCollector(Node[NodeFrameCollectorInputs, EmptySpec, EmptySpec, EmptySpec]):
    """Collect rendered image/mask frames and write GIFs on finalize."""

    Inputs = NodeFrameCollectorInputs

    def __init__(self, output_path: Path, output_mask_path: Path | None = None, fps: float = 20, **kwargs):
        """Initialize rendered image and mask collection.

        Parameters
        ----------
        output_path : pathlib.Path
            GIF path for rendered image frames.
        output_mask_path : pathlib.Path, optional
            GIF path for rendered mask frames.
        fps : float, optional
            Output animation frame rate [frames/s].
        **kwargs
            Additional keyword arguments forwarded to ``Node``.
        """
        self._output_path = Path(output_path)
        self._output_mask_path = Path(output_mask_path) if output_mask_path is not None else None
        self._fps = fps
        super().__init__(**kwargs)
        self._i = self.i

    def initialize(self):
        """Clear collected image and mask frames."""
        self._image_frames: list[np.ndarray] = []
        self._mask_frames: list[np.ndarray] = []

    def update(self, sim_time: float):
        """Collect current image and mask frames when available.

        Parameters
        ----------
        sim_time : float
            Current simulation time [s].
        """
        del sim_time
        image = self.i.image.read().value
        mask = self.i.mask.read().value
        if image is not None:
            self._image_frames.append(np.asarray(image, dtype=np.uint8))
        if mask is not None:
            self._mask_frames.append(np.asarray(mask, dtype=np.uint8))

    def finalize(self, fault_history=None):
        """Write collected image and mask GIFs.

        Parameters
        ----------
        fault_history : object, optional
            Fault history forwarded by the syssim runtime.
        """
        if self._image_frames:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(self._output_path, self._image_frames, duration=1.0 / max(self._fps, 1e-6), loop=0)
        if self._mask_frames and self._output_mask_path is not None:
            self._output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(self._output_mask_path, self._mask_frames, duration=1.0 / max(self._fps, 1e-6), loop=0)
