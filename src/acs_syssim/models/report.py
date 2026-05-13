from typing import NamedTuple
import pickle
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os
import subprocess
import shutil
import tempfile
import platform
from scipy.spatial.transform import Rotation as R
from textwrap import wrap

from syssim import Node, InputPort
from tqdm import tqdm
# matplotlib.use("Agg")  # Use Agg backend for PNG output
# matplotlib.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.size": 11,
#     }
# )

# Use a seaborn theme appropriate for conference paper figures.
# 'context="paper"' yields compact, publication-ready elements; 'whitegrid' keeps subtle grid lines.
sns.set_theme(context="paper", style="whitegrid", font="serif", rc={"font.size": 11})


class ReportInputs(NamedTuple):
    w_cmd: InputPort
    q_cmd: InputPort
    w_true: InputPort
    q_true: InputPort
    w_est: InputPort
    q_est: InputPort
    torque_cmd: InputPort
    rw1_cmd: InputPort
    rw2_cmd: InputPort
    rw3_cmd: InputPort
    rw4_cmd: InputPort
    rw5_cmd: InputPort
    rw6_cmd: InputPort
    rw7_cmd: InputPort
    rw8_cmd: InputPort
    health: InputPort
    mode: InputPort


class Reporter(Node):

    def __init__(
        self,
        **kwargs,
    ):

        self._i = ReportInputs(
            InputPort("w_cmd", self),
            InputPort("q_cmd", self),
            InputPort("w_true", self),
            InputPort("q_true", self),
            InputPort("w_est", self),
            InputPort("q_est", self),
            InputPort("torque_cmd", self),
            InputPort("rw1_cmd", self),
            InputPort("rw2_cmd", self),
            InputPort("rw3_cmd", self),
            InputPort("rw4_cmd", self),
            InputPort("rw5_cmd", self),
            InputPort("rw6_cmd", self),
            InputPort("rw7_cmd", self),
            InputPort("rw8_cmd", self),
            InputPort("health", self),
            InputPort("mode", self)
        )

        # Initialize data buffers
        self._buffers = {
            "w_cmd": [],
            "q_cmd": [],
            "w_true": [],
            "q_true": [],
            "w_est": [],
            "q_est": [],
            "torque_cmd": [],
            "rw1_cmd": [],
            "rw2_cmd": [],
            "rw3_cmd": [],
            "rw4_cmd": [],
            "rw5_cmd": [],
            "rw6_cmd": [],
            "rw7_cmd": [],
            "rw8_cmd": [],
            "health": [],
            "mode": [],
            "time": [],
        }

        super().__init__(
            self._i,
            (),
            **kwargs,
        )

    def initialize(self):
        return super().initialize()

    def update(self, sim_time):
        # Buffer all input port data
        self._buffers["time"].append(sim_time)
        self._buffers["w_cmd"].append(self._i.w_cmd.read())
        self._buffers["q_cmd"].append(self._i.q_cmd.read())
        self._buffers["w_true"].append(self._i.w_true.read())
        self._buffers["q_true"].append(self._i.q_true.read())
        self._buffers["w_est"].append(self._i.w_est.read())
        self._buffers["q_est"].append(self._i.q_est.read())
        self._buffers["torque_cmd"].append(self._i.torque_cmd.read())
        self._buffers["rw1_cmd"].append(self._i.rw1_cmd.read())
        self._buffers["rw2_cmd"].append(self._i.rw2_cmd.read())
        self._buffers["rw3_cmd"].append(self._i.rw3_cmd.read())
        self._buffers["rw4_cmd"].append(self._i.rw4_cmd.read())
        self._buffers["rw5_cmd"].append(self._i.rw5_cmd.read())
        self._buffers["rw6_cmd"].append(self._i.rw6_cmd.read())
        self._buffers["rw7_cmd"].append(self._i.rw7_cmd.read())
        self._buffers["rw8_cmd"].append(self._i.rw8_cmd.read())
        self._buffers["health"].append(self._i.health.read())
        self._buffers["mode"].append(self._i.mode.read())

        return super().update(sim_time)

    def finalize(self, fault_history: dict[float, dict[str, bool]] = None):
        # Helper function to convert buffered data to numpy array, filtering out None values
        def _buffer_to_array(buffer_data):
            if not buffer_data:
                return None
            # Filter out None values and convert to numpy array
            valid_data = [d for d in buffer_data if d is not None]
            if not valid_data:
                return None
            return np.array(valid_data)

        # Extract data from buffers
        time = np.array(self._buffers["time"])
        w_cmd = _buffer_to_array(self._buffers["w_cmd"])
        w_true = _buffer_to_array(self._buffers["w_true"])
        q_cmd = _buffer_to_array(self._buffers["q_cmd"])
        q_true = _buffer_to_array(self._buffers["q_true"])
        w_est = _buffer_to_array(self._buffers["w_est"])
        q_est = _buffer_to_array(self._buffers["q_est"])
        health = self._buffers["health"]  # Keep as list for fault processing
        rw1_cmd = _buffer_to_array(self._buffers["rw1_cmd"])
        rw2_cmd = _buffer_to_array(self._buffers["rw2_cmd"])
        rw3_cmd = _buffer_to_array(self._buffers["rw3_cmd"])
        rw4_cmd = _buffer_to_array(self._buffers["rw4_cmd"])
        rw5_cmd = _buffer_to_array(self._buffers["rw5_cmd"])
        rw6_cmd = _buffer_to_array(self._buffers["rw6_cmd"])
        rw7_cmd = _buffer_to_array(self._buffers["rw7_cmd"])
        rw8_cmd = _buffer_to_array(self._buffers["rw8_cmd"])
        mode = self._buffers["mode"]

        # Create output directory
        out_dir = os.path.abspath(os.path.join("/tmp", "reports"))
        os.makedirs(out_dir, exist_ok=True)
        base_name = (self.name or "report").replace(" ", "_")
        
        # Calculate fault detection latencies
        latency_info = self._calculate_fault_detection_latencies(health, fault_history, time)
        
        # Save all plot data
        plot_data = {
            'time': time,
            'w_cmd': w_cmd,
            'w_true': w_true,
            'q_cmd': q_cmd,
            'q_true': q_true,
            'w_est': w_est,
            'q_est': q_est,
            'health': health,
            'rw_cmds': [rw1_cmd, rw2_cmd, rw3_cmd, rw4_cmd, rw5_cmd, rw6_cmd, rw7_cmd, rw8_cmd],
            'mode': mode,
            'fault_history': fault_history or {},
            'base_name': base_name
        }
        
        data_file_path = os.path.join(out_dir, f"{base_name}_plot_data.pkl")
        with open(data_file_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"Reporter.finalize: saved plot data to {data_file_path}")

        # Generate plots
        self._generate_all_plots(plot_data, out_dir, latency_info)

        return super().finalize()

    def _calculate_fault_detection_latencies(self, health: list, fault_history: dict, time: np.ndarray):
        """
        Calculate detection latency for each fault by comparing when it was injected
        (fault_history) to when it was first detected (health).
        Returns a dict mapping fault_name -> list of (injection_time, detection_time, latency) tuples.
        
        Specialized mapping:
        - 'Gyro 1 Bias Creep' -> 'Gyro_1' in health
        - 'SRU 1 Bias Creep' -> 'SRU_1' in health
        - 'Encoder 4 Random Noise' -> 'ENC_4' or 'RWA_4' in health (whichever detected first)
        """
        if not fault_history or not health:
            return {}
        
        # Define fault-to-health component mapping
        fault_to_health_map = {
            'Gyro 1 Bias Creep': ['IMU_1'],
            'SRU 1 Bias Creep': ['SRU_1'],
            'Encoder 4 Random Noise': ['ENC_4', 'RWA_4'],
        }
        
        latency_info = {}

        # Make a list of tuples (time, health_dict, fault_dict)
        health_data = [(time[i], health[i], fault_history.get(time[i], {})) for i in range(len(time))]
        
        for map in fault_to_health_map.items():
            key, val = map
            fault_start_time = None
            fault_detected_time = None
            for t, health_dict, fault_dict in health_data:
                is_injected = fault_dict.get(key, False)
                # Check for injection
                if is_injected and fault_start_time is None:
                    fault_start_time = t  # Fault injection time
                # Check for detection
                if fault_start_time is not None and fault_detected_time is None:
                    # Check if any mapped health components are faulty
                    for health_component in val:
                        if health_dict.get(health_component) == "Faulty":
                            fault_detected_time = t  # Fault detection time
                            break
                if fault_start_time is not None and fault_detected_time is not None:
                    latency = fault_detected_time - fault_start_time
                    if latency >= 0:  # Only count positive latencies
                        if key not in latency_info:
                            latency_info[key] = []
                        latency_info[key].append((fault_start_time, fault_detected_time, latency))
                    break
                    
        # # Get sorted times from fault history
        # fault_times = sorted(fault_history.keys())
        
        # # For each fault, track its injection and detection
        # for fault_name in set().union(*fault_history.values()):
        #     latencies = []
            
        #     # Get the health components to look for
        #     health_components = fault_to_health_map.get(fault_name, [fault_name])
            
        #     # Find each injection event
        #     fault_active = False
        #     injection_time = None
            
        #     for time_idx, fault_time in enumerate(fault_times):
        #         is_injected = fault_history[fault_time].get(fault_name, False)
                
        #         if is_injected and not fault_active:
        #             # Rising edge: fault just became active
        #             fault_active = True
        #             injection_time = fault_time
        #         elif not is_injected and fault_active:
        #             # Falling edge: fault just became inactive
        #             fault_active = False
        #             injection_time = None
                
        #         # If fault is currently active, check if it's detected
        #         if fault_active and injection_time is not None:
        #             # Look in health data for when this fault was first detected
        #             for health_idx, health_dict in enumerate(health):
        #                 if health_dict is None:
        #                     continue
                        
        #                 # Check if any of the mapped health components is "Faulty"
        #                 detected = False
        #                 for health_component in health_components:
        #                     if health_dict.get(health_component) == "Faulty":
        #                         detected = True
        #                         break
                        
        #                 if detected:
        #                     # Fault was detected at this health index
        #                     health_time = time[health_idx] if health_idx < len(time) else time[-1]
        #                     latency = health_time - injection_time
        #                     if latency >= 0:  # Only count positive latencies
        #                         latencies.append((injection_time, health_time, latency))
        #                     break
            
        #     if latencies:
        #         latency_info[fault_name] = latencies
        
        return latency_info

    def _generate_all_plots(self, plot_data: dict, out_dir: str, latency_info: dict = None):
        """Generate all plots from plot data dictionary."""
        # Extract data
        time = plot_data['time']
        w_cmd = plot_data['w_cmd']
        w_true = plot_data['w_true']
        q_cmd = plot_data['q_cmd']
        q_true = plot_data['q_true']
        w_est = plot_data['w_est']
        q_est = plot_data['q_est']
        health = plot_data['health']
        rw_cmds = plot_data['rw_cmds']
        mode = plot_data['mode']
        fault_history = plot_data['fault_history']
        base_name = plot_data['base_name']

        # Save as PNG
        png_path = os.path.join(out_dir, f"{base_name}_errors.png")
        rw_png_path = os.path.join(out_dir, f"{base_name}_rw_cmds.png")
        est_png_path = os.path.join(out_dir, f"{base_name}_estimator.png")
        combined_png_path = os.path.join(out_dir, f"{base_name}_timeline.png")

        # GEN PLOTS
        self._plot_tracking_error(
            q_cmd, q_true, w_cmd, w_true, time, out_dir, base_name
        )
        self._plot_rw_cmd(
            *rw_cmds, time, out_dir, base_name
        )
        self._plot_estimator_outputs(
            w_est, q_est, w_true, q_true, time, out_dir, base_name
        )
        self._plot_combined_timeline(health, mode, fault_history, time, out_dir, base_name)

        # Create LaTeX document with latency information
        self._generate_latex_report(out_dir, base_name, latency_info)

    def _generate_latex_report(self, out_dir: str, base_name: str, latency_info: dict = None):
        """Generate LaTeX report with optional latency table."""
        # Create latency table if available
        latency_table = ""
        if latency_info:
            latency_table = self._create_latency_table(latency_info)
            # Print latency summary to console
            self._print_latency_summary(latency_info)
        
        tex_content = r"""\documentclass[11pt]{article}
    \usepackage[margin=1in]{geometry}
    \usepackage{graphicx}
    \usepackage{amsmath}
    \usepackage{float}
    \usepackage{booktabs}

    \begin{document}

    \title{Simulation Report}
    \date{\today}
    \maketitle

    """ + latency_table + r"""

    \section{Command Tracking Performance}

    The following figure shows the magnitude of errors between commanded and actual angular rates and orientations throughout the simulation.

    \begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{%s}
    \caption{Rate and orientation tracking errors. The blue solid line shows the magnitude of angular rate error, while the red dashed line shows the orientation error in radians.}
    \label{fig:tracking_errors}
    \end{figure}

    \section{Estimator Performance}

    The following figure shows the estimator outputs compared to the true values for angular rates and orientation (converted to Euler angles).

    \begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{%s}
    \caption{Estimator outputs vs. true values. Top plots show angular rate estimates vs. true rates in each axis. Bottom plots show orientation estimates vs. true orientation converted to Euler angles (roll, pitch, yaw).}
    \label{fig:estimator}
    \end{figure}

    \section{System Timeline}

    The following figure shows a comprehensive timeline including autonomy mode transitions, detected faults, and injected faults throughout the simulation.

    \begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{%s}
    \caption{Combined timeline showing autonomy modes, detected faults, and injected faults over the simulation duration.}
    \label{fig:timeline}
    \end{figure}

    \section{Reaction Wheel Commands}

    The following figure shows the torque commands sent to each of the eight reaction wheels throughout the simulation.

    \begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{%s}
    \caption{Reaction wheel torque commands for all eight wheels over the simulation duration.}
    \label{fig:rw_commands}
    \end{figure}

    \end{document}
    """ % (os.path.basename(os.path.join(out_dir, f"{base_name}_errors.png")), 
          os.path.basename(os.path.join(out_dir, f"{base_name}_estimator.png")), 
          os.path.basename(os.path.join(out_dir, f"{base_name}_timeline.png")), 
          os.path.basename(os.path.join(out_dir, f"{base_name}_rw_cmds.png")))

        tex_path = os.path.join(out_dir, f"{base_name}_report.tex")
        with open(tex_path, "w") as f:
            f.write(tex_content)

        # Compile with pdflatex
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            print(
                "Reporter.finalize: 'pdflatex' not found; PDF not compiled. Files saved in",
                out_dir,
            )
            return

        try:
            # Run pdflatex twice for proper references
            for _ in range(2):
                result = subprocess.run(
                    [
                        pdflatex,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        os.path.basename(tex_path),
                    ],
                    cwd=out_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )

            pdf_path = os.path.join(out_dir, f"{base_name}_report.pdf")
            print(f"Reporter: compiled PDF to {pdf_path}")
        except subprocess.CalledProcessError as e:
            print("Reporter: pdflatex failed:")
            print(e.stdout)
            print(e.stderr)
            return

        # Open PDF with system default viewer
        try:
            if platform.system() == "Linux":
                subprocess.Popen(["xdg-open", pdf_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", pdf_path])
            elif platform.system() == "Windows":
                os.startfile(pdf_path)
            print(f"Reporter: opened PDF: {pdf_path}")
        except Exception as e:
            print(f"Reporter: could not open PDF automatically: {e}")
            print(f"PDF available at: {pdf_path}")

    def _create_latency_table(self, latency_info: dict) -> str:
        """Create a LaTeX table for fault detection latencies."""
        if not latency_info:
            return ""
        
        table_rows = []
        for fault_name in sorted(latency_info.keys()):
            latencies = latency_info[fault_name]
            if latencies:
                # Use the first detection latency for the table
                injection_time, detection_time, latency = latencies[0]
                table_rows.append(
                    f"{fault_name} & {injection_time:.3f} s & {detection_time:.3f} s & {latency:.3f} s \\\\"
                )
        
        if not table_rows:
            return ""
        
        table = r"""\section{Fault Detection Latency}

    The following table summarizes the detection latency for each injected fault, defined as the time between when the fault was injected and when it was first detected by the health monitoring system.

    \begin{table}[H]
    \centering
    \begin{tabular}{lrrl}
    \toprule
    \textbf{Fault Name} & \textbf{Injection Time (s)} & \textbf{Detection Time (s)} & \textbf{Latency (s)} \\
    \midrule
    """ + "\n    ".join(table_rows) + r"""
    \bottomrule
    \end{tabular}
    \caption{Fault detection latencies showing the delay between fault injection and detection.}
    \label{tab:latencies}
    \end{table}

    """
        return table

    def _print_latency_summary(self, latency_info: dict):
        """Print fault detection latencies to console."""
        if not latency_info:
            print("Reporter: No fault detection latencies to report")
            return
        
        print("\n" + "="*70)
        print("FAULT DETECTION LATENCY SUMMARY")
        print("="*70)
        print(f"{'Fault Name':<30} {'Injection (s)':<15} {'Detection (s)':<15} {'Latency (s)':<15}")
        print("-"*70)
        
        for fault_name in sorted(latency_info.keys()):
            latencies = latency_info[fault_name]
            for injection_time, detection_time, latency in latencies:
                print(f"{fault_name:<30} {injection_time:<15.3f} {detection_time:<15.3f} {latency:<15.3f}")
        
        print("="*70 + "\n")

    # Defining methods to make the plots we would like to and save them.

    def _plot_tracking_error(
        self,
        cmd_q: np.ndarray,
        true_q: np.ndarray,
        cmd_w: np.ndarray,
        true_w: np.ndarray,
        time: np.ndarray,
        out_dir: str,
        base_name: str,
    ):
        # Ensure inputs exist
        if cmd_w is None or true_w is None or cmd_q is None or true_q is None or time is None:
            print("Reporter: insufficient data for tracking error plot, skipping")
            return

        # Calculate the error vectors
        we = np.linalg.norm(cmd_w - true_w, axis=1)

        # Orientation error
        rot_cmd = R.from_quat(cmd_q)
        rot_true = R.from_quat(true_q)
        err_q = rot_cmd.inv() * rot_true
        qe = err_q.magnitude()

        plt.switch_backend("Agg")
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(4, 3))

        # Plot both errors on the same y-axis
        line1 = ax.plot(time, we, color="tab:blue", label="Rate Error [rad/s]", linewidth=1.5)
        line2 = ax.plot(time, qe, color="tab:red", linestyle="--", label="Orientation Error [rad]", linewidth=1.5)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error")  # generic y-label; units moved to legend
        ax.grid(True, alpha=0.3)

        # Legend with units included in labels
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        legend = ax.legend(lines, labels, loc="upper right")
        legend.set_zorder(10)

        plt.title("Command Tracking Errors")
        plt.tight_layout()

        # Save as PNG
        png_path = os.path.join(out_dir, f"{base_name}_errors.png")
        try:
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Reporter.finalize: saved PNG to {png_path}")
        except Exception as e:
            print(f"Reporter.finalize: failed to save PNG: {e}")
            return super().finalize()

    def _plot_rw_cmd(
        self,
        rw1: np.ndarray,
        rw2: np.ndarray,
        rw3: np.ndarray,
        rw4: np.ndarray,
        rw5: np.ndarray,
        rw6: np.ndarray,
        rw7: np.ndarray,
        rw8: np.ndarray,
        time: np.ndarray,
        out_dir: str,
        base_name: str,
    ):
        plt.switch_backend("Agg")
        sns.set_style("whitegrid")

        # Each rwX is (N,3); reduce to (N,) by norm along axis=1
        rw_arrs = [rw1, rw2, rw3, rw4, rw5, rw6, rw7, rw8]
        rw_mags = [np.linalg.norm(rw, axis=1) for rw in rw_arrs]  # list of (N,)
        rw_mat = np.column_stack(rw_mags)  # (N,8)

        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(8):
            ax.plot(time, rw_mat[:, i], label=f"RW {i+1}", linewidth=1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Torque Magnitude [Nm]")
        ax.set_title("Reaction Wheel Torque Commands")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{base_name}_rw_cmds.png")
        try:
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Reporter.finalize: saved PNG to {png_path}")
        except Exception as e:
            print(f"Reporter.finalize: failed to save PNG: {e}")
            return super().finalize()

    def _plot_estimator_outputs(
        self,
        w_est: np.ndarray,
        q_est: np.ndarray,
        w_true: np.ndarray,
        q_true: np.ndarray,
        time: np.ndarray,
        out_dir: str,
        base_name: str,
    ):
        if w_est is None and q_est is None:
            print("Reporter: No estimator data available, skipping estimator plot")
            return
            
        plt.switch_backend("Agg")
        sns.set_style("whitegrid")

        # Create subplots: 2 rows, 3 columns (top row for w, bottom row for q as euler)
        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
        fig.suptitle('Estimator Performance', fontsize=14)

        # Plot angular rates if available
        if w_est is not None and w_true is not None:
            w_est = np.atleast_2d(w_est)
            w_true = np.atleast_2d(w_true)
            min_len = min(len(w_est), len(w_true), len(time))
            
            labels = ['X', 'Y', 'Z']
            for i in range(3):
                ax = axes[0, i]
                if i < w_est.shape[1] and i < w_true.shape[1]:
                    ax.plot(time[:min_len], w_true[:min_len, i], 'b-', label='True', linewidth=1.5)
                    ax.plot(time[:min_len], w_est[:min_len, i], 'r--', label='Estimated', linewidth=1.5)
                    ax.set_ylabel(f'$\\omega_{{{labels[i]}}}$ [rad/s]')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                if i == 1:  # middle plot
                    ax.set_title('Angular Rate Estimates')
        else:
            # Hide angular rate plots if no data
            for i in range(3):
                axes[0, i].set_visible(False)

        # Plot orientation as Euler angles if available
        if q_est is not None and q_true is not None:
            q_est = np.atleast_2d(q_est)
            q_true = np.atleast_2d(q_true)
            
            if q_est.shape[1] >= 4 and q_true.shape[1] >= 4:
                min_len = min(len(q_est), len(q_true), len(time))
                
                # Convert quaternions to Euler angles using scipy
                rot_est = R.from_quat(q_est[:min_len, :4])
                rot_true = R.from_quat(q_true[:min_len, :4])
                
                euler_est = rot_est.as_euler('xyz', degrees=True)  # Roll, Pitch, Yaw in degrees
                euler_true = rot_true.as_euler('xyz', degrees=True)
                
                labels = ['Roll', 'Pitch', 'Yaw']
                for i in range(3):
                    ax = axes[1, i]
                    ax.plot(time[:min_len], euler_true[:, i], 'b-', label='True', linewidth=1.5)
                    ax.plot(time[:min_len], euler_est[:, i], 'r--', label='Estimated', linewidth=1.5)
                    ax.set_ylabel(f'{labels[i]} [deg]')
                    ax.set_xlabel('Time [s]')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    if i == 1:  # middle plot
                        ax.set_title('Orientation Estimates (Euler Angles)')
            else:
                # Hide quaternion plots if insufficient data
                for i in range(3):
                    axes[1, i].set_visible(False)
        else:
            # Hide quaternion plots if no data
            for i in range(3):
                axes[1, i].set_visible(False)

        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{base_name}_estimator.png")
        try:
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Reporter.finalize: saved estimator PNG to {png_path}")
        except Exception as e:
            print(f"Reporter.finalize: failed to save estimator PNG: {e}")

    def _process_fault_detections(self, health: list[dict], time: np.ndarray):
        """
        Take a list of health dicts at each timestep and return a list of detected faults,
        start times and durations (rising to falling edge).
        The dictionaries in the input list contain keys that are the str names of components
        and a string value: 'Healthy' or 'Faulty'.
        """
        detected_faults = []
        for i in range(1, len(health)):
            for component, status in health[i].items():
                prev_status = health[i - 1].get(component, "Healthy")
                # Detect rising edge: Healthy -> Faulty
                if status == "Faulty" and prev_status == "Healthy":
                    start_time = time[i]
                    # Find falling edge: Faulty -> Healthy
                    for j in range(i + 1, len(health)):
                        next_status = health[j].get(component, "Healthy")
                        if next_status == "Healthy":
                            duration = time[j] - start_time
                            detected_faults.append((component, start_time, duration))
                            break
                    else:
                        # Fault persists to end of simulation
                        duration = time[-1] - start_time
                        detected_faults.append((component, start_time, duration))
        return detected_faults

    @property
    def i(self):
        return self._i

    def _plot_combined_timeline(self, health: list, mode: list, fault_history: dict, time: np.ndarray, out_dir: str, base_name: str):
        """Plot combined timeline showing autonomy modes, injected faults, and detected faults."""
        plt.switch_backend("Agg")
        sns.set_style("whitegrid")
        # Set larger font size for the entire figure
        plt.rcParams['font.size'] = 13
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 13
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        
        fig, (ax_mode, ax_injected, ax_detected) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        # fig.suptitle('Simulation Timeline', fontsize=18)
        
        # Plot 1: Autonomy Mode Timeline
        self._plot_mode_timeline_subplot(mode, time, ax_mode)
        
        # Plot 2: Injected Faults Timeline (now above detected faults)
        self._plot_injected_faults_subplot(fault_history, ax_injected)
        
        # Plot 3: Detected Faults Timeline  
        self._plot_detected_faults_subplot(health, time, ax_detected)
        
        # Set common x-axis label only on bottom plot
        ax_detected.set_xlabel('Time [s]')
        
        # Set x-axis limits for all subplots
        if len(time) > 0:
            for ax in [ax_mode, ax_injected, ax_detected]:
                ax.set_xlim(time[0], time[-1])
        
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{base_name}_timeline.png")
        try:
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Reporter.finalize: saved combined timeline PNG to {png_path}")
        except Exception as e:
            print(f"Reporter.finalize: failed to save combined timeline PNG: {e}")

    def _plot_mode_timeline_subplot(self, mode: list, time: np.ndarray, ax):
        """Plot autonomy mode timeline as subplot."""
        if not mode or all(m is None for m in mode):
            ax.text(0.5, 0.5, 'No Mode Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title('Autonomy Modes')
            return
            
        # Filter out None and 'unknown' values
        valid_indices = [i for i, m in enumerate(mode) if m is not None and m != "unknown"]
        if not valid_indices:
            ax.text(0.5, 0.5, 'No Valid Mode Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title('Autonomy Modes')
            return
            
        valid_modes = [mode[i] for i in valid_indices]
        valid_times = time[valid_indices]
        
        mode_colors = {'nominal': '#2E8B57', 'degraded': '#FF8C00', 'critical': '#DC143C'}
        mode_y_pos = {'nominal': 2, 'degraded': 1, 'critical': 0}
        
        # Create mode segments
        current_mode = valid_modes[0]
        current_start = valid_times[0]
        mode_segments = []
        
        for i in range(1, len(valid_modes)):
            if valid_modes[i] != current_mode:
                duration = valid_times[i] - current_start
                mode_segments.append((current_mode, current_start, duration))
                current_mode = valid_modes[i]
                current_start = valid_times[i]
        
        duration = valid_times[-1] - current_start
        mode_segments.append((current_mode, current_start, duration))
        
        # Plot mode segments
        plotted = set()
        for mode_name, start_time, duration in mode_segments:
            y_pos = mode_y_pos.get(mode_name, -1)
            color = mode_colors.get(mode_name, '#808080')
            label = mode_name if mode_name not in plotted else ""
            ax.barh(y_pos, duration, left=start_time, height=0.6, color=color, alpha=0.8, label=label)
            plotted.add(mode_name)
            # removed in-bar text labels to keep plot clean for publication
            # if duration > (valid_times[-1] - valid_times[0]) * 0.05:
            #     ax.text(start_time + duration/2, y_pos, mode_name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        unique_modes = [m for m in ['nominal','degraded','critical'] if m in plotted]
        y_ticks = [mode_y_pos[m] for m in unique_modes]
        # Capitalize mode names for y-axis labels
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([m.capitalize() for m in unique_modes])
        ax.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
        ax.set_title('Autonomy Modes')
        ax.grid(True, alpha=0.3)
        # remove legend from autonomy modes plot (leave only y-axis labels)
        # if len(plotted) > 1:
        #     ax.legend(loc='upper right')

    def _plot_detected_faults_subplot(self, health: list, time: np.ndarray, ax):
        """Plot detected faults timeline as subplot."""
        if not health or all(h is None for h in health):
            ax.text(0.5, 0.5, 'No Health Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title('Detected Faults')
            return
            
        valid_health = [h for h in health if h is not None]
        if not valid_health:
            ax.text(0.5, 0.5, 'No Valid Health Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title('Detected Faults')
            return
            
        detected_faults = self._process_fault_detections(valid_health, time[:len(valid_health)])
        
        if not detected_faults:
            ax.text(0.5, 0.5, 'No Faults Detected', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 0.5)
        else:
            y_positions = {}
            y_counter = 0
            colors = plt.cm.Dark2(np.linspace(0, 1, len(set(fault[0] for fault in detected_faults))))
            color_map = {}
            
            for i, (component, start_time, duration) in enumerate(detected_faults):
                if component not in y_positions:
                    y_positions[component] = y_counter
                    color_map[component] = colors[len(y_positions) - 1]
                    y_counter += 1
                
                y_pos = y_positions[component]
                ax.barh(y_pos, duration, left=start_time, height=0.6, 
                       color=color_map[component], alpha=0.7, 
                       label=component if component not in [f[0] for f in detected_faults[:i]] else "")
                # removed in-bar text labels (keep only y-axis labels)
                # ax.text(start_time + duration/2, y_pos, component, 
                #        ha='center', va='center', fontsize=8, fontweight='bold')
            
            ax.set_yticks(list(y_positions.values()))
            # Display labels: replace "IMU" with "Gyro" for clarity on plots,
            # but keep internal component names unchanged.
            raw_labels = list(y_positions.keys())
            display_labels = [lbl.replace("IMU", "Gyro") for lbl in raw_labels]
            # Wrap long labels with new line, only two words per line
            display_labels = ['\n'.join(lbl.split(' ')[:2]) if len(lbl.split(' ')) > 2 else lbl for lbl in display_labels]
            ax.set_yticklabels(display_labels)
            ax.set_ylim(-0.5, len(y_positions) - 0.5)
        
        ax.set_title('Detected Faults')
        ax.grid(True, alpha=0.3)

    def _plot_injected_faults_subplot(self, fault_history: dict, ax):
        """Plot injected faults timeline as subplot."""
        if not fault_history:
            ax.text(0.5, 0.5, 'No Fault History', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title('Injected Faults')
            return
        
        # Extract all fault names and times
        all_fault_names = set()
        times = sorted(fault_history.keys())
        for fault_status in fault_history.values():
            all_fault_names.update(fault_status.keys())
        
        if not all_fault_names:
            ax.text(0.5, 0.5, 'No Faults in History', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title('Injected Faults')
            return
        
        # Create y-position mapping for fault names, ordered by first injection time (last to first)
        fault_first_times = {}
        for time_key in times:
            for fault_name in fault_history[time_key]:
                if fault_history[time_key][fault_name] and fault_name not in fault_first_times:
                    fault_first_times[fault_name] = time_key
        
        # Sort by first injection time (latest first, so reverse=True)
        fault_names = sorted(all_fault_names, key=lambda name: fault_first_times.get(name, float('-inf')), reverse=False)
        y_positions = {name: i for i, name in enumerate(fault_names)}
        colors = plt.cm.tab10(np.linspace(0, 1, len(fault_names)))
        
        # Find fault activation segments
        for fault_name in fault_names:
            color = colors[y_positions[fault_name]]
            y_pos = y_positions[fault_name]
            active_start = None
            for i, time_key in enumerate(times):
                is_active = fault_history[time_key].get(fault_name, False)
                
                if is_active and active_start is None:
                    # Fault becomes active
                    active_start = time_key
                elif not is_active and active_start is not None:
                    # Fault becomes inactive
                    duration = time_key - active_start
                    # Wrap fault name with new line every two words for better display
                    ax.barh(y_pos, duration, left=active_start, height=0.6, 
                           color=color, alpha=0.7)
                    active_start = None
            
            # Handle fault active until end
            if active_start is not None and times:
                duration = times[-1] - active_start
                ax.barh(y_pos, duration, left=active_start, height=0.6, 
                       color=color, alpha=0.7)
        
        # f_name_wrap = '\n'.join(wrap(fault_name, 15))  # wrap fault name every 15 chars
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(['\n'.join(wrap(name, 15)) for name in y_positions.keys()])
        ax.set_ylim(-0.5, len(y_positions) - 0.5)
        ax.set_title('Injected Faults')
        ax.grid(True, alpha=0.3)

def load_and_regenerate_plots(data_file_path: str, output_dir: str = None):
    """Load saved plot data and regenerate all plots."""
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    with open(data_file_path, 'rb') as f:
        plot_data = pickle.load(f)
    
    # Use provided output directory or same directory as data file
    if output_dir is None:
        output_dir = os.path.dirname(data_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary reporter instance just for plot generation
    reporter = Reporter()
    time = plot_data['time']
    health = plot_data['health']
    fault_history = plot_data['fault_history']
    latency_info = reporter._calculate_fault_detection_latencies(health, fault_history, time)

    reporter._generate_all_plots(plot_data, output_dir, latency_info)
    
    print(f"Successfully regenerated plots in: {output_dir}")

def main():
    """Main function for standalone script usage."""
    parser = argparse.ArgumentParser(
        description="Regenerate simulation report plots from saved data",
        prog="report"
    )
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the saved plot data file (.pkl)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Output directory for plots (default: same as data file)",
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        load_and_regenerate_plots(args.data_file, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
