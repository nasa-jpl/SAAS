from collections import defaultdict, deque
import csv
import re
from typing import NamedTuple
from syssim import Node, InputPort, OutputPort
import io
import tempfile
import os
import threading
import subprocess
import numpy as np
from tqdm import tqdm

class NodeMONSIDDiagnoserInputs(NamedTuple):
    dynamics_rate: InputPort
    dynamics_orientation: InputPort
    rw1_cmd: InputPort
    rw2_cmd: InputPort
    rw3_cmd: InputPort
    rw4_cmd: InputPort
    rw1_momentum: InputPort
    rw2_momentum: InputPort
    rw3_momentum: InputPort
    rw4_momentum: InputPort
    enc1: InputPort
    enc2: InputPort
    enc3: InputPort
    enc4: InputPort
    imu1: InputPort
    imu2: InputPort
    sru1: InputPort
    sru2: InputPort

class NodeMONSIDDiagnoserOutputs(NamedTuple):
    rw1_health: OutputPort
    rw2_health: OutputPort
    rw3_health: OutputPort
    rw4_health: OutputPort
    fault_detected: OutputPort

class NodeMONSIDDiagnoser(Node):
    def __init__(self, buffer: int = 2, **kwargs):
        self._n_buf = buffer # How many inputs to buffer for input to MONSID model
        self._i = NodeMONSIDDiagnoserInputs(
            InputPort("DynamicsRate", self),
            InputPort("DynamicsOrientation", self),
            InputPort("RW1Cmd", self),
            InputPort("RW2Cmd", self),
            InputPort("RW3Cmd", self),
            InputPort("RW4Cmd", self),
            InputPort("RW1Momentum", self),
            InputPort("RW2Momentum", self),
            InputPort("RW3Momentum", self),
            InputPort("RW4Momentum", self),
            InputPort("Enc1", self),
            InputPort("Enc2", self),
            InputPort("Enc3", self),
            InputPort("Enc4", self),
            InputPort("IMU1", self),
            InputPort("IMU2", self),
            InputPort("SRU1", self),
            InputPort("SRU2", self),
        )
        self._o = NodeMONSIDDiagnoserOutputs(
            OutputPort("RW1Health", self),
            OutputPort("RW2Health", self),
            OutputPort("RW3Health", self),
            OutputPort("RW4Health", self),
            OutputPort("FaultDetected", self),
        )
        self._buffer = deque(maxlen=self._n_buf)  # Ring buffer with automatic size management
        self._previous_diagnosis = {}  # Track previous component health status
        # Buffer for RW commands to delay by one step
        self._rw_cmd_buffer = {
            'rw1_cmd': np.array([0.0, 0.0, 0.0]),
            'rw2_cmd': np.array([0.0, 0.0, 0.0]),
            'rw3_cmd': np.array([0.0, 0.0, 0.0]),
            'rw4_cmd': np.array([0.0, 0.0, 0.0])
        }
        self._header = ["time"]
        for ip in self._i:
            if "SRU" in ip.name or "DynamicsOrientation" in ip.name:
                # For the quaternion, we will have 4 components
                self._header += [f"{ip.name}_{n+1}" for n in range(4)]
            elif "Enc" in ip.name:
                self._header += [f"{ip.name}"]
            else:
                self._header += [f"{ip.name}_{n+1}" for n in range(3)]


        super().__init__(self._i, self._o, **kwargs)

    def initialize(self):
        """Initialize the node. This is called before the simulation starts."""
        # Make a temporary file path for the csv pipe that will be used for communication with MONSID
        self._tmp_path = f'{tempfile.gettempdir()}/monsid_pipe_{os.getpid()}.csv'
        # os.mkfifo(self._tmp_path)
        # fd = os.open(self._tmp_path, os.O_WRONLY | os.O_NONBLOCK)  # Open the FIFO for reading and writing
        # self._fifo_pipe_file = os.fdopen(fd, 'w')  
        self._tmp_csv = open(self._tmp_path, 'w', newline='')  # Open the temporary file for writing
        self._writer = csv.writer(self._tmp_csv, lineterminator='\n')

        # Check if we have a name and a save dir for this simulation
        # if self._system._sim_name is not None and self._system._save_dir is not None:
        #     self._csv_file_path = f"{self._system._save_dir}/{self._system._sim_name}_monsid_record.csv"
        # else:
        #     # Create a temporary file if no name or save dir is provided
        #     self._csv_file_path = tempfile.NamedTemporaryFile(delete=False, suffix="_monsid_record.csv").name
        # # Create the CSV file and write the header
        # self._file_handle = open(self._csv_file_path, mode='w')
        # self._writer = csv.writer(self._file_handle, lineterminator='\n')
        # self._writer.writerow(self._header)

    def update(self, sim_time: float):
        """Update the node. This is called at each simulation step."""
        pass
        # Read the inputs and store current RW commands for next step
        current_rw_cmds = {}
        row = [f"{sim_time:.10f}"]  # Start with the simulation time
        for ip in self._i:
            val = ip.read()
            
            # Handle RW commands with buffering
            if "Cmd" in ip.name:
                rw_key = ip.name.lower().replace("cmd", "_cmd")  # Convert "RW1Cmd" to "rw1_cmd"
                current_rw_cmds[rw_key] = val.copy()  # Store current value for next step
                # Use buffered value (from previous step) for CSV
                buffered_val = self._rw_cmd_buffer[rw_key]
                row += [f"{buffered_val[i]:.10f}" for i in range(3)]
            elif "SRU" in ip.name or "DynamicsOrientation" in ip.name:
                # Convert from real part first [w, x, y, z] to real part last [x, y, z, w]
                # Assume val is [w, x, y, z]
                row += [f"{val[i]:.10f}" for i in range(1, 4)] + [f"{val[0]:.10f}"]
            elif "Enc" in ip.name:
                row += [f"{val:.10f}"]
            else:
                row += [f"{val[i]:.10f}" for i in range(3)]
        
        # Update RW command buffer for next step
        for key, val in current_rw_cmds.items():
            self._rw_cmd_buffer[key] = val

        self._buffer.append(row)
        
        # Only process if we have enough data to establish state
        if len(self._buffer) < self._n_buf:
            self.o.rw1_health.shift_out(np.array(True))
            self.o.rw2_health.shift_out(np.array(True))
            self.o.rw3_health.shift_out(np.array(True))
            self.o.rw4_health.shift_out(np.array(True))
            self.o.fault_detected.shift_out(np.array([]))  # No faults detected yet
            return
        
        # Clear the file before writing (truncate to zero length)
        self._tmp_csv.seek(0)
        self._tmp_csv.truncate(0)
        self._writer.writerow(self._header)
        for row in self._buffer:
            self._writer.writerow(row)
        self._tmp_csv.flush()
        self._tmp_csv.seek(0)
            
        cmd = [
            "/home/j/Code/sync/saas/monsid_sdk/x64/Linux/debug/monsid-exec",
            "-l",
            "/home/j/Code/sync/saas/acs-monsid/build/debug/bin/acs-monsid.so",
            "-m",
            "acs_monsid_Model",
            "-i",
            self._tmp_path,
            "-fsr",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout

            # Step 2: Parse the output
            health_data = defaultdict(dict)
            lines = output.splitlines()
            i = 0
            current_timeslice = None

            # Find "Fault Identification count: %d" and extract the number of faults
            fault_count = 0
            for line in lines:
                match = re.search(r"Fault Identification count: (\d+)", line)
                if match:
                    fault_count = int(match.group(1))
                    break
            if fault_count == 0:
                # No faults detected, all components healthy
                self.o.rw1_health.shift_out(np.array(True))
                self.o.rw2_health.shift_out(np.array(True))
                self.o.rw3_health.shift_out(np.array(True))
                self.o.rw4_health.shift_out(np.array(True))
                self.o.fault_detected.shift_out(np.array([]))  # No faults detected
                return
            
            while i < len(lines):
                line = lines[i].strip()

                # Look for timeslice block start
                match = re.match(r"\* Fault threshold met @ timestamp: [\d.]+, timeslice: (\d+)", line)
                if match:
                    current_timeslice = int(match.group(1))
                    # Skip ahead to table
                    while i < len(lines) and not lines[i].strip().startswith("Health"):
                        i += 1
                    # Skip header lines
                    i += 3
                    # Read table rows
                    while i < len(lines):
                        row = lines[i].strip()
                        if not row or row.startswith("*"):
                            break
                        # Remove ANSI escape sequences
                        row_clean = re.sub(r'\x1b\[[0-9;]*m', '', row)
                        # Parse the component line
                        comp_match = re.match(r"(\w+) ?: ([\w ]+)\s+\|\s+([\w ]+)\s+\|\s+([\d.]+)", row_clean)
                        if comp_match:
                            component = comp_match.group(1)
                            status = comp_match.group(2).strip()
                            suspension_state = comp_match.group(3).strip()
                            rank = float(comp_match.group(4))
                            health_data[current_timeslice][component] = {
                                "status": status,
                                "suspension_state": suspension_state,
                                "rank": rank
                            }
                        i += 1
                else:
                    i += 1

            diagnosis_result = dict(health_data)
            components = list(diagnosis_result.values())

            component_final = components[-1]

            # Track newly detected faults (rising edge detection)
            newly_faulty_components = []
            
            # Check each component for new faults
            for comp_name, comp_data in component_final.items():
                is_faulty = comp_data["status"] == "Faulty"
                was_faulty = self._previous_diagnosis.get(comp_name, False)
                
                # Rising edge detection: was healthy, now faulty
                if is_faulty and not was_faulty:
                    newly_faulty_components.append(comp_name)
                
                # Update previous state
                self._previous_diagnosis[comp_name] = is_faulty

            self.o.rw1_health.shift_out(np.array(True) if component_final["RWA_1"]["status"] != "Faulty" else np.array(False))
            self.o.rw2_health.shift_out(np.array(True) if component_final["RWA_2"]["status"] != "Faulty" else np.array(False))
            self.o.rw3_health.shift_out(np.array(True) if component_final["RWA_3"]["status"] != "Faulty" else np.array(False))
            self.o.rw4_health.shift_out(np.array(True) if component_final["RWA_4"]["status"] != "Faulty" else np.array(False))
            self.o.fault_detected.shift_out(np.array(newly_faulty_components))

        except subprocess.CalledProcessError as e:
            self._monsid_result = {
                "returncode": e.returncode,
                "stdout": e.stdout,
                "stderr": e.stderr,
            }
            print(f"Error {e.returncode} running MONSID\n: {e.stdout}")
        except Exception as e:
            self._monsid_result = {
                "error": str(e),
            }
            print(f"Error running MONSID: {e}")

    def finalize(self):
        """Finalize the node. This is called after the simulation ends."""
        # Close the CSV file
        # self._file_handle.close()
        self._tmp_csv.close()
        # Remove the temporary FIFO file
        if os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return self._o

class NodeFaultPrinterInputs(NamedTuple):
    fault_detected: InputPort

class NodeFaultPrinter(Node):
    def __init__(self, **kwargs):
        self._i = NodeFaultPrinterInputs(
            InputPort("FaultDetected", self)
        )
        super().__init__(self._i, (), **kwargs)

    def initialize(self):
        """Initialize the node. This is called before the simulation starts."""
        pass

    def update(self, sim_time: float):
        """Update the node. This is called at each simulation step."""
        fault_list = self._i.fault_detected.read()
        
        # Check if any faults were detected
        if len(fault_list) > 0:
            for component_name in fault_list:
                # Use tqdm.write to print without interfering with progress bars
                tqdm.write(f"FAULT DETECTED: {component_name} at time {sim_time:.4f}s")

    def finalize(self):
        """Finalize the node. This is called after the simulation ends."""
        pass

    @property
    def i(self):
        return self._i
