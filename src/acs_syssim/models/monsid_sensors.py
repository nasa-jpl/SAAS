import csv
from typing import NamedTuple
from syssim import Node, InputPort

class NodeCSVLoggerInputs(NamedTuple):
    cmd_torque: InputPort
    sens_rate: InputPort
    sens_imu_rate: InputPort
    sens_q_sc_to_eci: InputPort

class NodeMONSIDCSVLogger(Node):
    def __init__(self, **kwargs):
        self._i = NodeCSVLoggerInputs(
            InputPort("TauCmd", self),
            InputPort("TrueRate", self),
            InputPort("MeasRate", self),
            InputPort("MeasQuat", self),
        )

        self._header = ["time"]
        for ip in self._i:
            if ip.name == "MeasQuat":
                # For the quaternion, we will have 4 components
                self._header += [f"{ip.name}_{n+1}" for n in range(4)]
            else:
                self._header += [f"{ip.name}_{n+1}" for n in range(3)]

        super().__init__(self._i, None, **kwargs)

    def initialize(self):
        """Initialize the node. This is called before the simulation starts."""

        # Check if we have a name and a save dir for this simulation
        if self._system._sim_name is not None and self._system._save_dir is not None:
            self._csv_file_path = f"{self._system._save_dir}/{self._system._sim_name}_monsid_record.csv"
        else:
            # Use a tmporary file
            self._csv_file_path = "/tmp/monsid_record.csv"
        # Create the CSV file and write the header
        self._file_handle = open(self._csv_file_path, mode='w')
        self._writer = csv.writer(self._file_handle, lineterminator='\n')
        self._writer.writerow(self._header)

    def update(self, sim_time: float):
        """Update the node. This is called at each simulation step."""

        # Read the inputs
        row = [sim_time]
        for ip in self._i:
            if ip.name == "MeasQuat":
                # For the quaternion, we will have 4 components
                row += [f"{ip.read()[i]:.4f}" for i in range(4)]
            else:
                # For the other inputs, we will have 3 components
                row += [f"{ip.read()[i]:.4f}" for i in range(3)]

        # Write to the CSV file
        self._writer.writerow(row)

    def finalize(self):
        """Finalize the node. This is called after the simulation ends."""
        # Close the CSV file
        self._file_handle.close()

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return ()
