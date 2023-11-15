from typing import Dict, Union
import numpy as np
from syssim.core import Node, InputPort, OutputPort


class FaultBasic:
    def __init__(self, spec: Dict, node: Node, port: Union[InputPort, OutputPort]):
        """A basic fault that can be constructed from a definition given in a TOML file. Allows the fault to occur at a random time and for a random duration with a given probability. Can either hold, randomize, or disconnect port values.

        Args:
            spec (Dict): dictionary of key value pairs derived from a markup language like TOML which specifies the definition of this fault
            node (Node): the node the fault will occur on
            port (Union[InputPort, OutputPort]): the port the fault will occur on

        Basic faults are defined using the TOML specification laid out in the README.md file in the same directory as this file.
        """
        self._spec = spec
        self._node = node
        self._port = port
        self._setup_parameters()
        self._is_active = False
        port.add_fault(self)
        self._state = (None, None, None)
        self._name = self._spec["name"]

    def __repr__(self) -> str:
        if self._state[2] == True:
            return f"Basic Fault [{self._name}]:\n\tNode = {self._node.name}\n\tPort = {self._port.name}\n\tStart time = {self._state[0]}\n\tDuration = {self._state[1]}"
        else:
            return f"Basic Fault [{self._name}]:\n\tNode = {self._node.name}\n\tPort = {self._port.name}\n\tDoes not occur"

    def _setup_parameters(self):
        """Setup the parameters of the class given its spec"""
        if "t" in self._spec["start-time"].keys():
            self._t = lambda: self._spec["start-time"]["t"]
        elif "gaussian" in self._spec["start-time"].keys():
            self._t = lambda: np.random.normal(
                self._spec["start-time"]["gaussian"]["mean"],
                self._spec["start-time"]["gaussian"]["dev"],
            )
        else:
            raise Exception()

        if "dt" in self._spec["duration"].keys():
            self._dt = lambda: self._spec["duration"]["dt"]
        elif "gaussian" in self._spec["duration"].keys():
            self._dt = lambda: np.random.normal(
                self._spec["duration"]["gaussian"]["mean"],
                self._spec["duration"]["gaussian"]["dev"],
            )
        else:
            raise Exception()

        try:
            self._occurance = self._spec["occurance"]["p"]
        except KeyError:
            raise Exception(
                "Must provide occurance probability for fault between 0.0 and 1.0."
            )
        self._p = lambda: np.random.choice(
            [True, False], p=[self._occurance, 1 - self._occurance]
        )

    def _gen_state(self):
        """Generate the state of fault. Includes randomizing all random values."""
        self._state = (self._t(), self._dt(), self._p())

    def start_time(self) -> float:
        """Get the start tie of the fault

        Returns:
            float: start time
        """
        return self._state[0]

    def duration(self) -> float:
        """Get the duration of the fault

        Returns:
            float: the duration
        """
        return self._state[1]

    def is_occuring(self) -> bool:
        """Get if the fault will occur in the current realization

        Returns:
            bool: is the fault occuring
        """
        return self._state[2]

    def is_active(self) -> bool:
        """Get if the fault is occuring now

        Returns:
            bool: is the fault active now
        """
        return self._is_active

    def get_name(self) -> str:
        """Get the name of the fault

        Returns:
            str: name
        """
        return self._name

    def update(self, sim_time: float):
        """Update the fault to see if it should be activated

        Args:
            sim_time (float): current simulation time
        """
        if sim_time >= self.start_time() and sim_time < (
            self.start_time() + self.duration()
        ):
            if self.is_occuring():
                self._is_active = True
        else:
            self._is_active = False

    def action(self, v: np.ndarray):
        """Apply the fault action to the port it affects

        Args:
            v (np.ndarray): the value from the port to mutate
        """
        action_type = self._spec["action"]["type"]

        if (
            isinstance(self._spec["action"]["index"], dict)
            and "start" in self._spec["action"]["index"]
            and "end" in self._spec["action"]["index"]
        ):
            index = np.arange(
                self._spec["action"]["index"]["start"],
                self._spec["action"]["index"]["stop"],
            )
        elif isinstance(self._spec["action"]["index"], list):
            index = self._spec["action"]["index"]
        else:
            index = [self._spec["action"]["index"]]
        if action_type == "random":
            if "gaussian" in self._spec["action"]["value"].keys():
                mu = self._spec["action"]["value"]["gaussian"]["mean"]
                sig = self._spec["action"]["value"]["gaussian"]["dev"]

                dist = lambda: np.random.normal(mu, sig)

            elif "uniform" in self._spec["action"]["value"].keys():
                low = self._spec["action"]["value"]["gaussian"]["low"]
                high = self._spec["action"]["value"]["gaussian"]["high"]

                dist = lambda: np.random.uniform(low, high)

            else:
                raise Exception(
                    "action.value.gaussian or action.value.uniform must exist."
                )
            for i in index:
                v[i] = dist()

        elif action_type == "hold":
            if isinstance(self._spec["action"]["value"], list):
                vals = self._spec["action"]["value"]
                if len(vals) != len(index):
                    raise Exception(
                        "For hold, must provide only one value or list of values the same length as index."
                    )
                for i, val in zip(index, vals):
                    v[i] = val
            else:
                for i in index:
                    v[i] = self._spec["action"]["value"]
        elif action_type == "disconnect":
            for i in index:
                v[i] = np.nan
        else:
            raise Exception(f"Action type of {action_type} not known.")
