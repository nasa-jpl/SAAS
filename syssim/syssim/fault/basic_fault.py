from typing import Dict, Union
import numpy as np
from syssim.core import Node, InputPort, OutputPort


class FaultBasic:
    def __init__(self, spec: Dict, node: Node, port: Union[InputPort, OutputPort]):
        """Fault defined from TOML-style specification.

        Parameters
        ----------
        spec : dict
            Parsed fault specification. See README for field schema.
        node : Node
            Node owning the port to fault.
        port : InputPort or OutputPort
            Port that will be mutated while the fault is active.

        Notes
        -----
        Fault parameters (start time, duration, occurrence, action type, and
        indices) support both fixed values and random draws per simulation
        batch.
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
        """Setup the parameters of the class given its spec."""
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
        """Generate fault realization for the current batch."""
        self._state = (self._t(), self._dt(), self._p())

    def start_time(self) -> float:
        """Start time for this realization."""
        return self._state[0]

    def duration(self) -> float:
        """Duration for this realization."""
        return self._state[1]

    def is_occuring(self) -> bool:
        """Whether the fault occurs in the current realization."""
        return self._state[2]

    def is_active(self) -> bool:
        """Whether the fault is active at the current simulation time."""
        return self._is_active

    def get_name(self) -> str:
        """Return the fault name."""
        return self._name

    def update(self, sim_time: float):
        """Update activation state based on simulation time."""
        if sim_time >= self.start_time() and sim_time < (
            self.start_time() + self.duration()
        ):
            if self.is_occuring():
                self._is_active = True
        else:
            self._is_active = False

    def action(self, v: np.ndarray):
        """Apply the configured mutation to the provided value."""
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
