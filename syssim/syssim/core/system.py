from typing import List, Union, Tuple, Dict
from os import PathLike, path, makedirs
from datetime import datetime

import rustworkx as rwx
from rustworkx.visualization import mpl_draw
import numpy as np
from tqdm import tqdm
import toml

from syssim.core.node import Node
from syssim.fault import FaultBasic
from syssim.core.fault import Fault


class NodeSystem:
    def __init__(self):
        """Simulatable collection of nodes connected by ports.

        The system owns all nodes, computes an execution order by topological
        sort, evaluates faults, and iterates the simulation timeline.

        Notes
        -----
        This class is analogous to a Simulink diagram: add nodes, connect
        their ports, register faults, then call :meth:`simulate`.
        """
        self._ex_plan: List[Node] = None
        self._nodes: List[Node] = list()
        self._faults : List[Fault] = list()
        self._detected_faults = list()
        self._fault_history: Dict[float, Dict[str, bool]] = {}  # Track fault status over time

    def __repr__(self):
        output = ""
        for n in self._nodes:
            output += f"{n.name} = {n.__class__.__name__}:\n\t"
            for p in n.i:
                output += f"{p.name}\n\t"
            for p in n.o:
                output += f"{p.name}\n\t"
            output += "\n"
        return output

    def add_node(self, n: Node):
        """Register a node with the system.

        Parameters
        ----------
        n : Node
            Node instance to register.

        Raises
        ------
        Exception
            If a node with the same name already exists in the system.
        """
        names = [n.name for n in self._nodes]
        if n.name == None:
            node_type = n.__class__.__name__
            i = 1
            default_name = f"{node_type}-{i}"
            while default_name in names:
                i += 1
                default_name = f"{node_type}-{i}"
            n.name = default_name
        elif n.name in names:
            raise Exception(
                f"Node with name {n.name} already in diagram. Cannot add node."
            )

        self._nodes.append(n)
        n._system = self

    def add_faults(self, fault : Union[Fault, List[Fault]]):
        """Register one or more faults.

        Parameters
        ----------
        fault : Fault or list of Fault
            Faults that have already been attached to a port or parameter.

        Raises
        ------
        Exception
            If a provided object is not an instance of :class:`Fault`.
        """
        if isinstance(fault, Fault):
            self._faults.append(fault)
        else:
            for f in fault:
                if isinstance(f, Fault):
                    self._faults.append(f)
                else:
                    raise Exception(
                        "Faults must be of type Fault or a subclass of Fault."
                    )

    def detect_fault(self, name: str, time: float):
        """Record a detected fault event.

        Parameters
        ----------
        name : str
            Name of the detected fault.
        time : float
            Simulation time of the detection.
        """
        self._detected_faults.append((name, time))

    def get_node(self, node_name: str) -> Node:
        """Return a node by name.

        Parameters
        ----------
        node_name : str
            Name of the node.

        Returns
        -------
        Node
            Matching node or ``None`` if not present.
        """
        for n in self._nodes:
            if n.name == node_name:
                return n
    # NOTE Deprecated for now
    # def get_faults(self, node: Node) -> List:
    #     """Get all the fauts registered for a given node in the system

    #     Args:
    #         node (Node): the node

    #     Returns:
    #         List[Faults]: list of faults for this node
    #     """
    #     return self._faults[node]

    def get_fault_detections(self) -> List[Tuple[str, float]]:
        """Return all fault detections.

        Returns
        -------
        list of tuple
            Pairs of fault name and detection time.
        """
        return self._detected_faults

    def get_faults(self) -> List:
        """Return registered faults.

        Returns
        -------
        list of Fault
            Faults attached to the system.
        """
        return self._faults

    def get_output_dir(self) -> str:
        """Directory path for simulation outputs.

        Returns
        -------
        str or None
            Full path to the run-specific output directory, or ``None`` if not
            set.
        """
        if (
            self._sim_name != None
            and self._sim_start_time != None
            and self._save_dir != None
        ):
            return f"{self._save_dir}/{self._sim_name}-{self._sim_start_time}/".replace(
                " ", "-"
            )
        else:
            return None

    def get_fault_history(self) -> Dict[float, Dict[str, bool]]:
        """Return the recorded fault activation timeline.

        Returns
        -------
        dict
            Mapping from simulation time to ``{fault_name: active}``.
        """
        return self._fault_history.copy()

    def simulate(
        self, t_f: float, dt=0.01, save_dir=None, sim_name=None, batches: int = 1
    ):
        """Run one or more simulation batches.

        Parameters
        ----------
        t_f : float
            Final simulation time (exclusive).
        dt : float, optional
            Default node period when not explicitly set on a node, by default
            0.01.
        save_dir : str, optional
            Directory where results (plots, artifacts) should be saved.
        sim_name : str, optional
            Name of the simulation run, used to build the output directory.
        batches : int, optional
            Number of realizations to run. Each batch reinitializes nodes and
            re-randomizes fault statistics.
        """
        if batches == 1:
            self._simulation_iterate(t_f, dt, save_dir, sim_name)
        else:
            for i in tqdm(range(batches), desc="Simulation Batches", position=0):
                self._simulation_iterate(t_f, dt, save_dir, sim_name)

    def compile(self) -> List[Node]:
        """Topologically sort nodes to obtain an execution order.

        Returns
        -------
        list of Node
            Nodes in the order they should be executed.

        Raises
        ------
        Exception
            If the graph contains a cycle (algebraic loop).
        """
        dg = self._build_dependency_graph()
        try:
            topo_i = rwx.topological_sort(dg)
        except rwx.DAGHasCycle:
            mpl_draw(dg, with_labels=True, labels=lambda n: n.name)
            raise Exception("Failed to compile graph. Topological cycle detected.")
        return [dg[i] for i in reversed(topo_i)]

    def _simulation_iterate(self, t_f: float, dt: float, save_dir: str, sim_name: str):
        """Run a single realization of the system.

        Parameters
        ----------
        t_f : float
            Final simulation time.
        dt : float
            Default simulation step.
        save_dir : str
            Output directory for artifacts.
        sim_name : str
            Simulation name for directory construction.
        """
        self._sim_start_time = datetime.now()
        self._save_dir = save_dir
        self._sim_name = sim_name
        self._fault_history.clear()  # Reset fault history for new simulation

        if self.get_output_dir() != None:
            if not path.exists(self.get_output_dir()):
                makedirs(self.get_output_dir())

        for n in self._nodes:
            # If the nodes execution period is not set, then set it from the the default given in the call.
            if n.period == None:
                n.period = dt

        # Compile to get execution plan
        self._ex_plan = self.compile()
        self._schedule = self._build_schedule(t_f)

        # Initialize all blocks
        for n in self._nodes:
            n.initialize()
        for f in self._faults:
            f.initialize()

        for st, ns in tqdm(
            self._schedule.items(), leave=False, position=1, desc=f"Sim --- {sim_name}"
        ):
            # print(f"Processing time step {st}...")
            # Update each fault
            for f in self._faults:
                f.update(st)

            # Record fault status at this timestep
            fault_status = {}
            for f in self._faults:
                fault_status[f.name] = f.triggered and f.active
            self._fault_history[st] = fault_status

            # Get all times and nodes to update at each time
            for n in self._ex_plan:
                if n in ns:
                    # If the node should be updated at this time, update it.
                    n.update(st)

        # Finalize all blocks with fault history
        for n in self._nodes:
            n.finalize(fault_history=self._fault_history)

        self._detected_faults.clear()

    def _build_dependency_graph(self) -> rwx.PyDAG:
        """Build a dependency graph for the current nodes.

        Returns
        -------
        rustworkx.PyDAG
            DAG with nodes corresponding to simulation nodes and edges pointing
            from dependents to dependencies.
        """
        dg = rwx.PyDAG()
        ndinx = dg.add_nodes_from(self._nodes)
        for ni in ndinx:
            n = self._nodes[ni]
            for p in n.depends():
                pi = self._nodes.index(p)
                dg.add_edge(ni, pi, None)
        return dg

    def _build_schedule(self, tf: float) -> Dict[float, List[Node]]:
        """Construct the simulation schedule.

        Parameters
        ----------
        tf : float
            Final time (exclusive).

        Returns
        -------
        dict
            Mapping time step to list of nodes scheduled for update.
        """
        sched = {}
        for n in self._nodes:
            for t in np.arange(0, tf, n.period):
                if t in sched.keys():
                    sched[t] += [n]
                else:
                    sched[t] = [n]
        return dict(sorted(sched.items()))

    def _initialize_faults(self):
        """Generate initial realizations of faults."""
        for fault_list in self._faults.values():
            for f in fault_list:
                f._gen_state()
