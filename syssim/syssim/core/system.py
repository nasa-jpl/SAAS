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


class NodeSystem:
    def __init__(self):
        """Represents a system that can be simulated. The system is made up of nodes and the connections between their ports, which in turn dictate how data flows thorugh the simulation. Think MATLAB Simulink."""
        self._ex_plan: List[Node] = None
        self._nodes: List[Node] = list()
        self._faults = dict()
        self._detected_faults = list()

    def __repr__(self):
        output = ""
        for n in self._nodes:
            output += f"{n.name} = {n.__class__.__name__}:\n\t"
            for p in n.ports().values():
                output += f"{p.name}\n\t"
            output += "\n"
        return output

    def add_node(self, n: Node):
        """Register a node into the system.

        Args:
            n (Node): node to register

        Raises:
            Exception: Raised if the node has already been added to the system
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

    def add_faults(self, toml_spec: Union[str, PathLike]):
        """Register a set of faults into the simulation. The nodes and ports to which the faults are assigned must already have been added to the system.

        Args:
            toml_spec (Union[str, PathLike]): path to the TOML file specifying the faults
        """
        spec = toml.load(toml_spec)

        for nn in spec:
            node = self.get_node(nn)
            for ports in spec[nn]:
                for fault_spec in spec[nn][ports]:
                    fault = FaultBasic(fault_spec, node, node[ports])
                    if node in self._faults:
                        self._faults[node].append(fault)
                    else:
                        self._faults[node] = [fault]

    def detect_fault(self, name: str, time: float):
        """Register a detection of a fault with the system. Used to allow plotting them later.

        Args:
            name (str): name of the fault that we think we detected
            time (float): time we think we detected it
        """
        self._detected_faults.append((name, time))

    def get_node(self, node_name: str) -> Node:
        """Get a node in the system by name

        Args:
            node_name (str): name of the node

        Returns:
            Node: requested node
        """
        for n in self._nodes:
            if n.name == node_name:
                return n

    def get_faults(self, node: Node) -> List:
        """Get all the fauts registered for a given node in the system

        Args:
            node (Node): the node

        Returns:
            List[Faults]: list of faults for this node
        """
        return self._faults[node]

    def get_fault_detections(self) -> List[Tuple[str, float]]:
        """Get all the faults we think we have detected

        Returns:
            List[Tuple[str, float]]: list of fault detections
        """
        return self._detected_faults

    def get_faults(self) -> List:
        """Get all the faults registered in this system

        Returns:
            List: list of faults
        """
        faults = []
        for f in self._faults.values():
            faults += f
        return faults

    def get_output_dir(self) -> str:
        """Get the directory in which we should store results for this simulation of this system

        Returns:
            str: directory to store results
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

    def simulate(
        self, t_f: float, dt=0.01, save_dir=None, sim_name=None, batches: int = 1
    ):
        """Simulate the system. All nodes are reinitialized between simulation batches and all realizations of fault statistics are rerandomized.

        Args:
            t_f (float): final time of the simulation
            dt (float, optional): The default node period to use. Any node that has not has its period of frequency explicitly set will use this value. Defaults to 0.01.
            save_dir (str, optional): directory in which to save results. Defaults to None.
            sim_name (str, optional): name to use for this simulation. Used to construct the save directory. Defaults to None.
            batches (int, optional): How many simulation batches to run. Defaults to 1.
        """
        if batches == 1:
            self._simulation_iterate(t_f, dt, save_dir, sim_name)
        else:
            for i in tqdm(range(batches), desc="Simulation Batches", position=0):
                self._simulation_iterate(t_f, dt, save_dir, sim_name)

    def compile(self) -> List[Node]:
        """Build an execution order for nodes in the system using topological sort.

        Raises:
            Exception: If the system contains a topological cycle (also called an "algebraic loop" and cannot be solved.

        Returns:
            List[Node]: A topological ordering of nodes
        """
        dg = self._build_dependency_graph()
        try:
            topo_i = rwx.topological_sort(dg)
        except rwx.DAGHasCycle:
            mpl_draw(dg, with_labels=True, labels=lambda n: n.name)
            raise Exception("Failed to compile graph. Topological cycle detected.")
        return [dg[i] for i in reversed(topo_i)]

    def _simulation_iterate(self, t_f: float, dt: float, save_dir: str, sim_name: str):
        """Run one simualtion of the system

        Args:
            t_f (float): final simulation time
            dt (float): default simulation step
            save_dir (str): directory to save outputs
            sim_name (str): simulation name
        """
        self._sim_start_time = datetime.now()
        self._save_dir = save_dir
        self._sim_name = sim_name

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
            if n in self._faults:
                for f in self._faults[n]:
                    f._gen_state()

        for st, ns in tqdm(
            self._schedule.items(), leave=False, position=1, desc=f"Sim: {sim_name}"
        ):
            # print(f"Processing time step {st}...")
            # Update each fault
            for n in self._faults:
                for f in self._faults[n]:
                    f.update(st)

            # Get all times and nodes to update at each time
            for n in self._ex_plan:
                if n in ns:
                    # If the node should be updated at this time, update it.
                    n.update(st)

        # Finalize all blocks
        for n in self._nodes:
            n.finalize()

        self._detected_faults.clear()

    def _build_dependency_graph(self) -> rwx.PyDAG:
        """Build a graph encoding the dependencies between nodes in this system.

        Returns:
            PyDAG: rustworkx directed acyclic graph
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
        """Build a dictionary that maps from simulation time to a list of nodes that must be updated at that time.

        Args:
            tf (float): final time of the simulation

        Returns:
            Dict[float, List[Node]]: the schedule for the simulation
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
