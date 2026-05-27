from __future__ import annotations

from datetime import datetime
from fractions import Fraction
from math import gcd, lcm
from pathlib import Path
from typing import Iterable

import rustworkx as rwx
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from syssim.core.fault import Fault, FaultContext
from syssim.core.logging import CsvSimulationLogger
from syssim.core.node import Node
from syssim.core.port import InputPort, OutputPort, PortSample


class NodeSystem:
    """Collection of connected nodes, faults, schedules, and logs.

    Parameters
    ----------
    strict_types : bool, optional
        Enforce declared port and parameter type contracts at runtime.
    enable_faults : bool, optional
        Evaluate and apply registered faults during simulation.
    """

    def __init__(self, *, strict_types: bool = False, enable_faults: bool = True):
        self.strict_types = bool(strict_types)
        self.enable_faults = bool(enable_faults)
        self._nodes: list[Node] = []
        self._faults: list[Fault] = []
        self._detected_faults: list[tuple[str, float]] = []
        self._fault_history: dict[float, dict[str, bool]] = {}
        self._ex_plan: list[Node] = []
        self._period_steps: dict[Node, int] = {}
        self._base_dt = 0.01
        self._base_dt_fraction = Fraction(1, 100)
        self._t = 0.0
        self._step_index = 0
        self._initialized = False
        self._output_dir: Path | None = None
        self._logger: CsvSimulationLogger | None = None

    def __repr__(self) -> str:
        """Return a compact text summary of registered nodes and ports.

        Returns
        -------
        str
            One line per node in the system.
        """
        lines = []
        for node in self._nodes:
            ports = [port.attr_name for port in node.iter_ports()]
            lines.append(f"{node.name} = {node.__class__.__name__}: {', '.join(ports)}")
        return "\n".join(lines)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Registered nodes.

        Returns
        -------
        tuple of Node
            Nodes in insertion order.
        """
        return tuple(self._nodes)

    @property
    def faults(self) -> tuple[Fault, ...]:
        """Registered faults.

        Returns
        -------
        tuple of Fault
            Faults in insertion order.
        """
        return tuple(self._faults)

    def add_node(self, node: Node) -> Node:
        """Register a node with the system.

        Parameters
        ----------
        node : Node
            Node instance to add.

        Returns
        -------
        Node
            The same node after name assignment and strictness configuration.

        Raises
        ------
        ValueError
            If another registered node already has the requested name.
        """
        names = {item.name for item in self._nodes}
        if node.name is None:
            node.name = self._default_name(node.__class__.__name__, names)
        elif node.name in names:
            raise ValueError(f"Node with name {node.name!r} is already registered")
        node._system = self
        node.set_strict_types(self.strict_types)
        self._nodes.append(node)
        return node

    def add_faults(self, faults: Fault | Iterable[Fault]) -> None:
        """Register one or more faults with the system.

        Parameters
        ----------
        faults : Fault or iterable of Fault
            Fault instance or collection of instances to add.

        Raises
        ------
        TypeError
            If any object is not a ``Fault``.
        ValueError
            If another registered fault already has the requested name.
        """
        if isinstance(faults, Fault):
            faults = (faults,)
        names = {fault.name for fault in self._faults if fault.name is not None}
        for fault in faults:
            if not isinstance(fault, Fault):
                raise TypeError("add_faults expects Fault instances")
            if fault.name is None:
                fault.name = self._default_name(fault.__class__.__name__, names)
            elif fault.name in names:
                raise ValueError(f"Fault with name {fault.name!r} is already registered")
            names.add(fault.name)
            self._faults.append(fault)

    def detect_fault(self, name: str, time: float) -> None:
        """Record an externally detected fault event.

        Parameters
        ----------
        name : str
            Fault name or detection label.
        time : float
            Simulation time of the detection.
        """
        self._detected_faults.append((name, float(time)))

    def get_node(self, node_name: str) -> Node | None:
        """Return a registered node by name.

        Parameters
        ----------
        node_name : str
            Name assigned to the node.

        Returns
        -------
        Node or None
            Matching node, or ``None`` if absent.
        """
        return next((node for node in self._nodes if node.name == node_name), None)

    def get_faults(self) -> list[Fault]:
        """Return registered faults as a mutable copy.

        Returns
        -------
        list of Fault
            Registered faults in insertion order.
        """
        return list(self._faults)

    def get_fault_detections(self) -> list[tuple[str, float]]:
        """Return recorded fault detections.

        Returns
        -------
        list of tuple
            Pairs ``(name, time)`` recorded by ``detect_fault``.
        """
        return list(self._detected_faults)

    def get_fault_history(self) -> dict[float, dict[str, bool]]:
        """Return fault active-state history.

        Returns
        -------
        dict
            Mapping from simulation time to ``{fault_name: active}``.
        """
        return dict(self._fault_history)

    def get_output_dir(self) -> str | None:
        """Return the active simulation output directory.

        Returns
        -------
        str or None
            Output directory path, or ``None`` when no directory is active.
        """
        return None if self._output_dir is None else str(self._output_dir)

    def initialize(self, dt: float | None = None) -> None:
        """Prepare nodes, faults, timing, and execution order for a run.

        Parameters
        ----------
        dt : float, optional
            Default sample period for nodes that do not define one.
        """
        self._prepare_timing(dt)
        self._ex_plan = self.compile()
        self._t = 0.0
        self._step_index = 0
        self._fault_history.clear()
        self._detected_faults.clear()
        for node in self._nodes:
            for parameter in node.iter_parameters():
                parameter.reset()
            node.initialize()
        for fault in self._faults:
            fault.initialize()
        self._initialized = True

    def step(self, dt: float | None = None) -> dict[str, PortSample]:
        """Advance the system by one manual step.

        Parameters
        ----------
        dt : float, optional
            Step duration. Defaults to the prepared base timestep.

        Returns
        -------
        dict
            Port samples keyed by fully qualified port name.
        """
        if not self._initialized:
            self.initialize(dt)
        step_dt = self._base_dt if dt is None else float(dt)
        self._t = round(self._t + step_dt, 12)
        self._step_index += max(1, round(step_dt / self._base_dt))
        self._run_time(self._t)
        return self.samples()

    def finalize(self) -> None:
        """Finalize nodes, close log files, and clear initialized state."""
        for node in self._nodes:
            node.finalize(fault_history=self._fault_history)
        if self._logger is not None:
            self._logger.close()
            self._logger = None
        self._initialized = False

    def simulate(
        self,
        t_f: float,
        dt: float | None = None,
        *,
        log_dir: str | Path | None = None,
        save_dir: str | Path | None = None,
        sim_name: str | None = None,
        log_values: bool = True,
        log_faults: bool = True,
        enable_faults: bool | None = None,
        batches: int = 1,
        show_progress: bool = True,
    ) -> None:
        """Simulate the system until a final time.

        Parameters
        ----------
        t_f : float
            Final simulation time in seconds.
        dt : float, optional
            Default sample period for nodes without an explicit period.
        log_dir : str or pathlib.Path, optional
            Directory for CSV logs. Logging is disabled when omitted.
        save_dir : str or pathlib.Path, optional
            Output directory for nodes that save artifacts without CSV logs.
        sim_name : str, optional
            Run name used to create timestamped output subdirectories.
        log_values : bool, optional
            Write port and parameter values to ``values.csv``.
        log_faults : bool, optional
            Write fault states to ``faults.csv``.
        enable_faults : bool, optional
            Temporary override for fault evaluation during this simulation.
        batches : int, optional
            Number of independent simulation batches to run.
        show_progress : bool, optional
            Display a Rich progress bar.
        """
        previous_enable_faults = self.enable_faults
        if enable_faults is not None:
            self.enable_faults = bool(enable_faults)
        try:
            task_id = None
            with self._progress_bar(show_progress) as progress:
                for batch in range(batches):
                    self._output_dir = self._resolve_output_dir(log_dir, save_dir, sim_name, batch, batches)
                    self._logger = (
                        CsvSimulationLogger(self._output_dir, values=log_values, faults=log_faults)
                        if log_dir is not None
                        else None
                    )
                    self.initialize(dt)
                    if task_id is None:
                        task_id = progress.add_task("Simulating", total=self._simulation_steps(t_f) * batches)
                    if batches > 1:
                        progress.update(task_id, description=f"Simulating batch {batch + 1}/{batches}")
                    while self._t < t_f - 1e-12:
                        self._run_time(self._t)
                        progress.advance(task_id)
                        self._step_index += 1
                        self._t = round(float(self._step_index * self._base_dt_fraction), 12)
                    self.finalize()
        finally:
            self.enable_faults = previous_enable_faults

    def compile(self) -> list[Node]:
        """Topologically sort registered nodes by port dependencies.

        Returns
        -------
        list of Node
            Execution plan for one simulation timestep.

        Raises
        ------
        ValueError
            If a dependency node is not registered with the system.
        RuntimeError
            If the node dependency graph contains an algebraic cycle.
        """
        dependencies = {node: tuple(node.depends()) for node in self._nodes}
        unknown = {dep for deps in dependencies.values() for dep in deps if dep not in self._nodes}
        if unknown:
            names = ", ".join(dep.name or dep.__class__.__name__ for dep in unknown)
            raise ValueError(f"Node dependencies are not registered in the system: {names}")

        graph = rwx.PyDAG()
        node_indices = graph.add_nodes_from(self._nodes)
        node_to_index = dict(zip(self._nodes, node_indices))
        for node, deps in dependencies.items():
            for dependency in deps:
                graph.add_edge(node_to_index[dependency], node_to_index[node], None)

        try:
            return [graph[index] for index in rwx.topological_sort(graph)]
        except rwx.DAGHasCycle as exc:
            raise RuntimeError("Failed to compile node graph; topological cycle detected") from exc

    def samples(self) -> dict[str, PortSample]:
        """Return all current port samples.

        Returns
        -------
        dict
            Port samples keyed by fully qualified port name.
        """
        return {port.full_name: port.read() for node in self._nodes for port in node.iter_ports()}

    def parameter_values(self) -> dict[str, object]:
        """Return all current parameter values.

        Returns
        -------
        dict
            Parameter values keyed by fully qualified parameter name.
        """
        return {parameter.full_name: parameter.value for node in self._nodes for parameter in node.iter_parameters()}

    def _run_time(self, time: float) -> None:
        self._evaluate_faults(time)
        for node in self._ex_plan:
            if self._node_due(node):
                self._apply_faults(time, node, {"input", "parameter"})
                node.update(time)
                self._apply_faults(time, node, {"output"})
        self._fault_history[time] = {fault.name: fault.active for fault in self._faults}
        if self._logger is not None:
            self._logger.log_values(time, self)
            self._logger.log_faults(time, self._faults)

    def _evaluate_faults(self, time: float) -> None:
        for fault in self._faults:
            context = FaultContext(self, fault, time, fault.targets)
            fault.evaluate(context) if self.enable_faults else setattr(fault, "_active", False)

    def _apply_faults(self, time: float, node: Node, kinds: set[str]) -> None:
        if not self.enable_faults:
            return
        for fault in self._faults:
            if not fault.active:
                continue
            targets = [target for target in fault.targets if _target_matches_node(target, node, kinds)]
            if targets:
                fault.mutate(FaultContext(self, fault, time, targets))

    def _node_due(self, node: Node) -> bool:
        return self._step_index % self._period_steps[node] == 0

    def _prepare_timing(self, dt: float | None) -> None:
        default_period = 0.01 if dt is None else float(dt)
        periods = []
        for node in self._nodes:
            if node.period is None:
                node.period = default_period
            periods.append(_fraction(node.period))
        self._base_dt_fraction = _fraction_gcd(periods)
        self._base_dt = float(self._base_dt_fraction)
        self._period_steps = {node: int(_fraction(node.period) / self._base_dt_fraction) for node in self._nodes}

    def _simulation_steps(self, t_f: float) -> int:
        final_time = float(t_f) - 1e-12
        if final_time <= 0.0:
            return 0
        return int(Fraction(str(final_time)) // self._base_dt_fraction) + 1

    @staticmethod
    def _progress_bar(show_progress: bool) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not show_progress,
        )

    def _resolve_output_dir(self, log_dir, save_dir, sim_name, batch: int, batches: int) -> Path | None:
        base = log_dir if log_dir is not None else save_dir
        if base is None:
            return None
        path = Path(base)
        if sim_name:
            suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = f"{sim_name}-{suffix}"
            if batches > 1:
                name = f"{name}-batch-{batch + 1}"
            path = path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _default_name(class_name: str, existing: set[str | None]) -> str:
        index = 1
        while f"{class_name}-{index}" in existing:
            index += 1
        return f"{class_name}-{index}"


def _target_matches_node(target, node: Node, kinds: set[str]) -> bool:
    if "input" in kinds and isinstance(target, InputPort) and target.node is node:
        return True
    if "output" in kinds and isinstance(target, OutputPort) and target.node is node:
        return True
    return "parameter" in kinds and getattr(target, "node", None) is node and hasattr(target, "set")


def _fraction(value: float) -> Fraction:
    return Fraction(str(float(value))).limit_denominator(1_000_000)


def _fraction_gcd(values: Iterable[Fraction]) -> Fraction:
    values = tuple(values)
    numerator_gcd = values[0].numerator
    denominator_lcm = values[0].denominator
    for value in values[1:]:
        numerator_gcd = gcd(numerator_gcd, value.numerator)
        denominator_lcm = lcm(denominator_lcm, value.denominator)
    return Fraction(numerator_gcd, denominator_lcm)
