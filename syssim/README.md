# syssim

General-purpose graph-based dynamical system simulation with first-class fault injection. Systems are assembled from typed nodes, timestamped ports, faultable parameters, and a `NodeSystem` orchestrator.

## Installation

```bash
pip install -e .
```

Dependencies are declared in `pyproject.toml` and pulled in automatically.

## Quick Start

```python
from syssim import NodeSystem
from syssim.fault import ZeroFault
from syssim.nodes.dynamics import NodeStateSpace
from syssim.nodes.source import NodeConstant
from syssim.nodes.viz import NodeScope, NodeScopeConfig

sys = NodeSystem()
u = NodeConstant(value=[1.0])
plant = NodeStateSpace(a=[[0.0]], b=[[1.0]], c=[[1.0]], x0=[0.0], sample_period=0.1)
scope = NodeScope(name="y-scope", config=NodeScopeConfig(show=True))
fault = ZeroFault(name="zero-input", trigger_time=1.0, targets=[u.o.constant_out])

sys.add_node(u)
sys.add_node(plant)
sys.add_node(scope)
sys.add_faults(fault)

u.o.constant_out >> plant.i.u
plant.o.y >> scope.i.scope

sys.simulate(t_f=2.0, dt=0.1, log_dir="runs/demo")
```

Run the examples for more complete setups: `python example/step/run.py`, `python example/step_fault/run.py`, or `python example/integrator_chain_delay/run.py`.

## Core Concepts

- **Nodes**: Units of behavior with top-level dataclass specs for inputs, outputs, parameters, and config. Nodes bind those specs through the generic `Node` base and implement `initialize`, `update`, and `finalize` hooks.
- **Ports**: `InputPort[T]` and `OutputPort[T]` carry `PortSample(value, time)`. One output can fan out to many inputs; each input can connect to one output.
- **Parameters**: `NodeParameter[T]` stores model coefficients and can be targeted by faults.
- **Faults**: Faults implement `trigger(context) -> bool` and `mutate(context) -> None`. Contexts provide read access to the full system and write access only to registered targets.
- **NodeSystem**: Holds nodes/faults, topologically orders nodes, schedules mixed update periods, supports closed-loop `simulate`, manual `step`, strict runtime type checks, and optional CSV logs.

## Defining Nodes

```python
from dataclasses import dataclass

from syssim import EmptySpec, InputPort, Node, NodeParameter, OutputPort, input_port, output_port, parameter

@dataclass
class GainInputs:
	u: InputPort[float] = input_port(float)


@dataclass
class GainOutputs:
	y: OutputPort[float] = output_port(float)


@dataclass
class GainParameters:
	gain: NodeParameter[float] = parameter(2.0, value_type=float)


class Gain(Node[GainInputs, GainOutputs, GainParameters, EmptySpec]):
	Inputs = GainInputs
	Outputs = GainOutputs
	Parameters = GainParameters

	def update(self, sim_time: float):
		sample = self.i.u.read()
		self.o.y.write(sample.value * self.p.gain.value, sim_time)
```

With this pattern, static analysis sees `gain.i.u`, `gain.o.y`, and `gain.p.gain` before the code runs.

`read()` always returns a timestamped `PortSample`; it can also be unpacked as `value, sample_time = self.i.u.read()`.

## Configuration

Node configuration uses dataclasses only. Each configurable node exposes a config dataclass with defaults:

```python
from syssim.nodes.viz import NodeScope, NodeScopeConfig

scope = NodeScope(
	name="scope",
	config=NodeScopeConfig(
		title="Step Response",
		xlabel="Time [s]",
		ylabel="Output",
		show=False,
		save=True,
	),
)
```

Custom nodes should define their own top-level config dataclass when they need configuration.

## Fault Definitions

`FaultBasic` also uses a dataclass config. This example forces a port to zero between 2s and 3s with 80% probability:

Attach the fault to one or more targets and register it with the system:

```python
from syssim.fault import FaultBasic, FaultBasicConfig

fault = FaultBasic(
	FaultBasicConfig(
		name="zero-output",
		start_time=2.0,
		duration=1.0,
		occurrence=0.8,
		action="hold",
		index=0,
		value=0.0,
	),
	port=plant.o.y,
)
sys.add_faults(fault)
```

`DisconnectFault` and `ZeroFault` are convenience faults that disconnect (returns NaN) or zero a port or parameter after `trigger_time`.

## Execution Model

1. Nodes default to the simulation `dt` unless you pass `sample_period` or `sample_frequency`.
2. `NodeSystem.compile()` topologically sorts nodes from port connections. Differential nodes declare no current-step dependencies.
3. `simulate` builds a base timestep from node periods and updates only nodes due at each time.
4. Faults are evaluated each timestep. Parameter/input faults apply before node update; output faults apply after node update and propagate to connected inputs.
5. `NodeSystem.step(dt)` manually advances the system and returns every port sample as a dictionary keyed by `node.port`.

## CSV Logs

Pass `log_dir` to `simulate` to write `values.csv` and `faults.csv`:

```python
sys.simulate(t_f=5.0, dt=0.1, log_dir="runs/clean")
```

`values.csv` records port and parameter values. `faults.csv` records enabled, triggered, active, target, and start/end event state.

## Visualization

`NodeScope` records a single input over time and plots it with matplotlib. Configure labels, title, legend, `show`, and `save` with `NodeScopeConfig`. Figures can be saved under the simulation output directory.

## Contributing

- Keep docstrings concise and use NumPy/SciPy style where useful.
- Prefer pure Python types and NumPy arrays for port payloads.
- Add small, focused example scripts under `example/` when introducing new nodes or faults.

## License

See [LICENSE](../LICENSE).