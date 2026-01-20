# syssim

General-purpose graph-based system simulation with first-class fault injection. The framework models systems as connected nodes (similar to Simulink or Modelica blocks) and lets you inject, schedule, and observe faults directly in the simulation loop.

## Installation

```bash
pip install -e .
```

Dependencies are declared in `pyproject.toml` and pulled in automatically (matplotlib, scipy, rustworkx, toml, tqdm).

## Quick Start

```python
from syssim import NodeSystem
from syssim.nodes.dynamics import NodeStateSpace
from syssim.nodes.source import NodeConstant
from syssim.nodes.viz import NodeScope

sys = NodeSystem()
u = NodeConstant(value=[1.0])
plant = NodeStateSpace(a=[[0.0]], b=[[1.0]], c=[[1.0]], x0=[0.0], sample_period=0.1)
scope = NodeScope(name="y-scope")

sys.add_node(u)
sys.add_node(plant)
sys.add_node(scope)

u.o.constant_out >> plant.i.u
plant.o.y >> scope.i.scope

sys.simulate(t_f=2.0, dt=0.1, sim_name="demo")
```

Run the examples for more complete setups: `python example/step/run.py` or `python example/step_fault/run.py`.

## Core Concepts

- **Nodes**: Units of behavior with input/output ports. Provide `initialize`, `update`, and `finalize` hooks and optional TOML-backed configuration per node name.
- **Ports**: `InputPort` reads data; `OutputPort` pushes data to connected inputs. One output can fan out to many inputs.
- **Parameters**: `NodeParameter` stores model coefficients that can also be faulted like ports.
- **Faults**: Objects that mutate port or parameter values. Faults implement an `action` method and lifecycle hooks. Use built-ins (e.g., `FaultBasic`, `DisconnectFault`, `ZeroFault`) or subclass `Fault` for custom behavior.
- **NodeSystem**: Holds nodes, establishes execution order via topological sort, and orchestrates simulation, scheduling node updates and fault evaluation.

## Configuration via TOML

Provide a TOML file when constructing a node to supply per-node configuration. Keys are node names:

```toml
[plant]
title = "Scope"
xlabel = "Time [s]"
ylabel = "Output"
show = true
save = true
```

In `NodeScope`, these fields control plot labels and saving behavior. Custom nodes can consume any keys they expect.

## Fault Definitions

`FaultBasic` loads its behavior from TOML. A minimal example that forces a port to zero between 2s and 3s with 80% probability:

```toml
[[fault]]
name = "zero-output"

[fault.start-time]
t = 2.0

[fault.duration]
dt = 1.0

[fault.occurance]
p = 0.8

[fault.action]
type = "hold"
index = 0
value = 0.0
```

Attach the fault to a port and register it with the system:

```python
from syssim.fault import FaultBasic

fault = FaultBasic(spec, node=plant, port=plant.o.y)
sys.add_faults(fault)
```

`DisconnectFault` and `ZeroFault` are convenience faults that disconnect (returns NaN) or zero a port after `trigger_time`.

## Execution Model

1. Nodes default to the simulation `dt` unless you pass `sample_period` or `sample_frequency` to the constructor.
2. `NodeSystem.compile()` topologically sorts nodes based on port connections (`Node.depends`). Differential nodes (`NodeDifferential`) declare no dependencies to break algebraic loops.
3. `simulate` builds a schedule over `[0, t_f)` and iterates time steps, updating faults first, then nodes scheduled for that time.
4. Fault activity is recorded; nodes receive fault history in `finalize` to support post-processing.

## Visualization

`NodeScope` records a single input over time and plots it with matplotlib. Configure labels, title, legend, `show`, and `save` via TOML. Figures can be saved under `save_dir/sim_name-start_time/`.

## Contributing

- Keep docstrings in NumPy/SciPy style.
- Prefer pure Python types and NumPy arrays for port payloads.
- Add small, focused example scripts under `example/` when introducing new nodes or faults.

## License

See [LICENSE](../LICENSE).