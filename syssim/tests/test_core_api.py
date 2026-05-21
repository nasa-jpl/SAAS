from dataclasses import dataclass

import numpy as np
import pytest

from syssim.core import EmptySpec, InputPort, Node, NodeParameter, NodeSystem, OutputPort, input_port, output_port, parameter
from syssim.nodes.source import NodeConstant
from syssim.nodes.viz import NodeScope, NodeScopeConfig


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


@dataclass
class ArraySourceOutputs:
    y: OutputPort[np.ndarray] = output_port(np.ndarray, dtype=float, shape=(2,))


class ArraySource(Node[EmptySpec, ArraySourceOutputs, EmptySpec, EmptySpec]):
    Outputs = ArraySourceOutputs

    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    def update(self, sim_time: float):
        self.o.y.write(self.value, sim_time)


def test_dataclass_node_run_step_returns_timestamped_outputs():
    node = Gain(name="gain")

    outputs = node.run_step(0.25, {"u": 3.0})

    assert outputs["y"].value == 6.0
    assert outputs["y"].time == 0.25


def test_config_is_dataclass_only():
    scope = NodeScope(config=NodeScopeConfig(title="Custom", show=False))

    assert scope.config.title == "Custom"
    assert scope.config.show is False
    with pytest.raises(TypeError):
        NodeScope(config={"show": False})
    with pytest.raises(TypeError):
        NodeScope(config="config-file")


def test_output_fanout_and_input_single_source_enforced():
    source_1 = NodeConstant(1.0, name="source-1")
    source_2 = NodeConstant(2.0, name="source-2")
    sink_1 = Gain(name="sink-1")
    sink_2 = Gain(name="sink-2")

    source_1.o.constant_out >> sink_1.i.u
    sink_2.i.u << source_1.o.constant_out

    assert sink_1.i.u.source is source_1.o.constant_out
    assert sink_2.i.u.source is source_1.o.constant_out
    with pytest.raises(ValueError):
        source_2.o.constant_out >> sink_1.i.u


def test_system_step_returns_all_port_samples():
    source = NodeConstant(3.0, sample_period=0.1, name="source")
    gain = Gain(sample_period=0.1, name="gain")
    source.o.constant_out >> gain.i.u
    system = NodeSystem()
    system.add_node(source)
    system.add_node(gain)

    samples = system.step(0.1)

    assert samples["source.constant_out"].value == 3.0
    assert samples["gain.u"].value == 3.0
    assert samples["gain.y"].value == 6.0
    assert samples["gain.y"].time == 0.1


def test_strict_type_mode_checks_python_types():
    source = NodeConstant("not-a-float", sample_period=0.1, name="source")
    gain = Gain(sample_period=0.1, name="gain")
    source.o.constant_out >> gain.i.u
    system = NodeSystem(strict_types=True)
    system.add_node(source)
    system.add_node(gain)

    with pytest.raises(TypeError):
        system.step(0.1)


def test_strict_type_mode_checks_ndarray_shape_and_dtype():
    source = ArraySource(np.array([1, 2], dtype=int), sample_period=0.1, name="array-source")
    system = NodeSystem(strict_types=True)
    system.add_node(source)

    with pytest.raises(TypeError):
        system.step(0.1)