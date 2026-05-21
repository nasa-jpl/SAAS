"""Test context-aware fault injection mechanisms in syssim."""
from dataclasses import dataclass

import numpy as np
import pytest

from syssim.core import EmptySpec, Fault, FaultContext, InputPort, Node, NodeParameter, NodeSystem, OutputPort, input_port, output_port, parameter
from syssim.nodes.source import NodeConstant
from syssim.fault.basic_fault import FaultBasic, FaultBasicConfig
from syssim.fault.disconnect import DisconnectFault, ZeroFault


@dataclass
class RecorderInputs:
    inp: InputPort[np.ndarray] = input_port(np.ndarray)


class Recorder(Node[RecorderInputs, EmptySpec, EmptySpec, EmptySpec]):
    Inputs = RecorderInputs

    def __init__(self, **kwargs):
        self.times = []
        self.values = []
        super().__init__(**kwargs)

    def update(self, sim_time: float):
        sample = self.i.inp.read()
        self.times.append(sim_time)
        self.values.append(sample.value.copy())


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
        self.o.y.write(self.i.u.read().value * self.p.gain.value, sim_time)


def test_zero_fault_mutates_registered_output_target():
    source = NodeConstant(np.array([1.0]), sample_period=0.1, name="source")
    recorder = Recorder(sample_period=0.1, name="recorder")
    source.o.constant_out >> recorder.i.inp
    fault = ZeroFault(name="zero-source", trigger_time=0.2, targets=[source.o.constant_out])
    system = NodeSystem()
    system.add_node(source)
    system.add_node(recorder)
    system.add_faults(fault)

    system.simulate(t_f=0.4, dt=0.1)

    values = np.array([item[0] for item in recorder.values])
    assert np.all(values[np.array(recorder.times) < 0.2] == 1.0)
    assert np.all(values[np.array(recorder.times) >= 0.2] == 0.0)


def test_disconnect_fault_mutates_to_nan():
    source = NodeConstant(np.array([1.0]), sample_period=0.1, name="source")
    recorder = Recorder(sample_period=0.1, name="recorder")
    source.o.constant_out >> recorder.i.inp
    fault = DisconnectFault(name="disconnect-source", trigger_time=0.0, targets=[source.o.constant_out])
    system = NodeSystem()
    system.add_node(source)
    system.add_node(recorder)
    system.add_faults(fault)

    system.step(0.1)

    assert np.isnan(recorder.values[-1][0])


def test_parameter_fault_applies_before_node_update():
    source = NodeConstant(3.0, sample_period=0.1, name="source")
    gain = Gain(sample_period=0.1, name="gain")
    source.o.constant_out >> gain.i.u
    fault = ZeroFault(name="zero-gain", trigger_time=0.0, targets=[gain.p.gain])
    system = NodeSystem()
    system.add_node(source)
    system.add_node(gain)
    system.add_faults(fault)

    samples = system.step(0.1)

    assert samples["gain.y"].value == 0.0


class BadFault(Fault):
    def __init__(self, illegal_target, **kwargs):
        self.illegal_target = illegal_target
        super().__init__(**kwargs)

    def trigger(self, context: FaultContext) -> bool:
        return True

    def mutate(self, context: FaultContext) -> None:
        context.write(self.illegal_target, np.array([0.0]))


def test_fault_write_access_is_limited_to_registered_targets():
    source = NodeConstant(np.array([1.0]), sample_period=0.1, name="source")
    recorder = Recorder(sample_period=0.1, name="recorder")
    source.o.constant_out >> recorder.i.inp
    fault = BadFault(recorder.i.inp, name="bad", targets=[source.o.constant_out])
    system = NodeSystem()
    system.add_node(source)
    system.add_node(recorder)
    system.add_faults(fault)

    with pytest.raises(PermissionError):
        system.step(0.1)


def test_basic_fault_duration_and_hold_action():
    source = NodeConstant(np.array([1.0]), sample_period=0.1, name="source")
    recorder = Recorder(sample_period=0.1, name="recorder")
    source.o.constant_out >> recorder.i.inp
    config = FaultBasicConfig(
        name="basic-zero",
        start_time=0.1,
        duration=0.2,
        occurrence=1.0,
        action="hold",
        index=0,
        value=0.0,
    )
    fault = FaultBasic(config, port=source.o.constant_out)
    system = NodeSystem()
    system.add_node(source)
    system.add_node(recorder)
    system.add_faults(fault)

    system.simulate(t_f=0.4, dt=0.1)

    by_time = {round(time, 1): value[0] for time, value in zip(recorder.times, recorder.values)}
    assert by_time[0.0] == 1.0
    assert by_time[0.1] == 0.0
    assert by_time[0.2] == 0.0
    assert by_time[0.3] == 1.0


def test_fault_history_tracking():
    """Verify that fault activation status is properly recorded over time."""
    source = NodeConstant(np.array([1.0]), sample_period=0.1, name="source")
    fault = ZeroFault(name="test-fault", trigger_time=0.2, targets=[source.o.constant_out])
    system = NodeSystem()
    system.add_node(source)
    system.add_faults(fault)

    system.simulate(t_f=0.5, dt=0.1)

    history = system.get_fault_history()
    assert history
    assert all(not status["test-fault"] for time, status in history.items() if time < 0.2)
    assert all(status["test-fault"] for time, status in history.items() if time >= 0.2)
