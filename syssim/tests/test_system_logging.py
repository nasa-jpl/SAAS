from dataclasses import dataclass

import numpy as np

from syssim.core import EmptySpec, InputPort, Node, NodeSystem, input_port
from syssim.fault.disconnect import ZeroFault
from syssim.nodes.source import NodeConstant


@dataclass
class RecorderInputs:
    inp: InputPort[np.ndarray] = input_port(np.ndarray)


class Recorder(Node[RecorderInputs, EmptySpec, EmptySpec, EmptySpec]):
    Inputs = RecorderInputs

    def __init__(self, **kwargs):
        self.times = []
        super().__init__(**kwargs)

    def update(self, sim_time: float):
        self.times.append(sim_time)


def test_mixed_sample_period_schedule():
    source = NodeConstant(np.array([1.0]), sample_period=0.1, name="source")
    recorder = Recorder(sample_period=0.2, name="recorder")
    source.o.constant_out >> recorder.i.inp
    system = NodeSystem()
    system.add_node(source)
    system.add_node(recorder)

    system.simulate(t_f=0.5, dt=0.1)

    assert recorder.times == [0.0, 0.2, 0.4]


def test_csv_value_and_fault_logs_are_written(tmp_path):
    source = NodeConstant(np.array([1.0]), sample_period=0.1, name="source")
    fault = ZeroFault(name="zero", trigger_time=0.1, targets=[source.o.constant_out])
    system = NodeSystem()
    system.add_node(source)
    system.add_faults(fault)

    system.simulate(t_f=0.3, dt=0.1, log_dir=tmp_path)

    values_log = tmp_path / "values.csv"
    faults_log = tmp_path / "faults.csv"
    assert values_log.exists()
    assert faults_log.exists()
    assert "source" in values_log.read_text()
    assert "zero" in faults_log.read_text()
    assert "start" in faults_log.read_text()