"""Test syssim step response against scipy reference implementation."""
from dataclasses import dataclass

import numpy as np
from scipy import signal

from syssim.core import EmptySpec, InputPort, Node, NodeSystem, input_port
from syssim.nodes.dynamics import NodeStateSpace
from syssim.nodes.source import NodeConstant


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


def test_step_response_matches_scipy():
    """Verify syssim LTI system produces same step response as scipy.
    
    Compares a simple first-order system (transfer function 1/(s+1))
    simulated in both frameworks over 10 seconds.
    """
    # Reference solution from scipy
    lti = signal.lti([1.0], [1.0, 1.0])
    t_ref = np.linspace(0, 10, 1001)
    _, y_ref = signal.step(lti, T=t_ref)
    
    # Build equivalent system in syssim
    dt = 0.01
    node_step = NodeConstant(np.array([1.0]), name="step-source")
    node_lti = NodeStateSpace(
        a=np.array([[-1.0]]),
        b=np.array([[1.0]]),
        c=np.array([[1.0]]),
        x0=np.array([0.0]),
        sample_period=dt,
        name="state-space-filter",
    )
    recorder = Recorder(sample_period=dt, name="recorder")
    
    sys = NodeSystem()
    sys.add_node(node_step)
    sys.add_node(node_lti)
    sys.add_node(recorder)
    node_step.o.constant_out >> node_lti.i.u
    node_lti.o.y >> recorder.i.inp
    
    # Simulate
    sys.simulate(t_f=10.0, dt=dt)
    
    # Interpolate scipy result to match syssim time points
    y_ref_interp = np.interp(recorder.times, t_ref, y_ref.flatten())
    y_sim_array = np.array([item[0] for item in recorder.values])
    
    # Allow small numerical differences due to integration method
    np.testing.assert_allclose(
        y_sim_array,
        y_ref_interp,
        rtol=0.05,  # 5% relative tolerance
        atol=0.01,  # absolute tolerance for near-zero values
        err_msg="syssim step response does not match scipy reference"
    )


def test_step_response_without_fault_reaches_steady_state():
    """Verify nominal system reaches expected steady-state value.
    
    For a unit step input to transfer function 1/(s+1), the steady-state
    output should approach 1.0.
    """
    dt = 0.01
    node_step = NodeConstant(np.array([1.0]), name="step-source")
    node_lti = NodeStateSpace(
        a=np.array([[-1.0]]),
        b=np.array([[1.0]]),
        c=np.array([[1.0]]),
        x0=np.array([0.0]),
        sample_period=dt,
        name="state-space-filter",
    )
    recorder = Recorder(sample_period=dt, name="recorder")
    
    sys = NodeSystem()
    sys.add_node(node_step)
    sys.add_node(node_lti)
    sys.add_node(recorder)
    node_step.o.constant_out >> node_lti.i.u
    node_lti.o.y >> recorder.i.inp
    
    # Capture final state
    sys.simulate(t_f=10.0, dt=dt)
    
    # After 10 seconds (10 time constants), should be very close to 1.0
    final_state = recorder.values[-1][0]

    np.testing.assert_allclose(
        final_state,
        1.0,
        rtol=0.01,
        err_msg="System did not reach expected steady state"
    )
