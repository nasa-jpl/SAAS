"""Test fault injection mechanisms in syssim."""
import numpy as np

from syssim.core import NodeSystem
from syssim.nodes.dynamics import NodeStateSpace
from syssim.nodes.source import NodeConstant
from syssim.fault.basic_fault import FaultBasic
from syssim.fault.disconect import DisconnectFault, ZeroFault


def test_zero_fault_produces_zero():
    """Verify ZeroFault returns zero values after trigger time."""
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
    
    sys = NodeSystem()
    sys.add_node(node_step)
    sys.add_node(node_lti)
    node_step.o.constant_out >> node_lti.i.u
    
    # Add zero fault at t=3.0
    fault = ZeroFault(name="zero-input", trigger_time=3.0)
    node_step.o.constant_out.add_fault(fault)
    sys.add_faults(fault)
    
    # Capture input values
    u_sim = []
    t_sim = []
    
    original_lti_update = node_lti.update
    def capture_update(sim_time):
        t_sim.append(sim_time)
        u_val = node_lti.i.u.read()
        u_sim.append(u_val[0] if u_val is not None else np.nan)
        original_lti_update(sim_time)
    
    node_lti.update = capture_update
    
    sys.simulate(t_f=5.0, dt=dt)
    
    t_array = np.array(t_sim)
    u_array = np.array(u_sim)
    
    # Check values before and after zero fault
    before_zero = u_array[t_array < 3.0]
    after_zero = u_array[t_array >= 3.0]
    
    # Before fault, should be ~1.0
    assert np.all(np.abs(before_zero - 1.0) < 0.1), \
        "Input not nominal before zero fault"
    
    # After fault, should be 0.0
    assert np.all(np.abs(after_zero) < 0.01), \
        "Expected zero values after zero fault"


def test_fault_history_tracking():
    """Verify that fault activation status is properly recorded over time."""
    dt = 0.1
    node_step = NodeConstant(np.array([1.0]), name="step-source")
    
    sys = NodeSystem()
    sys.add_node(node_step)
    
    # Add a fault that triggers at t=2.0
    fault = ZeroFault(name="test-fault", trigger_time=2.0)
    node_step.o.constant_out.add_fault(fault)
    sys.add_faults(fault)
    
    sys.simulate(t_f=5.0, dt=dt)
    
    # Get fault history
    history = sys.get_fault_history()
    
    # Check that history contains entries
    assert len(history) > 0, "Fault history should not be empty"
    
    # Check that fault is inactive before trigger time
    times_before = [t for t in history.keys() if t < 2.0]
    for t in times_before:
        assert history[t]["test-fault"] == False, \
            f"Fault should be inactive at t={t}"
    
    # Check that fault is active at/after trigger time
    times_after = [t for t in history.keys() if t >= 2.0]
    for t in times_after:
        assert history[t]["test-fault"] == True, \
            f"Fault should be active at t={t}"
