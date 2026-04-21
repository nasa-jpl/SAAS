"""Unit tests for gyroscope sensor node."""

import numpy as np
import pytest

from asteroid_flyby_syssim.gyroscope import NodeGyroscope, GyroscopeConfig
from syssim.core import InputPort, OutputPort
from syssim.core.fault import Fault


def test_gyroscope_output_differs_from_input():
    """Test that gyroscope output differs from true input due to noise/bias."""
    config = GyroscopeConfig(
        bias_rad_s=(0.01, 0.01, 0.01),
        white_noise_std_rad_s=1e-3,
        rng_seed=42,
    )
    gyro = NodeGyroscope(config=config, name="test_gyro")

    # Create a mock input port and connect it
    class MockNode:
        pass

    input_node = MockNode()
    input_port = InputPort("test_input", input_node)
    output_port = OutputPort("test_output", input_node)
    output_port >> gyro.i.angular_velocity

    gyro.initialize()

    # Test 1: With zero input, output should have bias/noise
    w_true = np.array([0.0, 0.0, 0.0], dtype=float)
    output_port.shift_out(w_true, 0.0)
    gyro.update(0.0)

    w_meas = gyro.o.measurement.read()
    assert w_meas is not None
    assert not np.allclose(w_meas, w_true), "Gyro output should differ from zero input (bias/noise present)"
    assert w_meas.shape == (3,), "Gyro output should be 3D vector"

    # Test 2: With non-zero input, output should be different
    w_true_nonzero = np.array([0.1, 0.2, 0.3], dtype=float)
    output_port.shift_out(w_true_nonzero, 0.01)
    gyro.update(0.01)

    w_meas2 = gyro.o.measurement.read()
    assert not np.allclose(w_meas2, w_true_nonzero), "Noisy gyro output should differ from true rate"


def test_gyroscope_scale_factor_error():
    """Test that scale factor errors are applied correctly."""
    scale_error = 0.05  # 5% scale error
    config = GyroscopeConfig(
        bias_rad_s=(0.0, 0.0, 0.0),
        scale_errors=(scale_error, scale_error, scale_error),
        white_noise_std_rad_s=0.0,  # No noise for this test
        bias_random_walk_std_rad_s2=0.0,  # No random walk
        rng_seed=42,
    )
    gyro = NodeGyroscope(config=config, name="test_gyro_scale")

    # Connect mock ports
    class MockNode:
        pass

    input_node = MockNode()
    input_port = InputPort("test_input", input_node)
    output_port = OutputPort("test_output", input_node)
    output_port >> gyro.i.angular_velocity

    gyro.initialize()

    # Test with known input
    w_true = np.array([1.0, 1.0, 1.0], dtype=float)
    output_port.shift_out(w_true, 0.0)
    gyro.update(0.0)

    w_meas = gyro.o.measurement.read()
    expected = w_true * (1.0 + scale_error)  # Should be scaled by (1 + 5%)
    assert np.allclose(w_meas, expected, atol=1e-6), \
        f"Expected {expected}, got {w_meas}"


def test_gyroscope_bias():
    """Test that constant bias is applied correctly."""
    bias = np.array([0.01, 0.02, 0.03], dtype=float)
    config = GyroscopeConfig(
        bias_rad_s=tuple(bias),
        scale_errors=(0.0, 0.0, 0.0),
        white_noise_std_rad_s=0.0,
        bias_random_walk_std_rad_s2=0.0,
        rng_seed=42,
    )
    gyro = NodeGyroscope(config=config, name="test_gyro_bias")

    # Connect mock ports
    class MockNode:
        pass

    input_node = MockNode()
    input_port = InputPort("test_input", input_node)
    output_port = OutputPort("test_output", input_node)
    output_port >> gyro.i.angular_velocity

    gyro.initialize()

    # Test with zero input
    w_true = np.array([0.0, 0.0, 0.0], dtype=float)
    output_port.shift_out(w_true, 0.0)
    gyro.update(0.0)

    w_meas = gyro.o.measurement.read()
    assert np.allclose(w_meas, bias, atol=1e-6), \
        f"Expected bias {bias}, got {w_meas}"


def test_gyroscope_random_walk():
    """Test that bias random walk accumulates over time."""
    config = GyroscopeConfig(
        bias_rad_s=(0.0, 0.0, 0.0),
        scale_errors=(0.0, 0.0, 0.0),
        white_noise_std_rad_s=0.0,
        bias_random_walk_std_rad_s2=1e-3,  # Large for testing
        rng_seed=42,
    )
    gyro = NodeGyroscope(config=config, name="test_gyro_random_walk")

    # Connect mock ports
    class MockNode:
        pass

    input_node = MockNode()
    input_port = InputPort("test_input", input_node)
    output_port = OutputPort("test_output", input_node)
    output_port >> gyro.i.angular_velocity

    gyro.initialize()

    # Update multiple times with zero input
    w_true = np.array([0.0, 0.0, 0.0], dtype=float)
    w_meas_history = []

    for i in range(100):
        output_port.shift_out(w_true, float(i) * 0.01)
        gyro.update(float(i) * 0.01)
        w_meas = gyro.o.measurement.read()
        w_meas_history.append(w_meas.copy())

    w_meas_history = np.array(w_meas_history)

    # Check that bias drift is increasing (random walk)
    bias_drift_magnitude = np.linalg.norm(w_meas_history, axis=1)
    assert bias_drift_magnitude[-1] > bias_drift_magnitude[0], \
        "Random walk bias should increase in magnitude over time"


def test_gyroscope_fault_injection():
    """Test that faults can modify bias parameter."""
    config = GyroscopeConfig(
        bias_rad_s=(0.0, 0.0, 0.0),
        scale_errors=(0.0, 0.0, 0.0),
        white_noise_std_rad_s=0.0,
        bias_random_walk_std_rad_s2=0.0,
        rng_seed=42,
    )
    gyro = NodeGyroscope(config=config, name="test_gyro_fault")

    # Connect mock ports
    class MockNode:
        pass

    input_node = MockNode()
    input_port = InputPort("test_input", input_node)
    output_port = OutputPort("test_output", input_node)
    output_port >> gyro.i.angular_velocity

    gyro.initialize()

    # Add a fault that sets bias to [0.1, 0.0, 0.0]
    def fault_action(bias_val):
        bias_val[0] = 0.1
        return bias_val

    # Create and attach fault to bias parameter
    fault = Fault(name="bias_fault", active=False, action=fault_action, on_object=gyro._param_bias)
    gyro._param_bias.add_fault(fault)

    # Test without fault
    w_true = np.array([0.0, 0.0, 0.0], dtype=float)
    output_port.shift_out(w_true, 0.0)
    gyro.update(0.0)
    w_meas_no_fault = gyro.o.measurement.read()

    # Activate fault
    fault.active = True
    output_port.shift_out(w_true, 0.01)
    gyro.update(0.01)
    w_meas_with_fault = gyro.o.measurement.read()

    # With fault, first component should be 0.1
    assert w_meas_with_fault[0] > w_meas_no_fault[0], \
        "Faulted bias should increase first component"
    assert np.isclose(w_meas_with_fault[0], 0.1, atol=1e-6), \
        f"Expected faulted bias x-component ≈ 0.1, got {w_meas_with_fault[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
