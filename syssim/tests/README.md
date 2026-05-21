# syssim Tests

Unit tests for the syssim library.

## Running Tests

### Install test dependencies

```bash
pip install -e ".[test]"
```

### Run all tests

```bash
pytest tests/
```

### Run specific test file

```bash
pytest tests/test_step_response.py
pytest tests/test_fault_injection.py
```

### Run with verbose output

```bash
pytest tests/ -v
```

### Run a specific test function

```bash
pytest tests/test_step_response.py::test_step_response_matches_scipy
pytest tests/test_fault_injection.py::test_basic_fault_injection_modifies_output
```

## Test Coverage

- **test_core_api.py**: Validates dataclass node specs, timestamped samples, fanout/reconnection rules, manual stepping, and strict runtime type checks.

- **test_step_response.py**: Validates that syssim produces equivalent results to scipy for LTI systems
  - `test_step_response_matches_scipy`: Compares step response against scipy reference
  - `test_step_response_without_fault_reaches_steady_state`: Verifies correct steady-state behavior

- **test_fault_injection.py**: Validates fault injection mechanisms
  - `test_zero_fault_produces_zero`: Tests ZeroFault behavior
  - `test_disconnect_fault_mutates_to_nan`: Tests DisconnectFault behavior
  - `test_parameter_fault_applies_before_node_update`: Tests parameter mutation timing
  - `test_fault_write_access_is_limited_to_registered_targets`: Tests context write guards
  - `test_basic_fault_duration_and_hold_action`: Tests FaultBasic with hold action
  - `test_fault_history_tracking`: Verifies fault activation status is recorded over time

- **test_system_logging.py**: Validates mixed-period scheduling and CSV value/fault logging.
