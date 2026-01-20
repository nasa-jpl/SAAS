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

- **test_step_response.py**: Validates that syssim produces equivalent results to scipy for LTI systems
  - `test_step_response_matches_scipy`: Compares step response against scipy reference
  - `test_step_response_without_fault_reaches_steady_state`: Verifies correct steady-state behavior

- **test_fault_injection.py**: Validates fault injection mechanisms
  - `test_basic_fault_injection_modifies_output`: Tests FaultBasic with hold action
  - `test_disconnect_fault_produces_nan`: Tests DisconnectFault behavior
  - `test_zero_fault_produces_zero`: Tests ZeroFault behavior
  - `test_fault_history_tracking`: Verifies fault activation status is recorded over time
