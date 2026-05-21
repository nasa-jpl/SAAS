# Faults

`FaultBasic` is configured with a dataclass:

```python
from syssim.fault import FaultBasic, FaultBasicConfig

fault = FaultBasic(
	FaultBasicConfig(
		name="fault",
		start_time=10.0,
		duration=60.0,
		occurrence=0.9,
		action="hold",
		value=10.0,
		index=0,
	),
	port=node.o.output,
)
```

For stochastic runs, pass zero-argument callables to `start_time_distribution`,
`duration_distribution`, or `value_distribution`.
```