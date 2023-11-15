The FaultBasic class is meant to be totally initilized from a TOML specification.
This allows static specification of faults.
All faults can be formulated beforehand and centralized in one document for review.
Below is an example of a TOML specification for the basic fault that shows all the options available

```toml
[[nodename.portname]]
name = 'fault'

start-time.t = 10.0
# OR
start-time.gaussian.mean = 50.0
start-time.gaussian.dev = 10.0

duration.dt = 60.0
# OR
duration.gaussian.mean = 5.0
duration.gaussian.dev = 1.0

occurance.p = 0.90

action.type = "random"
action.value.gaussian.mean = 10.0
action.value.gaussian.dev = 1.0
# OR
action.value.uniform.low = 1.0
action.value.uniform.high = 10.0

# OR

action.type = "hold"
action.value = [10.0, 2.0, 3.0]
# OR
action.value = 10.0

action.type = "disconnect"

action.index = [0, 2, 3]
# OR
action.index = 0
action.index.start = 0
action.index.end = 4
```