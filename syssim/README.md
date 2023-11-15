# syssim
This module provides a general purpose framework for simulating systems by breaking up the system into individual units of behavior and then connecting them together.
This is the same capability provided by well known tools such as MATLAB Simulink and Modelica.
However, this framework also provides fault injection into the simulation as a first class feature.

## Components
### Nodes
Nodes are the basic units of system behavior.
These are just like blocks in Simulink. 
Nodes are built by the user to encode units of behavior that they want to simulate.
Nodes can have config variables defined via TOML like the following:
```toml
[nodename]
config1="hellow world"
config2=1.0
config3=true
# etc...
```
### Ports
Nodes contain ports that allow data to be moved into and out of the node.
There are InputPorts and OutputPorts for these purposes.

### Faults
Faults are defined in the API as the ability to change the data passing through a port programatically.
As such, objects implementing a fault will need to have an ```action()```which applys the change to the data, and a ```is_active()``` to determine if the action should be applied.
See the ```FaultBasic``` class for an example of how this is done.

## Node System
Nodes are instatiated and then added to an instance of ```NodeSystem```. 
They can then be connected and the whole system can be simulated.
Nodes can also output results to a plot (e.g., [scope](syssim/nodes/viz.py)).
For an example of how to set up a ```NodeSystem``` please see the [example](example/) directory.