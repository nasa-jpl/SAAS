import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from syssim.core import NodeSystem
from syssim.nodes.dynamics import NodeStateSpace
from syssim.nodes.source import NodeConstant
from syssim.nodes.viz import NodeScope, NodeScopeConfig
from syssim.fault.disconnect import ZeroFault


""" 
An example script showing how to build a system that simulate the step response of an LTI System. This version includes a fault. Therefore, the output shoule noticably differ from the reference nominal response provided by the Scipy library.
"""

# Setup system in scipy
lti = signal.lti([1.0], [1.0, 1.0])
t, y = signal.step(lti, T=np.linspace(0, 10, 1000))

# Setup system in syssim
node_step = NodeConstant(np.array([1.0]), name="step-source")
node_lti = NodeStateSpace(
    a=np.array([[-1.0]]),
    b=np.array([[1.0]]),
    c=np.array([[1.0]]),
    x0=np.array([0.0]),
    name="state-space-filter",
)
node_scope = NodeScope(name="scope", config=NodeScopeConfig(show=True))
zero_fault = ZeroFault(name="zero-fault", trigger_time=3.0, targets=[node_step.o.constant_out])

# Setup system by adding nodes and specifying connections
n_sys = NodeSystem()
n_sys.add_node(node_step)
n_sys.add_node(node_lti)
n_sys.add_node(node_scope)
n_sys.add_faults(zero_fault)

node_step.o.constant_out >> node_lti.i.u
node_lti.o.y >> node_scope.i.scope

# Can print the system to show some info about it...
print(n_sys)

# Simulate for 10 seconds
n_sys.simulate(10)

# Plot the Scipy response. Should be the same as the syssim response.
plt.figure()
plt.plot(t, y)
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Filter Step Response (Scipy)")
plt.grid(True)
plt.show()
