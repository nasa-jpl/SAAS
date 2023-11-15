#! python
import sys

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from syssim.core import NodeSystem
from syssim.nodes.dynamics import NodeStateSpace
from syssim.nodes.source import NodeConstant
from syssim.nodes.viz import NodeScope


""" 
An example script showing how to build a system that simulate the step response of an LTI System. This version includes a fault. Therefore, the output shoule noticably differ from the reference nominal response provided by the Scipy library.

Run this script as:
python run.py 'path to config.toml' 'path to basic_fault_config.toml
"""

# Setup system in scipy
lti = signal.lti([1.0], [1.0, 1.0])
t, y = signal.step(lti, T=np.linspace(0, 10, 1000))

# Setup system in syssim
node_step = NodeConstant(np.array([1]), name="step-source")
node_lti = NodeStateSpace(
    np.array([[-1]]),
    np.array([[1]]),
    np.array([[1]]),
    np.array([0]),
    name="state-space-filter",
)
node_scope = NodeScope(name="scope", config=sys.argv[1])

# Setup system by adding nodes and specifying connections
n_sys = NodeSystem()
n_sys.add_node(node_step)
n_sys.add_node(node_lti)
n_sys.add_node(node_scope)
n_sys.add_faults(sys.argv[2])
node_step["output"] = node_lti["input_u"]
node_lti["output_y"] = node_scope["input_scope"]

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
