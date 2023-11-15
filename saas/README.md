# Simulation for the Analysis of Autonomy at a System-level
This package uses the syssim module to implement a high-level, low-fidelity simulation of a spacecraft with six degrees of freedom.
The user can then inject faults into the simulation to test approached to autonomous failt detection, identification, and recover, commonly refered to as just fault protection (FP).

NOTE: at the moment, the simulation has a lot of the machinery to be general for any fault one might want to simulate with the spacecraft hardware models provided.
However, for the moment, the sim is specialized to a proof of concept scenario simulating a solar panel gimbal fault inspired by the fault that caused the failure of the Mars Global Surveyor (MGS) mission.
This should be adjusted in the future.

The nodes in this simulation are implemented in the package submodules. 
See the READMEs in each individual submodule for descriptions of the nodes implemented in this submodule.
The system definition can be found in the [main](saas/__main__.py) file. 

## Usage
```
usage: saas [-h] [-fc FAULT_CONFIG] [-d SIM_DURATION] [--dt DT] [-b BATCHES] [-n SIM_NAME] [-s SAVE_DIR] node_config

The Simulation for the Analysis of Autonomy at a System-level

positional arguments:
  node_config           Path to the node configuration parameter specification

options:
  -h, --help            show this help message and exit
  -fc FAULT_CONFIG, --fault-config FAULT_CONFIG
                        Path to the fault configuration specification
  -d SIM_DURATION, --sim-duration SIM_DURATION
                        Simulation duration in seconds
  --dt DT               Default simulation timestep
  -b BATCHES, --batches BATCHES
                        Number of simulation batches to run
  -n SIM_NAME, --sim-name SIM_NAME
                        Name of this simulation for record keeping
  -s SAVE_DIR, --save-dir SAVE_DIR
                        Directory to save the ouputs of this simulation
                        ```