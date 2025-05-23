#! python
import argparse

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation
import toml

from syssim import NodeSystem
from syssim.nodes.source import NodeConstant
from syssim.nodes.viz import NodeScope

from acs_syssim.models.controller import NodeRateControlSimple
from acs_syssim.models.imu import NodeIMUSimple
from acs_syssim.models.rwa import NodeRWASimple
from acs_syssim.models.sixdofsc import NodeSCRigidBodyRotationDynamics
from acs_syssim.models.monsid_sensors import NodeMONSIDCSVLogger



def rigid_body_x0(w_low, w_high):
    # qx, qy, qz, qw = Rotation.from_euler("x", 0, degrees=True).as_quat()
    qx, qy, qz, qw = Rotation.random().as_quat(canonical=True)
    # qx, qy, qz, qw = Rotation.identity().as_quat(canonical=True)

    q0 = np.array([qw, qx, qy, qz])

    w = default_rng().uniform(w_low, w_high)
    theta = default_rng().uniform(0, np.pi)
    phi = default_rng().uniform(-np.pi, np.pi)
    w_sc = np.array(
        [
            w * np.sin(theta) * np.cos(phi),
            w * np.sin(theta) * np.sin(phi),
            w * np.cos(theta),
        ]
    )
    # w_sc = np.array([w, 0, 0])
    # w_sc = np.array([0, 0, 0])

    return np.concatenate([q0, w_sc])


parser = argparse.ArgumentParser(
    description="The Simulation for the Analysis of Autonomy at a System-level",
    prog="saas",
)
parser.add_argument(
    "node_config",
    type=str,
    help="Path to the node configuration parameter specification",
)
parser.add_argument(
    "-f",
    "--fault-config",
    type=str,
    help="Path to the fault configuration parameter specification",
    default=None,
)
parser.add_argument(
    "-d",
    "--sim-duration",
    type=float,
    help="Simulation duration in seconds",
    default=60.0,
    dest="sim_duration",
)
parser.add_argument("--dt", help="Default simulation timestep", type=float, default=1e-2)


args = parser.parse_args()

config = toml.load(args.node_config)

x0_rb = rigid_body_x0(0.0, 6.0)

system = NodeSystem()
# TODO Move the initialization of the state of this component to the init method of the component. 
node_rb = NodeSCRigidBodyRotationDynamics(x0_rb, config=args.node_config)
node_imu = NodeIMUSimple(config=args.node_config, name="node_imu")
node_imu.frequency = 100
node_rwa = NodeRWASimple(np.zeros((3,)), config=args.node_config, name="rwa")
node_rwa.frequency = 60.0
node_control = NodeRateControlSimple(config=args.node_config)
node_control.frequency = 1.0
node_rate_cmd = NodeConstant(np.array([0.0, 0.0, 0.0]), config=args.node_config)
node_viz_true_rate = NodeScope(config=args.node_config, name="viz_true_rate")
node_viz_true_rate.frequency = 100
node_viz_torque = NodeScope(config=args.node_config, name="viz_torque")
node_viz_torque.frequency = 100
node_viz_imu_rate = NodeScope(config=args.node_config, name="viz_imu_rate")
node_viz_imu_rate.frequency = 100
node_inertia = NodeConstant(np.diag([1.0, 1.0, 1.0]), config=args.node_config)
node_monsid_logger = NodeMONSIDCSVLogger(config=args.node_config, name="monsid_logger")

system.add_node(node_rb)
system.add_node(node_imu)
system.add_node(node_rwa)
system.add_node(node_control)
system.add_node(node_rate_cmd)
system.add_node(node_viz_true_rate)
system.add_node(node_viz_torque)
system.add_node(node_viz_imu_rate)
system.add_node(node_inertia)
system.add_node(node_monsid_logger)

node_rb.o.output_w_sc >> node_imu.i.input_true_angular_rate
node_rb.o.output_w_sc >> node_viz_true_rate.i.scope

node_imu.o.output_measure_angular_rate >> node_control.i.input_w
node_imu.o.output_measure_angular_rate >> node_viz_imu_rate.i.scope

node_rwa.o.rwa_tau >> node_rb.i.input_tau_external_sc
node_rwa.o.rwa_tau >> node_viz_torque.i.scope
node_rwa.o.rwa_mtm >> node_rb.i.input_mtm_internal_sc

node_control.o.output_tau_cmd >> node_rwa.i.tau_cmd

node_rate_cmd.o.constant_out >> node_control.i.input_w_cmd

node_inertia.o.constant_out >> node_control.i.input_sc_inertia_moment
node_inertia.o.constant_out >> node_rb.i.input_inertia_moment

node_monsid_logger.i.cmd_torque << node_rwa.o.rwa_tau
node_monsid_logger.i.sens_rate << node_rb.o.output_w_sc
node_monsid_logger.i.sens_imu_rate << node_imu.o.output_measure_angular_rate

if args.fault_config is not None:
    system.add_faults(args.fault_config)

print(system)

system.simulate(
    args.sim_duration,
    args.dt,
    save_dir=None,
    sim_name=None,
)
