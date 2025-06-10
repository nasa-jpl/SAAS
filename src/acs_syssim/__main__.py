#! python
import argparse

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation
import toml

from syssim import NodeSystem
from syssim.nodes.source import NodeConstant
from syssim.nodes.viz import NodeScope
from syssim.fault.disconect import ZeroFault

from acs_syssim.models.controller import NodeRateControlSimple
from acs_syssim.models.imu import NodeIMUSimple
from acs_syssim.models.rwa import NodeRWASimple
from acs_syssim.models.sixdofsc import NodeSCRigidBodyRotationDynamics
from acs_syssim.models.monsid_sensors import NodeMONSIDCSVLogger
from acs_syssim.models.sru import NodeStellarReferenceUnitSimple
from acs_syssim.models.fault import DiagonalInertiaPerturbFault

def rigid_body_x0(w_low, w_high):
    qx, qy, qz, qw = Rotation.random().as_quat(canonical=True)

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
    "-d",
    "--sim-duration",
    type=float,
    help="Simulation duration in seconds",
    default=50.0,
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
node_sru = NodeStellarReferenceUnitSimple(config=args.node_config, name="node_sru")
node_sru.frequency = 10
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
node_monsid_logger = NodeMONSIDCSVLogger(config=args.node_config, name="monsid_logger")

# Faults Declaration
imu_zero = ZeroFault(
    "imu_zero", trigger_time=10.0
)
imu_zero.active = False
inertia_fault = DiagonalInertiaPerturbFault(
    "inertia_fault",
)
inertia_fault.active = True

system.add_node(node_rb)
system.add_node(node_imu)
system.add_node(node_sru)
system.add_node(node_rwa)
system.add_node(node_control)
system.add_node(node_rate_cmd)
system.add_node(node_viz_true_rate)
system.add_node(node_viz_torque)
system.add_node(node_viz_imu_rate)
system.add_node(node_monsid_logger)
system.add_faults([imu_zero, inertia_fault])

# Node connections
node_rb.o.output_w_sc >> node_imu.i.input_true_angular_rate
node_rb.o.output_w_sc >> node_viz_true_rate.i.scope
node_rb.o.output_q_sc_to_eci >> node_sru.i.input_q_sc2eci

node_imu.o.output_measure_angular_rate >> node_control.i.input_w
node_imu.o.output_measure_angular_rate >> node_viz_imu_rate.i.scope

node_rwa.o.rwa_tau >> node_rb.i.input_tau_external_sc
node_rwa.o.rwa_tau >> node_viz_torque.i.scope
node_rwa.o.rwa_mtm >> node_rb.i.input_mtm_internal_sc

node_control.o.output_tau_cmd >> node_rwa.i.tau_cmd

node_rate_cmd.o.constant_out >> node_control.i.input_w_cmd

node_monsid_logger.i.cmd_torque << node_rwa.o.rwa_tau
node_monsid_logger.i.sens_rate << node_rb.o.output_w_sc
node_monsid_logger.i.sens_imu_rate << node_imu.o.output_measure_angular_rate
node_monsid_logger.i.sens_q_sc_to_eci << node_sru.o.output_q_sc2eci_measure

# Port fault registration
node_imu.o.output_measure_angular_rate.add_fault(imu_zero)

# Param fault registration
node_rb.p.inertia_moment.add_fault(inertia_fault)

print(system)

system.simulate(
    args.sim_duration,
    args.dt,
    save_dir=None,
    sim_name=None,
)
