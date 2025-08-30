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

from acs_syssim.models.controller import NodePointingControlSimple
from acs_syssim.models.imu import NodeIMUSimple
from acs_syssim.models.rwa import NodeRWASimple
from acs_syssim.models.sixdofsc import NodeSCRigidBodyRotationDynamics
from acs_syssim.models.monsid import NodeMONSIDDiagnoser, NodeFaultPrinter
from acs_syssim.models.sru import NodeStellarReferenceUnitSimple
from acs_syssim.models.fault import DiagonalInertiaPerturbFault
from acs_syssim.models.adder import AdderNode, ConcatNode
from acs_syssim.models.mixer import (
    ReactionWheelMixerNode,
    InternalAngularMomentumMuxerNode,
)
from acs_syssim.models.encoder import NodeWheelEncoder
from acs_syssim.models.estimator import NodeKalmanEstimator


def rigid_body_x0(w_low, w_high):
    qx, qy, qz, qw = Rotation.random().as_quat(canonical=True)
    # qx, qy, qz, qw = Rotation.from_euler('X', 90, degrees=True).as_quat(canonical=True)

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
    # w_sc = np.zeros(3) 
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
    default=20.0,
    dest="sim_duration",
)
parser.add_argument(
    "--dt", help="Default simulation timestep", type=float, default=1e-2
)


args = parser.parse_args()

config = toml.load(args.node_config)

x0_rb = rigid_body_x0(0.0, 6.0)

# Create body fixed direction vectors for each of six reaction wheels. 
rw_body_axis = [
    [0.0, 0.0, 1.0],  # aligned with z-axis
    [(9 / 8) ** 0.5, 0.0, -1 / 3],
    [-((2 / 9) ** 0.5), (2 / 3) ** 0.5, -1 / 3],
    [-((2 / 9) ** 0.5), -((2 / 3) ** 0.5), -1 / 3],
    [0.0, 0.0, -1.0],
    [(9 / 8) ** 0.5, 0.0, 1 / 3],
    [-((2 / 9) ** 0.5), (2 / 3) ** 0.5, 1 / 3],
    [-((2 / 9) ** 0.5), -((2 / 3) ** 0.5), 1 / 3],
]


system = NodeSystem()
# TODO Move the initialization of the state of this component to the init method of the component.
node_rb = NodeSCRigidBodyRotationDynamics(x0_rb, config=args.node_config, use_finite_difference=True)
node_imu1 = NodeIMUSimple(config=args.node_config, name="node_imu1")
node_imu1.frequency = 100
node_sru1 = NodeStellarReferenceUnitSimple(config=args.node_config, name="node_sru1")
node_sru1.frequency = 100
node_imu2 = NodeIMUSimple(config=args.node_config, name="node_imu2")
node_imu2.frequency = 100
node_sru2 = NodeStellarReferenceUnitSimple(config=args.node_config, name="node_sru2")
node_sru2.frequency = 100
node_encoder1 = NodeWheelEncoder(config=args.node_config, name="encoder_1")
node_encoder2 = NodeWheelEncoder(config=args.node_config, name="encoder_2")
node_encoder3 = NodeWheelEncoder(config=args.node_config, name="encoder_3")
node_encoder4 = NodeWheelEncoder(config=args.node_config, name="encoder_4")
node_encoder5 = NodeWheelEncoder(config=args.node_config, name="encoder_5")
node_encoder6 = NodeWheelEncoder(config=args.node_config, name="encoder_6")
node_encoder7 = NodeWheelEncoder(config=args.node_config, name="encoder_7")
node_encoder8 = NodeWheelEncoder(config=args.node_config, name="encoder_8")
node_rwa_1 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[0], config=args.node_config, name="rwa_1"
)
node_rwa_1.frequency = 100.0
node_rwa_2 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[1], config=args.node_config, name="rwa_2"
)
node_rwa_2.frequency = 100.0
node_rwa_3 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[2], config=args.node_config, name="rwa_3"
)
node_rwa_3.frequency = 100.0
node_rwa_4 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[3], config=args.node_config, name="rwa_4"
)
node_rwa_4.frequency = 100.0
node_rwa_5 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[4], config=args.node_config, name="rwa_5"
)
node_rwa_5.frequency = 100.0
node_rwa_6 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[5], config=args.node_config, name="rwa_6"
)
node_rwa_6.frequency = 100.0
node_rwa_7 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[6], config=args.node_config, name="rwa_7"
)
node_rwa_7.frequency = 100.0
node_rwa_8 = NodeRWASimple(
    np.zeros((1,)), body_axis=rw_body_axis[7], config=args.node_config, name="rwa_8"
)
node_rwa_8.frequency = 100.0
node_int_ang_momentum_muxer = InternalAngularMomentumMuxerNode(
    inertia1=0.12e-1,
    inertia2=0.12e-1,
    inertia3=0.12e-1,
    inertia4=0.12e-1,
    inertia5=0.12e-1,
    inertia6=0.12e-1,
    inertia7=0.12e-1,
    inertia8=0.12e-1,
    axis1_vector=rw_body_axis[0],
    axis2_vector=rw_body_axis[1],
    axis3_vector=rw_body_axis[2],
    axis4_vector=rw_body_axis[3],
    axis5_vector=rw_body_axis[4],
    axis6_vector=rw_body_axis[5],
    axis7_vector=rw_body_axis[6],
    axis8_vector=rw_body_axis[7],
    config=args.node_config, name="int_ang_momentum_muxer"
)
node_control = NodePointingControlSimple(config=args.node_config)
node_control.frequency = 100.0
node_estimator = NodeKalmanEstimator(
    x0=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    config=args.node_config,
    name="estimator",
)
node_estimator.frequency = 100.0
node_rate_cmd = NodeConstant(np.array([0.0, 0.0, 0.0]), config=args.node_config)
node_quat_cmd = NodeConstant(np.array([1.0, 0.0, 0.0, 0.0]), config=args.node_config)
node_health1 = NodeConstant(True, config=args.node_config, name="health1")
node_health2 = NodeConstant(True, config=args.node_config, name="health2")
node_health3 = NodeConstant(True, config=args.node_config, name="health3")
node_health4 = NodeConstant(True, config=args.node_config, name="health4")
node_health5 = NodeConstant(True, config=args.node_config, name="health5")
node_health6 = NodeConstant(True, config=args.node_config, name="health6")
node_health7 = NodeConstant(True, config=args.node_config, name="health7")
node_health8 = NodeConstant(True, config=args.node_config, name="health8")
node_viz_true_rate = NodeScope(config=args.node_config, name="viz_true_rate")
node_viz_true_rate.frequency = 100
node_viz_torque = NodeScope(config=args.node_config, name="viz_torque")
node_viz_torque.frequency = 100
node_viz_rate = NodeScope(config=args.node_config, name="viz_imu_rate")
node_viz_rate.frequency = 100
node_viz_pointing = NodeScope(config=args.node_config, name="viz_pointing")
node_viz_pointing.frequency = 100
node_viz_est_rate = NodeScope(config=args.node_config, name="viz_est_rate")
node_viz_est_rate.frequency = 100
node_viz_est_pointing = NodeScope(config=args.node_config, name="viz_est_pointing")
node_viz_est_pointing.frequency = 100
node_monsid_diagnoser = NodeMONSIDDiagnoser(config=args.node_config, name="monsid_logger")
node_torque_adder = AdderNode(8, name="torque_adder")
# Add a fault printer node used for displaying detected faults
node_fault_printer = NodeFaultPrinter(config=args.node_config, name="fault_printer")
node_fault_printer.frequency = 100
node_rw_mixer = ReactionWheelMixerNode(

    axis1=np.array(rw_body_axis[0]),
    axis2=np.array(rw_body_axis[1]),
    axis3=np.array(rw_body_axis[2]),
    axis4=np.array(rw_body_axis[3]),
    axis5=np.array(rw_body_axis[4]),
    axis6=np.array(rw_body_axis[5]),
    axis7=np.array(rw_body_axis[6]),
    axis8=np.array(rw_body_axis[7]),
    config=args.node_config,
    name="rw_mixer",
)
node_internal_mtm_adder = AdderNode(8, name="internal_mtm_adder")
node_concate_monsid_diagnosis = ConcatNode(8, name="concat_monsid_diagnosis")
node_viz_monsid_diagnosis = NodeScope(
    config=args.node_config, name="viz_monsid_diagnosis"
)
# New: visualize individual RW torque commands
node_concat_rw_torques = ConcatNode(8, name="concat_rw_torques")
node_viz_rw_torque_cmds = NodeScope(config=args.node_config, name="viz_rw_torque_cmds")
node_viz_rw_torque_cmds.frequency = 100
# Faults Declaration
imu_zero = ZeroFault("imu_zero", trigger_time=2.0)
imu_zero.active = False
sru_zero = ZeroFault("sru_zero", trigger_time=2.1)
sru_zero.active = False
# inertia_fault = DiagonalInertiaPerturbFault(
#     "inertia_fault",
# )
# inertia_fault.active = False

system.add_node(node_rb)
system.add_node(node_imu1)
system.add_node(node_sru1)
system.add_node(node_imu2)
system.add_node(node_sru2)
system.add_node(node_rwa_1)
system.add_node(node_rwa_2)
system.add_node(node_rwa_3)
system.add_node(node_rwa_4)
system.add_node(node_rwa_5)
system.add_node(node_rwa_6)
system.add_node(node_rwa_7)
system.add_node(node_rwa_8)
system.add_node(node_encoder1)
system.add_node(node_encoder2)
system.add_node(node_encoder3)
system.add_node(node_encoder4)
system.add_node(node_encoder5)
system.add_node(node_encoder6)
system.add_node(node_encoder7)
system.add_node(node_encoder8)
system.add_node(node_rw_mixer)
system.add_node(node_torque_adder)
system.add_node(node_internal_mtm_adder)
system.add_node(node_int_ang_momentum_muxer)
system.add_node(node_control)
system.add_node(node_estimator)
system.add_node(node_rate_cmd)
system.add_node(node_quat_cmd)
system.add_node(node_viz_true_rate)
system.add_node(node_viz_torque)
system.add_node(node_viz_rate)
system.add_node(node_viz_pointing)
system.add_node(node_viz_est_rate)
system.add_node(node_viz_est_pointing)
system.add_node(node_monsid_diagnoser)
system.add_node(node_health1)
system.add_node(node_health2)
system.add_node(node_health3)
system.add_node(node_health4)
system.add_node(node_health5)
system.add_node(node_health6)
system.add_node(node_health7)
system.add_node(node_health8)
system.add_node(node_concate_monsid_diagnosis)
system.add_node(node_viz_monsid_diagnosis)
system.add_node(node_fault_printer)
# Register new nodes
system.add_node(node_concat_rw_torques)
system.add_node(node_viz_rw_torque_cmds)

system.add_faults([imu_zero, sru_zero])

# Node connections
node_rb.o.output_w_sc >> node_imu1.i.input_true_angular_rate
node_rb.o.output_w_sc >> node_imu2.i.input_true_angular_rate

node_rb.o.output_w_sc >> node_viz_true_rate.i.scope

node_rb.o.output_q_sc_to_eci >> node_sru1.i.input_q_sc2eci
node_rb.o.output_q_sc_to_eci >> node_sru2.i.input_q_sc2eci
node_rb.o.output_q_sc_to_eci >> node_viz_pointing.i.scope

node_estimator.o.est_w >> node_control.i.input_w
node_estimator.o.est_q >> node_control.i.input_q
# node_sru1.o.output_q_sc2eci_measure >> node_control.i.input_q
# node_imu1.o.output_measure_angular_rate >> node_control.i.input_w

node_imu1.o.output_measure_angular_rate >> node_viz_rate.i.scope
node_estimator.o.est_w >> node_viz_est_rate.i.scope
node_estimator.o.est_q >> node_viz_est_pointing.i.scope

node_torque_adder.o.sum >> node_viz_torque.i.scope

node_control.o.output_tau_cmd >> node_rw_mixer.i.commanded_torque

# TODO Temporary until I can implement the MONSID health monitor
node_health1.o.constant_out >> node_rw_mixer.i.axis1_health
node_health2.o.constant_out >> node_rw_mixer.i.axis2_health
node_health3.o.constant_out >> node_rw_mixer.i.axis3_health
node_health4.o.constant_out >> node_rw_mixer.i.axis4_health
node_health5.o.constant_out >> node_rw_mixer.i.axis5_health
node_health6.o.constant_out >> node_rw_mixer.i.axis6_health
node_health7.o.constant_out >> node_rw_mixer.i.axis7_health
node_health8.o.constant_out >> node_rw_mixer.i.axis8_health

node_rw_mixer.o.wheel1_torque >> node_rwa_1.i.tau_cmd
node_rw_mixer.o.wheel2_torque >> node_rwa_2.i.tau_cmd
node_rw_mixer.o.wheel3_torque >> node_rwa_3.i.tau_cmd
node_rw_mixer.o.wheel4_torque >> node_rwa_4.i.tau_cmd
node_rw_mixer.o.wheel5_torque >> node_rwa_5.i.tau_cmd
node_rw_mixer.o.wheel6_torque >> node_rwa_6.i.tau_cmd
node_rw_mixer.o.wheel7_torque >> node_rwa_7.i.tau_cmd
node_rw_mixer.o.wheel8_torque >> node_rwa_8.i.tau_cmd

node_rw_mixer.o.wheel1_torque >> node_torque_adder.i.input_0
node_rw_mixer.o.wheel2_torque >> node_torque_adder.i.input_1
node_rw_mixer.o.wheel3_torque >> node_torque_adder.i.input_2
node_rw_mixer.o.wheel4_torque >> node_torque_adder.i.input_3
node_rw_mixer.o.wheel5_torque >> node_torque_adder.i.input_4
node_rw_mixer.o.wheel6_torque >> node_torque_adder.i.input_5
node_rw_mixer.o.wheel7_torque >> node_torque_adder.i.input_6
node_rw_mixer.o.wheel8_torque >> node_torque_adder.i.input_7

# Connect each wheel torque to the concat node for visualization
node_rw_mixer.o.wheel1_torque >> node_concat_rw_torques.i.input_0
node_rw_mixer.o.wheel2_torque >> node_concat_rw_torques.i.input_1
node_rw_mixer.o.wheel3_torque >> node_concat_rw_torques.i.input_2
node_rw_mixer.o.wheel4_torque >> node_concat_rw_torques.i.input_3
node_rw_mixer.o.wheel5_torque >> node_concat_rw_torques.i.input_4
node_rw_mixer.o.wheel6_torque >> node_concat_rw_torques.i.input_5
node_rw_mixer.o.wheel7_torque >> node_concat_rw_torques.i.input_6
node_rw_mixer.o.wheel8_torque >> node_concat_rw_torques.i.input_7

# Feed concatenated torques to the viz scope
node_concat_rw_torques.o.concat >> node_viz_rw_torque_cmds.i.scope

node_rwa_1.o.rw_mtm >> node_internal_mtm_adder.i.input_0
node_rwa_2.o.rw_mtm >> node_internal_mtm_adder.i.input_1
node_rwa_3.o.rw_mtm >> node_internal_mtm_adder.i.input_2
node_rwa_4.o.rw_mtm >> node_internal_mtm_adder.i.input_3
node_rwa_5.o.rw_mtm >> node_internal_mtm_adder.i.input_4
node_rwa_6.o.rw_mtm >> node_internal_mtm_adder.i.input_5
node_rwa_7.o.rw_mtm >> node_internal_mtm_adder.i.input_6
node_rwa_8.o.rw_mtm >> node_internal_mtm_adder.i.input_7

node_encoder1.o.enc_out >> node_int_ang_momentum_muxer.i.wheel1_speed
node_encoder2.o.enc_out >> node_int_ang_momentum_muxer.i.wheel2_speed
node_encoder3.o.enc_out >> node_int_ang_momentum_muxer.i.wheel3_speed
node_encoder4.o.enc_out >> node_int_ang_momentum_muxer.i.wheel4_speed
node_encoder5.o.enc_out >> node_int_ang_momentum_muxer.i.wheel5_speed
node_encoder6.o.enc_out >> node_int_ang_momentum_muxer.i.wheel6_speed
node_encoder7.o.enc_out >> node_int_ang_momentum_muxer.i.wheel7_speed
node_encoder8.o.enc_out >> node_int_ang_momentum_muxer.i.wheel8_speed

node_int_ang_momentum_muxer.o.angular_momentum >> node_control.i.input_mtm_int

node_rwa_1.o.rw_speed >> node_encoder1.i.enc_in
node_rwa_2.o.rw_speed >> node_encoder2.i.enc_in
node_rwa_3.o.rw_speed >> node_encoder3.i.enc_in
node_rwa_4.o.rw_speed >> node_encoder4.i.enc_in
node_rwa_5.o.rw_speed >> node_encoder5.i.enc_in
node_rwa_6.o.rw_speed >> node_encoder6.i.enc_in
node_rwa_7.o.rw_speed >> node_encoder7.i.enc_in
node_rwa_8.o.rw_speed >> node_encoder8.i.enc_in

node_rate_cmd.o.constant_out >> node_control.i.input_w_cmd
node_quat_cmd.o.constant_out >> node_control.i.input_q_cmd

node_torque_adder.o.sum >> node_rb.i.input_tau_external_sc
node_internal_mtm_adder.o.sum >> node_rb.i.input_mtm_internal_sc

# node_monsid_logger.i.cmd_torque << node_control.o.output_tau_cmd
# node_monsid_logger.i.sens_rate << node_rb.o.output_w_sc

# node_monsid_logger.i.sens_imu_rate << node_imu1.o.output_measure_angular_rate
# node_monsid_logger.i.sens_q_sc_to_eci << node_sru1.o.output_q_sc2eci_measure

node_estimator.i.imu1_rate << node_imu1.o.output_measure_angular_rate
node_estimator.i.sru1_q << node_sru1.o.output_q_sc2eci_measure
node_estimator.i.imu2_rate << node_imu2.o.output_measure_angular_rate
node_estimator.i.sru2_q << node_sru2.o.output_q_sc2eci_measure
node_estimator.i.torque_cmd << node_control.o.output_tau_cmd

node_monsid_diagnoser.i.enc1 << node_encoder1.o.enc_out
node_monsid_diagnoser.i.enc2 << node_encoder2.o.enc_out
node_monsid_diagnoser.i.enc3 << node_encoder3.o.enc_out
node_monsid_diagnoser.i.enc4 << node_encoder4.o.enc_out
node_monsid_diagnoser.i.enc5 << node_encoder5.o.enc_out
node_monsid_diagnoser.i.enc6 << node_encoder6.o.enc_out
node_monsid_diagnoser.i.enc7 << node_encoder7.o.enc_out
node_monsid_diagnoser.i.enc8 << node_encoder8.o.enc_out
node_monsid_diagnoser.i.imu1 << node_imu1.o.output_measure_angular_rate
node_monsid_diagnoser.i.imu2 << node_imu2.o.output_measure_angular_rate
node_monsid_diagnoser.i.sru1 << node_sru1.o.output_q_sc2eci_measure
node_monsid_diagnoser.i.sru2 << node_sru2.o.output_q_sc2eci_measure
node_monsid_diagnoser.i.rw1_cmd << node_rw_mixer.o.wheel1_torque
node_monsid_diagnoser.i.rw2_cmd << node_rw_mixer.o.wheel2_torque
node_monsid_diagnoser.i.rw3_cmd << node_rw_mixer.o.wheel3_torque
node_monsid_diagnoser.i.rw4_cmd << node_rw_mixer.o.wheel4_torque
node_monsid_diagnoser.i.rw5_cmd << node_rw_mixer.o.wheel5_torque
node_monsid_diagnoser.i.rw6_cmd << node_rw_mixer.o.wheel6_torque
node_monsid_diagnoser.i.rw7_cmd << node_rw_mixer.o.wheel7_torque
node_monsid_diagnoser.i.rw8_cmd << node_rw_mixer.o.wheel8_torque
node_monsid_diagnoser.i.dynamics_rate << node_estimator.o.est_w
node_monsid_diagnoser.i.dynamics_orientation << node_estimator.o.est_q
node_monsid_diagnoser.i.rw1_momentum << node_rwa_1.o.rw_mtm
node_monsid_diagnoser.i.rw2_momentum << node_rwa_2.o.rw_mtm
node_monsid_diagnoser.i.rw3_momentum << node_rwa_3.o.rw_mtm
node_monsid_diagnoser.i.rw4_momentum << node_rwa_4.o.rw_mtm
node_monsid_diagnoser.i.rw5_momentum << node_rwa_5.o.rw_mtm
node_monsid_diagnoser.i.rw6_momentum << node_rwa_6.o.rw_mtm
node_monsid_diagnoser.i.rw7_momentum << node_rwa_7.o.rw_mtm
node_monsid_diagnoser.i.rw8_momentum << node_rwa_8.o.rw_mtm

node_monsid_diagnoser.o.rw1_health >> node_concate_monsid_diagnosis.i.input_0
node_monsid_diagnoser.o.rw2_health >> node_concate_monsid_diagnosis.i.input_1
node_monsid_diagnoser.o.rw3_health >> node_concate_monsid_diagnosis.i.input_2
node_monsid_diagnoser.o.rw4_health >> node_concate_monsid_diagnosis.i.input_3
node_monsid_diagnoser.o.rw5_health >> node_concate_monsid_diagnosis.i.input_4
node_monsid_diagnoser.o.rw6_health >> node_concate_monsid_diagnosis.i.input_5
node_monsid_diagnoser.o.rw7_health >> node_concate_monsid_diagnosis.i.input_6
node_monsid_diagnoser.o.rw8_health >> node_concate_monsid_diagnosis.i.input_7

node_concate_monsid_diagnosis.o.concat >> node_viz_monsid_diagnosis.i.scope

node_fault_printer.i.fault_detected << node_monsid_diagnoser.o.fault_detected

# Port fault registration
node_imu1.o.output_measure_angular_rate.add_fault(imu_zero)
node_sru1.o.output_q_sc2eci_measure.add_fault(sru_zero)

# Param fault registration
# node_rb.p.inertia_moment.add_fault(inertia_fault)

print(system)

system.simulate(
    args.sim_duration,
    args.dt,
    save_dir=None,
    sim_name=None,
)

def main():
    pass