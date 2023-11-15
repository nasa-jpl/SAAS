#! python
import argparse

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation
import toml

from syssim import NodeSystem
from syssim.nodes.source import NodeConstant

from saas.dynamics import (
    NodeSCOrbitalDynamics,
    NodeSCRigidBodyRotationDynamics,
)
from saas.viz import NodeSCAttitudeScope, NodeSCTrajScope, NodeVizAttitudeError
from saas.controller import NodePointingControlSimple
from saas.actuator import NodeRWASimple
from saas.fs import (
    NodePointingNadir,
    NodePointingCom,
    NodePointingSunStuck,
    NodePointingCooling,
)
from saas.utility import NodeQuatRotation
from saas.sensor import NodeIMUSimple, NodeStellarReferenceUnitSimple
from saas.metrics import (
    NodeMetricPointingError,
    NodeMetricSoCTemperature,
    NodeMetricScience,
)
from saas.environment import (
    NodeEarthPosition,
    NodeSunPosition,
    NodeEarthOcclusion,
    NodeSunOcclusion,
)
from saas.power import NodeSimpleBattery
from saas.structure import NodeSimpleRectanglePrismSCBus
from saas.power import NodeSimpleGimbalSolarPanel
from saas.fs import NodeSimpleFlightSoftware, NodeFaultProtectionMGS
from saas.utility import get_poliastro_body, NodeRandomTimeDate


def circular_orbit_x0(r_low, r_high, mu):
    r = default_rng().uniform(r_low, r_high)
    theta = default_rng().uniform(0, np.pi)
    phi = default_rng().uniform(-np.pi, np.pi)

    r_eci = np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )

    v = np.sqrt(mu / r)

    e_theta = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])

    n_theta = default_rng().uniform(0, 1)
    n_phi = np.sqrt(1 - n_theta**2)

    v_eci = v * (n_theta * e_theta + n_phi * e_phi)

    return np.concatenate([r_eci, v_eci])


def circular_orbit_yz_plane_x0(r_low, r_high, mu):
    r = default_rng().uniform(r_low, r_high)
    theta = 0
    phi = np.radians(90)

    r_eci = np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )

    v = np.sqrt(mu / r)

    e_theta = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])

    # n_theta = 0
    # n_phi = 1

    v_eci = v * e_theta

    return np.concatenate([r_eci, v_eci])


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
    "-fc",
    "--fault-config",
    type=str,
    help="Path to the fault configuration specification",
    default=None,
    dest="fault_config",
)
parser.add_argument(
    "-d",
    "--sim-duration",
    type=float,
    help="Simulation duration in seconds",
    default=7200,
    dest="sim_duration",
)
parser.add_argument("--dt", help="Default simulation timestep", type=float, default=1.0)
parser.add_argument(
    "-b", "--batches", type=int, help="Number of simulation batches to run", default=1
)
parser.add_argument(
    "-n",
    "--sim-name",
    type=str,
    help="Name of this simulation for record keeping",
    default=None,
    dest="sim_name",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    help="Directory to save the ouputs of this simulation",
    default=None,
    dest="save_dir",
)

args = parser.parse_args()

config = toml.load(args.node_config)

central_body = get_poliastro_body(
    config["NodeSCOrbitalDynamics-1"]["central_body_name"]
)

x0_orbit = circular_orbit_x0(
    central_body.R_mean.value + 370e3,
    central_body.R_mean.value + 430e3,
    central_body.k.value,
)


x0_rb = rigid_body_x0(0.001, 0.005)

system = NodeSystem()
node_orbit = NodeSCOrbitalDynamics(x0_orbit, config=args.node_config)
node_rb = NodeSCRigidBodyRotationDynamics(x0_rb, config=args.node_config)
node_orbit_scope = NodeSCTrajScope(config=args.node_config)
node_orbit_scope.period = 30.0
node_att_scope = NodeSCAttitudeScope(config=args.node_config)
node_att_scope.period = 30.0
node_imu = NodeIMUSimple(config=args.node_config)
node_imu.frequency = 1.0
node_sru = NodeStellarReferenceUnitSimple(config=args.node_config)
node_sru.frequency = 1.0
node_rwa = NodeRWASimple(np.zeros((3,)), config=args.node_config, name="rwa")
node_rwa.frequency = 60.0
node_control = NodePointingControlSimple(np.zeros((3,)), config=args.node_config)
node_control.frequency = 1.0
node_nadir_pointing = NodePointingNadir(config=args.node_config)
node_com_pointing = NodePointingCom(config=args.node_config)
node_com_pointing.period = 60
node_ss_pointing = NodePointingSunStuck(config=args.node_config)
node_ss_pointing.period = 60
node_cool_pointing = NodePointingCooling(config=args.node_config)
node_cool_pointing.period = 60
node_metric_pointing = NodeMetricPointingError(config=args.node_config)
node_x_unit = NodeConstant(np.array([1, 0, 0]))
node_boresight = NodeQuatRotation()
node_sun = NodeSunPosition(config=args.node_config)
node_sun.period = 30 * 60.0
node_earth = NodeEarthPosition(config=args.node_config)
node_earth.period = 30 * 60.0
node_occlusion = NodeSunOcclusion(config=args.node_config)
node_occlusion.period = 60.0
node_earth_occlusion = NodeEarthOcclusion(config=args.node_config)
node_earth_occlusion.period = 60.0
node_batt = NodeSimpleBattery(np.array([0.50]), config=args.node_config)
node_batt_draw = NodeConstant(5.0)
node_thermal = NodeSimpleRectanglePrismSCBus(np.array([275]), config=args.node_config)
node_solar_panel = NodeSimpleGimbalSolarPanel(config=args.node_config)
node_solar_panel.frequency = 10.0
node_metric_soc_temp = NodeMetricSoCTemperature(config=args.node_config)
node_viz_att_cntrl_error = NodeVizAttitudeError(config=args.node_config)
node_fs = NodeSimpleFlightSoftware(config=args.node_config, name="fs")
node_metric_science = NodeMetricScience(config=args.node_config)
node_fault_mgs = NodeFaultProtectionMGS(config=args.node_config)
node_fault_mgs.frequency = 10
node_start_datetime = NodeRandomTimeDate(config=args.node_config)

system.add_node(node_orbit)
system.add_node(node_rb)
system.add_node(node_imu)
system.add_node(node_sru)
system.add_node(node_rwa)
system.add_node(node_control)
system.add_node(node_nadir_pointing)
system.add_node(node_com_pointing)
system.add_node(node_ss_pointing)
system.add_node(node_cool_pointing)
system.add_node(node_x_unit)
system.add_node(node_boresight)
system.add_node(node_sun)
system.add_node(node_earth)
system.add_node(node_occlusion)
system.add_node(node_earth_occlusion)
system.add_node(node_batt)
system.add_node(node_batt_draw)
system.add_node(node_thermal)
system.add_node(node_solar_panel)
system.add_node(node_fs)
system.add_node(node_fault_mgs)
system.add_node(node_start_datetime)

system.add_node(node_orbit_scope)
system.add_node(node_att_scope)
system.add_node(node_metric_pointing)
system.add_node(node_metric_soc_temp)
system.add_node(node_viz_att_cntrl_error)
system.add_node(node_metric_science)

# Other nodes
"""
x Thermal model
x Solar panel model
x Battery model + constant rate of drain
x Earth direction
x Sun direction
x Occlusion
- Output metrics + plotting faults
- Fault definitions
- Flight software
- auto FP node
"""


node_orbit["output_a_ff_eci"] = node_imu["true_a"]
node_rb["output_w_sc"] = node_imu["true_w"]
node_rb["output_q_sc_to_eci"] = node_imu["q_sc2eci"]

node_rb["output_q_sc_to_eci"] = node_sru["input_q_sc2eci"]

# pointing solutions
node_orbit["output_r_eci"] = node_nadir_pointing["input_r_eci"]
node_orbit["output_v_eci"] = node_nadir_pointing["input_v_eci"]
# node_fs['output_fs_state'] = node_nadir_pointing["input_fs_state"]

node_sun["out_sun_unit_icrs"] = node_com_pointing["in_sun_n"]
node_earth["out_earth_unit_icrs"] = node_com_pointing["in_earth_n"]
# node_fs['output_fs_state'] = node_com_pointing["input_fs_state"]

node_sun["out_sun_unit_icrs"] = node_ss_pointing["in_sun_n"]
node_earth["out_earth_unit_icrs"] = node_ss_pointing["in_earth_n"]
# node_fs['output_fs_state'] = node_ss_pointing["input_fs_state"]

node_sun["out_sun_unit_icrs"] = node_cool_pointing["in_sun_n"]
# node_fs['output_fs_state'] = node_cool_pointing["input_fs_state"]


node_fs["out_q_cmd"] = node_control["input_q_cmd"]
node_sru["output_q_sc2eci_measure"] = node_control["input_q"]
node_fs["out_w_cmd"] = node_control["input_w_cmd"]

node_imu["measure_w"] = node_control["input_w"]
node_rb["output_w_sc"] = node_control["input_w"]
node_rwa["rwa_mtm"] = node_control["input_mtm_int"]

node_control["output_tau_cmd"] = node_rwa["tau_cmd"]

node_rwa["rwa_mtm"] = node_rb["input_mtm_internal_sc"]
node_rwa["rwa_tau"] = node_rb["input_tau_external_sc"]

node_x_unit["output"] = node_boresight["in_vec"]
node_sru["output_q_sc2eci_measure"] = node_boresight["in_quat"]

node_orbit["output_r_eci"] = node_earth["in_sc_pos_icrs"]
node_orbit["output_r_eci"] = node_sun["in_sc_pos_icrs"]
node_orbit["output_r_eci"] = node_occlusion["in_sc_pos_icrs"]
node_orbit["output_r_eci"] = node_earth_occlusion["in_sc_pos_icrs"]

node_start_datetime["out_datetime"] = node_earth["in_start_datetime"]
node_start_datetime["out_datetime"] = node_sun["in_start_datetime"]
node_start_datetime["out_datetime"] = node_occlusion["in_start_datetime"]
node_start_datetime["out_datetime"] = node_earth_occlusion["in_start_datetime"]

# solar panel
node_sun["out_solar_constant"] = node_solar_panel["in_solar_constant"]
node_sun["out_sun_unit_icrs"] = node_solar_panel["in_sun_unit_sc"]
node_fs["out_solar_gimbal"] = node_solar_panel["in_sp_gimbal_angle"]
node_occlusion["out_is_occluded"] = node_solar_panel["in_is_occluded"]
node_rb["output_q_sc_to_eci"] = node_solar_panel["input_q_sc2eci"]

# thermals
node_sun["out_solar_constant"] = node_thermal["in_solar_constant"]
node_sun["out_sun_unit_icrs"] = node_thermal["in_sun_unit"]
node_rb["output_q_sc_to_eci"] = node_thermal["in_q_sc2eci"]

# battery
node_solar_panel["out_power"] = node_batt["in_current"]
node_batt_draw["output"] = node_batt["in_current_draw"]

# flight software
node_control["output_q_err"] = node_fs["in_point_err"]
node_nadir_pointing["output_q_cmd"] = node_fs["in_q_cmd_nadir"]
node_ss_pointing["output_q_cmd"] = node_fs["in_q_cmd_ss"]
node_com_pointing["output_q_cmd"] = node_fs["in_q_cmd_com"]
node_cool_pointing["output_q_cmd"] = node_fs["in_q_cmd_cool"]
node_nadir_pointing["output_w_cmd"] = node_fs["in_w_cmd_nadir"]
node_ss_pointing["output_w_cmd"] = node_fs["in_w_cmd_ss"]
node_com_pointing["output_w_cmd"] = node_fs["in_w_cmd_com"]
node_cool_pointing["output_w_cmd"] = node_fs["in_w_cmd_cool"]
node_nadir_pointing["output_gimbal_cmd"] = node_fs["in_gimbal_cmd_nadir"]
node_ss_pointing["output_sp_gimbal"] = node_fs["in_gimbal_cmd_ss"]
node_com_pointing["output_sp_gimbal"] = node_fs["in_gimbal_cmd_com"]
node_cool_pointing["output_sp_gimbal"] = node_fs["in_gimbal_cmd_cool"]
node_thermal["out_sc_temp"] = node_fs["in_sc_temp"]
node_batt["out_soc"] = node_fs["in_sc_soc"]
node_earth_occlusion["out_is_occluded"] = node_fs["in_is_occluded_earth"]
node_occlusion["out_is_occluded"] = node_fs["in_is_occluded_sun"]

node_fs["out_solar_gimbal"] = node_fault_mgs["in_gimbal_cmd"]
node_solar_panel["out_gimbal_measure"] = node_fault_mgs["in_gimbal_measure"]

node_fault_mgs["out_gimbal_fault"] = node_fs["in_is_fault"]

# metrics and viz...
node_orbit["output_r_eci"] = node_orbit_scope["input_r_eci"]
node_orbit["output_v_eci"] = node_orbit_scope["input_v_eci"]
node_sun["out_sun_pos_icrs"] = node_orbit_scope["input_sc_sun_icrs"]
node_earth["out_earth_pos_icrs"] = node_orbit_scope["input_sc_earth_icrs"]
node_earth_occlusion["out_is_occluded"] = node_orbit_scope["input_earth_occ"]
node_occlusion["out_is_occluded"] = node_orbit_scope["input_sun_occ"]
node_start_datetime["out_datetime"] = node_orbit_scope["in_start_datetime"]

node_rb["output_q_sc_to_eci"] = node_att_scope["input_q_sc_to_eci"]
node_rb["output_w_sc"] = node_att_scope["input_w_sc"]
node_nadir_pointing["output_nadir"] = node_att_scope["input_nadir"]
node_solar_panel["out_gimbal"] = node_att_scope["input_gimbal"]
node_sun["out_sun_unit_icrs"] = node_att_scope["input_sun_unit"]
node_earth["out_earth_unit_icrs"] = node_att_scope["input_earth_unit"]
node_fs["out_fs_state"] = node_att_scope["input_fs_state"]
node_earth_occlusion["out_is_occluded"] = node_att_scope["input_earth_occ"]
node_occlusion["out_is_occluded"] = node_att_scope["input_sun_occ"]
node_start_datetime["out_datetime"] = node_att_scope["in_start_datetime"]

node_nadir_pointing["output_nadir"] = node_metric_pointing["input_cmd_vec"]
node_boresight["out_vec"] = node_metric_pointing["input_meas_vec"]
node_start_datetime["out_datetime"] = node_metric_pointing["in_start_datetime"]

node_batt["out_soc"] = node_metric_soc_temp["in_soc"]
node_thermal["out_sc_temp"] = node_metric_soc_temp["in_temp"]
node_start_datetime["out_datetime"] = node_metric_soc_temp["in_start_datetime"]

node_control["output_q_err"] = node_viz_att_cntrl_error["in_q_e"]
node_control["output_w_err"] = node_viz_att_cntrl_error["in_w_e"]

node_fs["out_science"] = node_metric_science["in_science"]
node_fs["out_science_downlinked"] = node_metric_science["in_science_dl"]
node_start_datetime["out_datetime"] = node_metric_science["in_start_datetime"]

if args.fault_config is not None:
    system.add_faults(args.fault_config)

# cProfile.run("system.simulate(3600 * 0.2, 1)")
system.simulate(
    args.sim_duration,
    args.dt,
    save_dir=args.save_dir,
    sim_name=args.sim_name,
    batches=args.batches,
)
