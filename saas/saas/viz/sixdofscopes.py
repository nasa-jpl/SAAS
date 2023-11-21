from typing import Sequence

import matplotlib.pyplot as plt

import matplotlib.animation as anim
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from syssim import Node, InputPort

from saas.utility.body import get_poliastro_body


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

    def set_positions_3d(
        self, posA: Sequence[float], posB: Sequence[Sequence], posC: Sequence[float]
    ):
        self._verts3d = posA, posB, posC


class NodeSCTrajScope(Node):
    def __init__(self, **kwargs):
        input_r_eci = InputPort("input_r_eci", self)
        input_v_eci = InputPort("input_v_eci", self)
        input_sc_sun_icrs = InputPort("input_sc_sun_icrs", self)
        input_sc_earth_icrs = InputPort("input_sc_earth_icrs", self)
        input_earth_occ = InputPort("input_earth_occ", self)
        input_sun_occ = InputPort("input_sun_occ", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        ports = {
            input_r_eci.name: input_r_eci,
            input_v_eci.name: input_v_eci,
            input_sc_sun_icrs.name: input_sc_sun_icrs,
            input_sc_earth_icrs.name: input_sc_earth_icrs,
            input_earth_occ.name: input_earth_occ,
            input_sun_occ.name: input_sun_occ,
            in_start_datetime.name: in_start_datetime,
        }

        super().__init__(ports, **kwargs)

    def initialize(self):
        self._t = list()
        self._r = list()
        self._v = list()
        self._e = list()
        self._s = list()
        self._occ = list()
        self._central_body = get_poliastro_body(self._config["central_body"])

    def update(self, sim_time: float):
        r = self._ports["input_r_eci"].read()
        v = self._ports["input_v_eci"].read()

        self._t.append(sim_time)
        self._r.append(r)
        self._v.append(v)

        e = self._ports["input_sc_earth_icrs"].read()
        if np.any(e) == None:
            self._e.append(np.zeros((3,)))
        else:
            self._e.append(e)

        s = self._ports["input_sc_sun_icrs"].read()
        if np.any(s) == None:
            self._s.append(np.zeros((3,)))
        else:
            self._s.append(s)

        occ = (
            self._ports["input_sun_occ"].read(),
            self._ports["input_earth_occ"].read(),
        )
        self._occ.append(occ)

    def finalize(self):
        t0 = self._ports["in_start_datetime"].read()
        if self._config["dark_mode"]:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
            
        r_max = np.max(np.linalg.norm(self._r, axis=1)) * 1e-3
        v_max = np.max(self._v) * 1e-3
        v_min = np.min(self._v) * 1e-3

        self._fig = plt.figure(
            figsize=(
                18,
                10,
            ),
            dpi=96,
        )

        self._ax_pos = plt.subplot(1, 2, 1, projection="3d", proj_type="persp")
        self._ax_pos.set_box_aspect([1.0, 1.0, 1.0])
        self._ax_pos.set_title("S/C Trajectory in ECI")
        self._ax_pos.set_xlabel("x [km]")
        self._ax_pos.set_ylabel("y [km]")
        self._ax_pos.set_zlabel("z [km]")
        self._ax_pos.set_xlim(-r_max * 1.5, r_max * 1.5)
        self._ax_pos.set_ylim(-r_max * 1.5, r_max * 1.5)
        self._ax_pos.set_zlim(-r_max * 1.5, r_max * 1.5)

        self._ax_vel = plt.subplot(1, 2, 2)
        self._ax_vel.set_title("S/C Velocity in ECI")
        self._ax_vel.set_xlabel(f"t + {t0.strftime('%Y-%m-%d %H:%M:%S')} [min]")
        self._ax_vel.set_ylabel("speed [km/s]")
        self._ax_vel.set_xlim(0, self._t[-1] / 60)
        self._ax_vel.set_ylim(1.2 * v_min, 1.2 * v_max)

        self._artist_sc_traj = self._ax_pos.plot(
            [], [], [], label="S/C Trajectory", color="green", linestyle="dashed"
        )[0]

        self._artist_sc = self._ax_pos.scatter(
            [], [], [], label="S/C", color="orange", s=10, marker="*"
        )

        u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        self._artist_cb = self._ax_pos.plot_surface(
            self._central_body.R.value * 1e-3 * x,
            self._central_body.R.value * 1e-3 * y,
            self._central_body.R.value * 1e-3 * z,
            color="red",
            alpha=0.4,
        )

        self._artist_earth = Arrow3D(
            (0, 1), (0, 1), (0, 1), mutation_scale=25, label="earth", color="green"
        )
        # self._ax_pos.plot([], [], [], color="green", label="earth")[0]
        self._artist_sun = Arrow3D(
            (0, 1), (0, 1), (0, 1), mutation_scale=25, label="sun", color="yellow"
        )
        # self._ax_pos.plot([], [], [], color="yellow", label="sun")[0]
        self._ax_pos.add_artist(self._artist_earth)
        self._ax_pos.add_artist(self._artist_sun)

        self._artist_vel = self._ax_vel.plot(
            [], np.array([[], [], []]).T, label=["x", "y", "z"], lw=2.0
        )

        self._ax_pos.legend()
        self._ax_vel.legend()

        def frame_update(frame: int):
            ra = np.array(self._r[0:frame]) / 1e3
            rnorm = np.linalg.norm(self._r[frame]) * 1e-3

            t = self._t[0:frame]
            v = self._v[0:frame]
            e_sc = self._e[frame]
            s_sc = self._s[frame]

            self._artist_sc_traj.set_data_3d(ra[:, 0], ra[:, 1], ra[:, 2])
            self._artist_sc._offsets3d = (ra[-1:, 0], ra[-1:, 1], ra[-1:, 2])

            e_dir_icrs = (
                1.2
                * rnorm
                * (self._r[frame] + e_sc)
                / np.linalg.norm(self._r[frame] + e_sc)
            )
            s_dir_icrs = (
                1.2
                * rnorm
                * (self._r[frame] + s_sc)
                / np.linalg.norm(self._r[frame] + s_sc)
            )

            e_occ = self._occ[frame][1]
            s_occ = self._occ[frame][0]

            self._artist_earth.set_positions_3d(
                (0, e_dir_icrs[0]), (0, e_dir_icrs[1]), (0, e_dir_icrs[2])
            )

            # self._artist_earth.set_data_3d(
            #     [0, e_dir_icrs[0]], [0, e_dir_icrs[1]], [0, e_dir_icrs[2]]
            # )
            if e_occ == True:
                self._artist_earth.set_alpha(0.2)
            else:
                self._artist_earth.set_alpha(1.0)

            self._artist_sun.set_positions_3d(
                (0, s_dir_icrs[0]), (0, s_dir_icrs[1]), (0, s_dir_icrs[2])
            )

            # self._artist_sun.set_data_3d(
            #     [0, s_dir_icrs[0]], [0, s_dir_icrs[1]], [0, s_dir_icrs[2]]
            # )
            if s_occ == True:
                self._artist_sun.set_alpha(0.2)
            else:
                self._artist_sun.set_alpha(1.0)

            self._ax_pos.draw_artist(self._artist_earth)
            self._ax_pos.draw_artist(self._artist_sun)

            self._artist_vel[0].set_data(np.array(t) / 60, np.array(v)[:, 0] / 1e3)
            self._artist_vel[1].set_data(np.array(t) / 60, np.array(v)[:, 1] / 1e3)
            self._artist_vel[2].set_data(np.array(t) / 60, np.array(v)[:, 2] / 1e3)

        dta = self._config["anim_duration"]
        dts = (self._t[-1] - self._t[0]) * 60
        sa = dts / dta
        vs = (self._t[1] - self._t[0]) * 60
        va = np.clip(vs / sa * 1000, 1, None)

        anim_obj = anim.FuncAnimation(
            self._fig, frame_update, np.arange(1, len(self._t)), interval=va
        )

        progress_bar = tqdm(
            total=len(self._t),
            desc=f"Saving anim from {self.name}",
            unit="frame",
            leave=False,
            position=2,
        )
        if self._config["save"] and self._system.get_output_dir() is not None:
            anim_obj.save(
                self._system.get_output_dir() + f"{self.name}.mov".replace(" ", "_"),
                dpi=80,
                bitrate=-1,
                progress_callback=lambda i, n: progress_bar.update(1),
            )
        progress_bar.close()
        if self._config["show"]:
            plt.show()

        plt.close()


class NodeSCAttitudeScope(Node):
    def __init__(self, config: str = None):
        input_q_sc_to_eci = InputPort("input_q_sc_to_eci", self)
        input_w_sc = InputPort("input_w_sc", self)
        input_nadir = InputPort("input_nadir", self)
        input_gimbal = InputPort("input_gimbal", self)
        input_sun_unit = InputPort("input_sun_unit", self)
        input_earth_unit = InputPort("input_earth_unit", self)
        input_fs_state = InputPort("input_fs_state", self)
        input_earth_occ = InputPort("input_earth_occ", self)
        input_sun_occ = InputPort("input_sun_occ", self)
        in_start_datetime = InputPort("in_start_datetime", self)

        ports = {
            input_q_sc_to_eci.name: input_q_sc_to_eci,
            input_w_sc.name: input_w_sc,
            input_nadir.name: input_nadir,
            input_gimbal.name: input_gimbal,
            input_sun_unit.name: input_sun_unit,
            input_earth_unit.name: input_earth_unit,
            input_fs_state.name: input_fs_state,
            input_earth_occ.name: input_earth_occ,
            input_sun_occ.name: input_sun_occ,
            in_start_datetime.name: in_start_datetime,
        }

        super().__init__(ports, config)

    def initialize(self):
        self._vertices = np.array(
            [
                [-2, -1, -0.5],
                [2, -1, -0.5],
                [2, 1, -0.5],
                [-2, 1, -0.5],
                [-2, -1, 0.5],
                [2, -1, 0.5],
                [2, 1, 0.5],
                [-2, 1, 0.5],
            ]
        )

        self._faces = np.array(
            [
                [0, 1, 2, 3],
                [0, 4, 7, 3],
                [4, 5, 6, 7],
                [5, 1, 2, 6],
                [0, 1, 5, 4],
                [3, 2, 6, 7],
            ]
        )

        self._v_sp = np.array(
            [
                [-1, -1, 0],
                [-1, -3, 0],
                [1, -3, 0],
                [1, -1, 0],
                [-1, 1, 0],
                [-1, 3, 0],
                [1, 3, 0],
                [1, 1, 0],
            ]
        )

        self._f_sp = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        )

        self._colors = ["red", "green", "blue", "yellow", "orange", "purple"]
        self._t = list()
        self._w = list()
        self._q = list()
        self._n = list()
        self._s = list()
        self._e = list()
        self._g = list()
        self._occ = list()
        self._fs_state = list()
        self._frame_data = list()

    def update(self, sim_time: float):
        q = self._ports["input_q_sc_to_eci"].read()
        if np.any(q) == None:
            self._q.append(Rotation.identity().as_quat(canonical=True))
        else:
            self._q.append(q)

        w = self._ports["input_w_sc"].read()
        if np.any(w) == None:
            self._w.append(np.zeros((3,)))
        else:
            self._w.append(w)

        n = self._ports["input_nadir"].read()
        if np.any(n) == None:
            self._n.append(np.zeros((3,)))
        else:
            self._n.append(n)

        g = np.copy(self._ports["input_gimbal"].read())
        if g == None:
            self._g.append(np.array([0]))
        else:
            self._g.append(g)

        e = self._ports["input_earth_unit"].read()
        if np.any(e) == None:
            self._e.append(np.zeros((3,)))
        else:
            self._e.append(e)

        s = self._ports["input_sun_unit"].read()
        if np.any(s) == None:
            self._s.append(np.zeros((3,)))
        else:
            self._s.append(s)

        self._fs_state.append(self._ports["input_fs_state"].read())

        occ = (
            self._ports["input_sun_occ"].read(),
            self._ports["input_earth_occ"].read(),
        )
        self._occ.append(occ)

        self._t.append(sim_time / 60)

    def finalize(self):
        t0 = self._ports["in_start_datetime"].read()
        if self._config["dark_mode"]:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        self._frames = list()

        self._fig = plt.figure(
            figsize=(
                18,
                10,
            ),
            dpi=96,
        )

        self._ax_orient = plt.subplot(1, 2, 1, projection="3d")

        self._ax_orient.set_xlim(-4, 4)
        self._ax_orient.set_ylim(-4, 4)
        self._ax_orient.set_zlim(-4, 4)
        self._ax_orient.set_xlabel("X")
        self._ax_orient.set_ylabel("Y")
        self._ax_orient.set_zlabel("Z")
        self._ax_orient.set_title("S/C Attitude")

        self._ax_w = plt.subplot(1, 2, 2)
        self._ax_w.set_title("S/C Angular Velocity")
        self._ax_w.set_xlabel(f"t + {t0.strftime('%Y-%m-%d %H:%M:%S')} [min]")
        self._ax_w.set_ylabel("angular velocity [rad/s]")
        self._ax_w.set_xlim(self._t[0], self._t[-1])
        self._ax_w.set_ylim(
            1.2 * np.array(self._w).min(), 1.2 * np.array(self._w).max()
        )

        # S/C bus
        cube = [list(self._vertices[face][:]) for face in self._faces]
        self._sc_artist = Poly3DCollection(cube, edgecolor="k")
        self._sc_artist.set_facecolor(self._colors)
        self._ax_orient.add_collection(self._sc_artist)

        # Solar panels
        solar_panels = [list(self._v_sp[face][:]) for face in self._f_sp]
        self._sp_artist = Poly3DCollection(solar_panels, edgecolor="k")
        self._sp_artist.set_facecolor("green")
        self._ax_orient.add_collection(self._sp_artist)

        # self._nadir_artist = self._ax_orient.plot(
        #     [], [], color="blue", label="nadir", lw=6
        # )[0]

        # self._sun_artist = self._ax_orient.plot(
        #     [], [], color="yellow", label="sun", lw=6
        # )[0]

        # self._earth_artist = self._ax_orient.plot(
        #     [], [], color="green", label="earth", lw=6
        # )[0]

        self._earth_artist = Arrow3D(
            (0, 1), (0, 1), (0, 1), mutation_scale=25, label="earth", color="green"
        )
        self._sun_artist = Arrow3D(
            (0, 1), (0, 1), (0, 1), mutation_scale=25, label="sun", color="yellow"
        )
        self._nadir_artist = Arrow3D(
            (0, 1), (0, 1), (0, 1), mutation_scale=25, label="nadir", color="blue"
        )
        self._ax_orient.add_artist(self._earth_artist)
        self._ax_orient.add_artist(self._sun_artist)
        self._ax_orient.add_artist(self._nadir_artist)

        # State text
        self._state_text_artist = self._ax_orient.text(0, 0, 3, s=f"", fontsize=10)

        # Angular velocity
        self._w_artist = self._ax_w.plot([], np.array([[], [], []]).T, lw=2.0)
        self._w_artist[0].set_label("w_x")
        self._w_artist[1].set_label("w_y")
        self._w_artist[2].set_label("w_z")

        self._ax_w.legend()
        self._ax_orient.legend()

        def init():
            sc_to_eci = Rotation.from_quat(
                [
                    self._q[0][1],
                    self._q[0][2],
                    self._q[0][3],
                    self._q[0][0],
                ]
            )
            rotated_vertices = sc_to_eci.apply(self._vertices)
            cube = [list(rotated_vertices[face][:]) for face in self._faces]

            self._sc_artist.set_verts(cube)

            r_gimbal = Rotation.from_euler("y", self._g[0])
            sp_to_eci = sc_to_eci * r_gimbal
            rotated_vertices = sp_to_eci.apply(self._v_sp)
            solar = [list(rotated_vertices[face][:]) for face in self._f_sp]

            self._sp_artist.set_verts(solar)

            v = np.array([[0, 0, 0], 3 * self._n[0]])
            self._nadir_artist.set_positions_3d(v[:, 0], v[:, 1], v[:, 2])
            v = np.array([[0, 0, 0], 3 * self._e[0]])
            self._earth_artist.set_positions_3d(v[:, 0], v[:, 1], v[:, 2])
            v = np.array([[0, 0, 0], 3 * self._s[0]])
            self._sun_artist.set_positions_3d(v[:, 0], v[:, 1], v[:, 2])

            self._w_artist[0].set_data([], [])
            self._w_artist[1].set_data([], [])
            self._w_artist[2].set_data([], [])

        def frame_update(frame):
            sc_to_eci = Rotation.from_quat(
                [
                    self._q[frame][1],
                    self._q[frame][2],
                    self._q[frame][3],
                    self._q[frame][0],
                ]
            )
            rotated_vertices = sc_to_eci.apply(self._vertices)
            cube = [list(rotated_vertices[face][:]) for face in self._faces]
            self._sc_artist.set_verts(cube)

            r_gimbal = Rotation.from_euler("y", self._g[frame])
            sp_to_eci = sc_to_eci * r_gimbal
            rotated_vertices = sp_to_eci.apply(self._v_sp)
            solar = [list(rotated_vertices[face][:]) for face in self._f_sp]

            self._sp_artist.set_verts(solar)

            v = np.array([[0, 0, 0], 3 * self._n[frame]])
            self._nadir_artist.set_positions_3d(v[:, 0], v[:, 1], v[:, 2])

            v = np.array([[0, 0, 0], 3 * self._e[frame]])
            self._earth_artist.set_positions_3d(v[:, 0], v[:, 1], v[:, 2])
            if self._occ[frame][1] == True:
                self._earth_artist.set_alpha(0.2)
            else:
                self._earth_artist.set_alpha(1.0)

            v = np.array([[0, 0, 0], 3 * self._s[frame]])
            self._sun_artist.set_positions_3d(v[:, 0], v[:, 1], v[:, 2])
            if self._occ[frame][0] == True:
                self._sun_artist.set_alpha(0.2)
            else:
                self._sun_artist.set_alpha(1.0)

            self._w_artist[0].set_data(self._t[0:frame], np.array(self._w)[0:frame, 0])
            self._w_artist[1].set_data(self._t[0:frame], np.array(self._w)[0:frame, 1])
            self._w_artist[2].set_data(self._t[0:frame], np.array(self._w)[0:frame, 2])

            state = self._fs_state[frame]
            self._state_text_artist.set_text(f"{state}")

        dta = self._config["anim_duration"]
        dts = (self._t[-1] - self._t[0]) * 60
        sa = dts / dta
        vs = (self._t[1] - self._t[0]) * 60
        va = np.clip(vs / sa * 1000, 1, None)

        anim_obj = anim.FuncAnimation(
            self._fig, frame_update, np.arange(1, len(self._t)), init, interval=va
        )

        progress_bar = tqdm(
            total=len(self._t),
            desc=f"Saving anim from {self.name}",
            unit="frame",
            leave=False,
            position=2,
        )
        if self._config["save"] and self._system.get_output_dir() is not None:
            anim_obj.save(
                self._system.get_output_dir() + f"{self.name}.mov".replace(" ", "_"),
                dpi=80,
                bitrate=-1,
                progress_callback=lambda i, n: progress_bar.update(1),
            )
        progress_bar.close()
        if self._config["show"]:
            plt.show()

        plt.close()
