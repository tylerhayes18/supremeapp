"""Rigid-body dynamics validation of the SkyArm in PyBullet.

Builds the machine as a URDF straight from skyarm.spec (real masses,
link lengths, inertias) and runs it under gravity with TORQUE-LIMITED
motors: each joint may apply at most its gearbox output capacity from
the spec.  If a motor/gearbox combination is undersized, the arm sags
or oscillates and the tracking test fails — this is the mechanical
proof the static torque budget in spec.py actually holds up
dynamically, including inertial loads the static analysis ignores.

    python -m skyarm.sim.dynamics          # run all scenarios, report

Scenarios:
  hold_horizontal   worst gravity case: full arm horizontal + payload,
                    motors must hold position with bounded droop
  mission           the demo pick-and-place trajectory, tracked with
                    torque-limited PD control; bounded tracking error
  payload_sweep     increasing payload at full horizontal reach until
                    the shoulder can no longer hold — reports the real
                    dynamic payload margin
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .. import kinematics as kin
from .. import spec

G = 9.81


# ---------------------------------------------------------------------------
# URDF generation from the spec
# ---------------------------------------------------------------------------

def _box_inertia(m, x, y, z):
    return (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12,
            m * (x * x + y * y) / 12)


def _link(name, mass, size, xyz="0 0 0", rpy="0 0 0"):
    ix, iy, iz = _box_inertia(mass, *size)
    sx, sy, sz = size
    return f"""
  <link name="{name}">
    <inertial>
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <mass value="{mass}"/>
      <inertia ixx="{ix:.6f}" iyy="{iy:.6f}" izz="{iz:.6f}"
               ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <geometry><box size="{sx} {sy} {sz}"/></geometry>
    </collision>
  </link>"""


def build_urdf(payload_kg: float = spec.PAYLOAD_KG) -> str:
    """The machine as a URDF string (metres, kg).  Joint order matches
    skyarm.sim.world: gx, gy, yaw, shoulder, elbow, wrist_pitch."""
    L1 = spec.UPPER_ARM_MM / 1000.0
    L2 = spec.FOREARM_MM / 1000.0
    L3 = spec.WRIST_TO_TIP_MM / 1000.0
    stack = spec.CARRIAGE_STACK_MM / 1000.0

    # masses from the spec budget; motor masses ride on the proximal link
    m_carriage = spec.CARRIAGE_MASS_KG + 2.2 + spec.NEMA24.mass_kg
    m_upper = spec.M_UPPER_TUBE
    m_elbow_assy = spec.M_ELBOW_ASSY            # at the elbow, on upper link
    m_forearm = spec.M_FOREARM_TUBE
    m_wrist = spec.M_WRIST_CLUSTER              # at wrist, on forearm link
    m_tip = spec.M_GRIPPER + payload_kg

    u = []
    u.append('<?xml version="1.0"?>\n<robot name="skyarm">')
    u.append(_link("base", 50.0, (0.2, 0.2, 0.02)))
    u.append(_link("bridge", spec.BRIDGE_MASS_KG, (0.06, 2.6, 0.06)))
    u.append(_link("carriage", m_carriage, (0.16, 0.15, stack),
                   xyz=f"0 0 {-stack / 2}"))
    # upper link carries tube (cg mid) + elbow assembly (cg at far end)
    cg_u = (m_upper * L1 / 2 + m_elbow_assy * L1) / (m_upper + m_elbow_assy)
    u.append(_link("upper", m_upper + m_elbow_assy, (0.05, 0.05, L1),
                   xyz=f"0 0 {-cg_u}"))
    cg_f = (m_forearm * L2 / 2 + m_wrist * L2) / (m_forearm + m_wrist)
    u.append(_link("forearm", m_forearm + m_wrist, (0.04, 0.04, L2),
                   xyz=f"0 0 {-cg_f}"))
    u.append(_link("hand", m_tip, (0.05, 0.05, L3), xyz=f"0 0 {-L3 / 2}"))

    def joint(name, jtype, parent, child, xyz, axis, lo=0.0, hi=0.0):
        lim = (f'<limit lower="{lo}" upper="{hi}" effort="1000" '
               'velocity="10"/>') if jtype == "revolute" else \
              (f'<limit lower="{lo}" upper="{hi}" effort="2000" '
               'velocity="2"/>') if jtype == "prismatic" else ""
        return f"""
  <joint name="{name}" type="{jtype}">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{xyz}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>{lim}
  </joint>"""

    xt = kin.X_TRAVEL_MM / 1000.0
    yt = kin.Y_TRAVEL_MM / 1000.0
    u.append(joint("gx", "prismatic", "base", "bridge", "0 0 0", "1 0 0", 0, xt))
    u.append(joint("gy", "prismatic", "bridge", "carriage", "0 0 0", "0 1 0", 0, yt))
    u.append(joint("yaw", "revolute", "carriage", "upper_pre", "0 0 0",
                   "0 0 1", -math.pi, math.pi))
    u.append(_link("upper_pre", 0.05, (0.04, 0.04, 0.04)))
    u.append(joint("shoulder", "revolute", "upper_pre", "upper",
                   f"0 0 {-stack}", "0 1 0",
                   math.radians(spec.SHOULDER.travel_deg[0]),
                   math.radians(spec.SHOULDER.travel_deg[1])))
    u.append(joint("elbow", "revolute", "upper", "forearm", f"0 0 {-L1}",
                   "0 1 0", math.radians(spec.ELBOW.travel_deg[0]),
                   math.radians(spec.ELBOW.travel_deg[1])))
    u.append(joint("wrist", "revolute", "forearm", "hand", f"0 0 {-L2}",
                   "0 1 0", math.radians(spec.WRIST_PITCH.travel_deg[0]),
                   math.radians(spec.WRIST_PITCH.travel_deg[1])))
    u.append("\n</robot>")
    return "".join(u)


# ---------------------------------------------------------------------------
# Torque-limited simulation harness
# ---------------------------------------------------------------------------

JOINTS = ("gx", "gy", "yaw", "shoulder", "elbow", "wrist")

# gearbox output capacity per revolute joint (Nm); gantry force capacity (N)
CAPACITY = {
    "gx": spec.X_AXIS.capacity_n(),
    "gy": spec.Y_AXIS.capacity_n(),
    "yaw": spec.YAW.capacity_nm(),
    "shoulder": spec.SHOULDER.capacity_nm(),
    "elbow": spec.ELBOW.capacity_nm(),
    "wrist": spec.WRIST_PITCH.capacity_nm(),
}


@dataclass
class SimResult:
    max_err: dict            # joint -> worst |target-actual| (rad or m)
    max_torque: dict         # joint -> worst commanded effort (transients)
    steady_torque: dict = None  # joint -> mean effort over the final second
    steady_err: dict = None     # joint -> mean |error| over the final second
    droop_tip_mm: float = 0.0


class Dynamics:
    def __init__(self, payload_kg: float = spec.PAYLOAD_KG):
        import pybullet as pb
        self.pb = pb
        self.cid = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -G, physicsClientId=self.cid)
        pb.setTimeStep(1 / 480.0, physicsClientId=self.cid)
        urdf = build_urdf(payload_kg)
        with tempfile.NamedTemporaryFile("w", suffix=".urdf",
                                         delete=False) as f:
            f.write(urdf)
            path = f.name
        ceil_z = kin.CEILING_MM / 1000.0
        self.robot = pb.loadURDF(path, basePosition=(0, 0, ceil_z),
                                 useFixedBase=True, physicsClientId=self.cid)
        self.jmap = {}
        for i in range(pb.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = pb.getJointInfo(self.robot, i, physicsClientId=self.cid)
            self.jmap[info[1].decode()] = i
        # disable default velocity motors so we control torque explicitly
        for name in JOINTS:
            pb.setJointMotorControl2(self.robot, self.jmap[name],
                                     pb.VELOCITY_CONTROL, force=0,
                                     physicsClientId=self.cid)

    def close(self):
        self.pb.disconnect(self.cid)

    def set_state(self, targets):
        for name, val in targets.items():
            self.pb.resetJointState(self.robot, self.jmap[name], val,
                                    physicsClientId=self.cid)

    def tip_z(self):
        import pybullet as pb
        st = self.pb.getLinkState(self.robot, self.jmap["wrist"],
                                  computeForwardKinematics=True,
                                  physicsClientId=self.cid)
        pos, orn = st[4], st[5]
        l3 = spec.WRIST_TO_TIP_MM / 1000.0
        tip, _ = self.pb.multiplyTransforms(pos, orn, (0, 0, -l3),
                                            (0, 0, 0, 1))
        return tip[2]

    # closed-loop steppers act as stiff position servos; model them as
    # gravity-feedforward + PD, torque-clamped to the gearbox capacity.
    # kd must satisfy kd/I < timestep rate or the discrete loop chatters
    # (the wrist, with ~0.013 kg.m^2 reflected inertia, found that out).
    GAINS = {"gx": (40000.0, 4000.0), "gy": (30000.0, 3000.0),
             "yaw": (400.0, 40.0), "shoulder": (1500.0, 150.0),
             "elbow": (600.0, 60.0), "wrist": (40.0, 1.5)}

    def run(self, target_fn, seconds) -> SimResult:
        """Step with feedforward+PD control clamped to gearbox capacity."""
        pb = self.pb
        steps = int(seconds * 480)
        max_err = {n: 0.0 for n in JOINTS}
        max_tau = {n: 0.0 for n in JOINTS}
        acc_tau = {n: 0.0 for n in JOINTS}
        acc_err = {n: 0.0 for n in JOINTS}
        n_acc = 0
        t_steady = seconds - 1.0
        idx = [self.jmap[n] for n in JOINTS]
        dt = 1 / 480.0
        for s in range(steps):
            t = s / 480.0
            targets = target_fn(t)
            targets_prev = target_fn(max(t - dt, 0.0))
            tvel = {n: (targets[n] - targets_prev[n]) / dt for n in JOINTS}
            q, qd = [], []
            for j in idx:
                st = pb.getJointState(self.robot, j, physicsClientId=self.cid)
                q.append(st[0])
                qd.append(st[1])
            # gravity + coriolis compensation (what current-controlled
            # closed-loop drives effectively deliver)
            ff = pb.calculateInverseDynamics(self.robot, q, qd,
                                             [0.0] * len(q),
                                             physicsClientId=self.cid)
            for k, name in enumerate(JOINTS):
                kp, kd = self.GAINS[name]
                err = targets[name] - q[k]
                tau = ff[k] + kp * err + kd * (tvel[name] - qd[k])
                cap = CAPACITY[name]
                tau = max(-cap, min(cap, tau))
                pb.setJointMotorControl2(self.robot, idx[k],
                                         pb.TORQUE_CONTROL, force=tau,
                                         physicsClientId=self.cid)
                if t > 0.5:           # ignore the initial transient
                    max_err[name] = max(max_err[name], abs(err))
                    max_tau[name] = max(max_tau[name], abs(tau))
                if t >= t_steady:
                    acc_tau[name] += abs(tau)
                    acc_err[name] += abs(err)
                    if name == JOINTS[-1]:
                        n_acc += 1
            pb.stepSimulation(physicsClientId=self.cid)
        n_acc = max(n_acc, 1)
        return SimResult(max_err, max_tau,
                         {n: acc_tau[n] / n_acc for n in JOINTS},
                         {n: acc_err[n] / n_acc for n in JOINTS})


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def hold_horizontal(payload_kg=spec.PAYLOAD_KG, seconds=4.0) -> SimResult:
    """Arm fully horizontal with payload: motors must hold against
    gravity using no more than spec capacity."""
    sim = Dynamics(payload_kg)
    pose = {"gx": 1.0, "gy": 1.0, "yaw": 0.0,
            "shoulder": math.pi / 2, "elbow": 0.0, "wrist": 0.0}
    sim.set_state(pose)
    z0 = sim.tip_z()
    res = sim.run(lambda t: pose, seconds)
    res.droop_tip_mm = (z0 - sim.tip_z()) * 1000.0
    sim.close()
    return res


def mission(seconds_per_leg=2.5) -> SimResult:
    """Track the demo mission waypoints (joint-space, smooth blends)."""
    from .world import World, demo_mission
    world = demo_mission(World())
    waypoints = []
    while world.mission or not world.settled:
        world.step(1 / 120)
        if world.settled and world.mission:
            pass
    # rebuild the list of settled poses by re-running mission symbolically
    world2 = demo_mission(World())
    poses = [world2.pose()]
    for item in list(world2.mission):
        if item[0] == "goto":
            poses.append(kin.inverse(item[1], item[2]))
    legs = []
    for p in poses:
        legs.append({"gx": p.gx / 1000.0, "gy": p.gy / 1000.0,
                     "yaw": math.radians(p.yaw),
                     "shoulder": math.radians(p.shoulder),
                     "elbow": math.radians(p.elbow),
                     "wrist": math.radians(p.wrist_pitch)})

    def smooth(a, b, u):
        u = min(max(u, 0.0), 1.0)
        u = 3 * u * u - 2 * u ** 3
        return {k: a[k] + (b[k] - a[k]) * u for k in a}

    total = seconds_per_leg * (len(legs) - 1)

    def target_fn(t):
        i = min(int(t / seconds_per_leg), len(legs) - 2)
        return smooth(legs[i], legs[i + 1], t / seconds_per_leg - i)

    sim = Dynamics()
    sim.set_state(legs[0])
    res = sim.run(target_fn, total)
    sim.close()
    return res


def payload_sweep(max_kg=4.0, step=0.5):
    """Increase payload at full horizontal reach; report droop."""
    rows = []
    kg = 0.5
    while kg <= max_kg + 1e-9:
        r = hold_horizontal(kg, seconds=4.0)
        rows.append((kg, r.droop_tip_mm, r.steady_torque["shoulder"]))
        kg += step
    return rows


def report() -> str:
    lines = ["SkyArm dynamics validation (PyBullet, torque-limited)",
             "=" * 60]
    r = hold_horizontal()
    lines.append(f"\n[hold horizontal, {spec.PAYLOAD_KG} kg payload]")
    lines.append(f"  tip droop under hold: {r.droop_tip_mm:6.1f} mm")
    for j in ("shoulder", "elbow", "wrist"):
        lines.append(f"  {j:9s} steady torque {r.steady_torque[j]:5.1f}"
                     f" (peak {r.max_torque[j]:5.1f})"
                     f" / {CAPACITY[j]:.1f} Nm capacity")

    m = mission()
    lines.append("\n[demo mission tracking]")
    worst = max((v for k, v in m.max_err.items()
                 if k not in ("gx", "gy")), default=0.0)
    lines.append(f"  worst joint tracking error: {math.degrees(worst):.2f} deg")
    lines.append(f"  worst gantry tracking error: "
                 f"{max(m.max_err['gx'], m.max_err['gy']) * 1000:.1f} mm")
    for k in ("shoulder", "elbow", "wrist"):
        lines.append(f"  {k:9s} peak torque {m.max_torque[k]:6.1f}"
                     f" / {CAPACITY[k]:.1f} Nm")

    lines.append("\n[payload sweep, arm horizontal]")
    for kg, droop, tau in payload_sweep():
        flag = "OK " if droop < 30 else "SAG"
        lines.append(f"  {kg:3.1f} kg: droop {droop:7.1f} mm, "
                     f"shoulder {tau:5.1f} Nm  {flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
