"""Iterative Newton-Euler dynamics for the three-link manipulator.

Implements Eq. (6) (outward pass) and Eq. (7) (inward pass) of the reference
paper, following Craig's convention. The routine returns both the joint torques
tau_arm of Eq. (8) and the base force/torque pair (f_1, tau_1) that the paper
uses to build the cross-coupled terms of Eq. (10) and Eq. (11).

Frame convention
----------------
Frame {0} is the arm's base frame. Its origin coincides with the UAV body-frame
origin, as the paper assumes, but its axes are rotated so that each joint
rotates about the frame's z-axis and each link extends along the frame's x-axis:

    x_0 = -z_B   (the arm hangs below the vehicle)
    y_0 = -x_B
    z_0 =  y_B   (joint axis: the paper's "rotation about the y-axis")

That matches the paper's remark that the first joint rotates about the body
y-axis, which is why the reaction torque about y is the dominant one.

Sign of the reaction
--------------------
The paper defines f_arm_rea = -f_1 and tau_arm_rea = -tau_1 in Section II, but
Eqs. (5), (10), (11) and (12) then carry +f_1 and +tau_1 into the UAV model.
Only one of the two can be right. Newton's third law gives the reaction on the
UAV as the negative of the force the base exerts on link 1, so this module
returns f_1 and tau_1 as computed by the inward pass and lets the caller apply
the sign; `reaction_on_uav()` applies the physically correct negation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ARM, TOTAL_MASS, ArmParams

# Rotation from the UAV body frame B into the arm base frame {0}. Rows are the
# axes of {0} expressed in body coordinates.
R_BODY_TO_ARM = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
R_ARM_TO_BODY = R_BODY_TO_ARM.T

_Z_AXIS = np.array([0.0, 0.0, 1.0])


def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors.

    `np.cross` carries enough dispatch overhead to dominate this module: it
    costs about 60 us per call against 5 us for the explicit form, and the
    recursions below evaluate roughly thirty of them per pass. On the offline
    simulation that difference is the whole run time.
    """
    return np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class ArmReaction:
    """Result of one Newton-Euler evaluation."""

    joint_torques: np.ndarray  # tau_arm, Eq. (8), shape (3,)
    base_force: np.ndarray  # f_1 in body coordinates, shape (3,)
    base_torque: np.ndarray  # tau_1 in body coordinates, shape (3,)

    def reaction_on_uav(self) -> tuple[np.ndarray, np.ndarray]:
        """Force and torque the arm applies to the UAV, in body coordinates."""
        return -self.base_force, -self.base_torque


def newton_euler(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    omega: np.ndarray,
    omega_dot: np.ndarray,
    f_z: float,
    arm: ArmParams = ARM,
    total_mass: float = TOTAL_MASS,
) -> ArmReaction:
    """Run the outward and inward recursions once.

    Parameters
    ----------
    q, qd, qdd
        Joint angles, rates and accelerations, shape (n_links,).
    omega, omega_dot
        UAV body angular velocity and acceleration [p, q, r], in body
        coordinates, shape (3,).
    f_z
        Magnitude of the UAV thrust along the body z-axis. Following Eq. (6c)
        the base linear acceleration is v_dot_0 = [0, 0, f_z / m] in body
        coordinates, which already has gravity folded in.
    """
    n = arm.n_links
    inertia = arm.link_inertia()
    com = arm.com_offset()
    offset = arm.link_offset()

    # Base of the recursion, expressed in frame {0}.
    w = R_BODY_TO_ARM @ np.asarray(omega, dtype=float)
    wd = R_BODY_TO_ARM @ np.asarray(omega_dot, dtype=float)
    vd = R_BODY_TO_ARM @ np.array([0.0, 0.0, f_z / total_mass])

    # Rotation from frame {i} to frame {i+1} and the origin offset in frame {i}.
    # The first joint sits at the body origin, so its offset is zero.
    offsets = [np.zeros(3)] + [offset] * (n - 1)
    rotations = [_rot_z(float(q[i])).T for i in range(n)]  # {i+1}_i R

    forces: list[np.ndarray] = []
    moments: list[np.ndarray] = []

    # Outward pass, Eq. (6a)-(6f).
    for i in range(n):
        R = rotations[i]
        p = offsets[i]

        vd = R @ (cross(wd, p) + cross(w, cross(w, p)) + vd)
        wd = R @ wd + cross(R @ w, qd[i] * _Z_AXIS) + qdd[i] * _Z_AXIS
        w = R @ w + qd[i] * _Z_AXIS

        vd_com = cross(wd, com) + cross(w, cross(w, com)) + vd
        forces.append(arm.mass * vd_com)
        moments.append(inertia @ wd + cross(w, inertia @ w))

    # Inward pass, Eq. (7a)-(7c).
    f_next = np.zeros(3)
    tau_next = np.zeros(3)
    joint_torques = np.zeros(n)

    for i in range(n - 1, -1, -1):
        # Rotation from frame {i+1} back into frame {i}; the frame past the last
        # link carries no force, so its rotation is irrelevant.
        R_back = rotations[i + 1].T if i + 1 < n else np.eye(3)
        p_next = offsets[i + 1] if i + 1 < n else offset

        f_i = R_back @ f_next + forces[i]
        tau_i = (
            moments[i]
            + R_back @ tau_next
            + cross(com, forces[i])
            + cross(p_next, R_back @ f_next)
        )
        joint_torques[i] = float(tau_i @ _Z_AXIS)
        f_next, tau_next = f_i, tau_i

    # Viscous damping, which the recursions of Eq. (6)-(7) do not carry.
    #
    # These return the torque needed to produce the requested accelerations
    # through the links' inertia alone, so a joint that Gazebo damps needs that
    # much more torque to move at the same rate. Adding it here reaches every
    # caller consistently: `joint_accelerations` subtracts it as part of the
    # bias, `mass_matrix` probes at zero velocity so it stays clean, and
    # `manipulator_lpv` recovers it inside C because it is linear in Theta_dot.
    joint_torques = joint_torques + arm.damping * np.asarray(qd, dtype=float)

    # f_next / tau_next now hold f_1 and tau_1 in frame {0}; rotate to body.
    return ArmReaction(
        joint_torques=joint_torques,
        base_force=R_ARM_TO_BODY @ f_next,
        base_torque=R_ARM_TO_BODY @ tau_next,
    )


def mass_matrix(
    q: np.ndarray, arm: ArmParams = ARM, total_mass: float = TOTAL_MASS
) -> np.ndarray:
    """Manipulator mass matrix M(Theta) of Eq. (8).

    Obtained column by column: with all velocities and the base excitation set
    to zero, tau_arm reduces to M(Theta) @ qdd, so probing with unit joint
    accelerations reads off the columns directly.
    """
    n = arm.n_links
    zeros3 = np.zeros(3)
    columns = []
    for j in range(n):
        qdd = np.zeros(n)
        qdd[j] = 1.0
        result = newton_euler(
            q, np.zeros(n), qdd, zeros3, zeros3, 0.0, arm=arm, total_mass=total_mass
        )
        columns.append(result.joint_torques)
    return np.column_stack(columns)


def joint_accelerations(
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    omega: np.ndarray,
    omega_dot: np.ndarray,
    f_z: float,
    arm: ArmParams = ARM,
    total_mass: float = TOTAL_MASS,
) -> np.ndarray:
    """Forward dynamics: solve Eq. (9) for qdd given the applied joint torques.

    Uses the standard decomposition tau = M(Theta) qdd + b, where b collects the
    Coriolis, centrifugal and gravity-like terms C and G of Eq. (8) and is
    obtained from one Newton-Euler evaluation with qdd = 0.
    """
    n = arm.n_links
    bias = newton_euler(
        q, qd, np.zeros(n), omega, omega_dot, f_z, arm=arm, total_mass=total_mass
    ).joint_torques
    return np.linalg.solve(mass_matrix(q, arm=arm, total_mass=total_mass), tau - bias)
