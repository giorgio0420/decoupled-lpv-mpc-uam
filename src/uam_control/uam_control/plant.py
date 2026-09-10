"""Truth model for the offline simulation.

The plant integrated here is the paper's own model, Eq. (12a) and (12b), with
the manipulator dynamics of Eq. (9). That is deliberate: the offline run
validates the *control design* against the model the controllers were derived
from, and reproduces the comparison of Section V. It does not validate the
model itself. Independent multibody physics is what the Gazebo run provides, and
disagreement between the two is informative rather than a bug in either.

The one thing the paper leaves implicit is that Eq. (12b) is an implicit
equation: the reaction matrices of Eq. (10) depend on the joint accelerations,
and the joint accelerations of Eq. (9) depend on the angular acceleration. Both
sides are affine in the pair (omega_dot, Theta_ddot), so the coupled system is
solved exactly here rather than lagged by one step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .controllers import euler_rate_to_body_rate, rotation_body_to_inertial
from .coupling import uav_gyroscopic_matrices
from .newton_euler import cross, newton_euler
from .params import ARM, GRAVITY, TOTAL_MASS, UAV


@dataclass
class UamState:
    """Full state of the aerial manipulator."""

    position: np.ndarray  # zeta = [x, y, z]
    velocity: np.ndarray  # zeta_dot
    attitude: np.ndarray  # eta = [phi, theta, psi]
    body_rate: np.ndarray  # omega = [p, q, r]
    joints: np.ndarray  # Theta
    joint_rates: np.ndarray  # Theta_dot

    @classmethod
    def at_rest(cls, height: float = 1.5) -> "UamState":
        n = ARM.n_links
        return cls(
            position=np.array([0.0, 0.0, height]),
            velocity=np.zeros(3),
            attitude=np.zeros(3),
            body_rate=np.zeros(3),
            joints=np.zeros(n),
            joint_rates=np.zeros(n),
        )

    def translational_vector(self) -> np.ndarray:
        """State ordering [x, xdot, y, ydot, z, zdot] used by the LMPC."""
        return np.array(
            [
                self.position[0],
                self.velocity[0],
                self.position[1],
                self.velocity[1],
                self.position[2],
                self.velocity[2],
            ]
        )

    def joint_vector(self) -> np.ndarray:
        return np.concatenate([self.joints, self.joint_rates])


class DivergedError(RuntimeError):
    """Raised when the plant state has left the range the model describes."""


@dataclass
class Accelerations:
    linear: np.ndarray
    body_rate_dot: np.ndarray
    joint_dot: np.ndarray


def solve_accelerations(
    state: UamState,
    thrust: float,
    uav_torque: np.ndarray,
    joint_torque: np.ndarray,
    disturbance_force: np.ndarray | None = None,
    disturbance_torque: np.ndarray | None = None,
) -> Accelerations:
    """Solve Eq. (12) and Eq. (9) simultaneously for the accelerations.

    Both the rotational residual and the joint residual are affine in the
    unknown pair z = (omega_dot, Theta_ddot), so the exact solution comes from
    one evaluation at zero plus one per basis direction.
    """
    n = ARM.n_links
    omega = state.body_rate
    inertia = UAV.inertia_tensor()
    gyroscopic = cross(omega, inertia @ omega)

    def residual(z: np.ndarray) -> np.ndarray:
        omega_dot, joint_ddot = z[:3], z[3:]
        result = newton_euler(
            state.joints, state.joint_rates, joint_ddot, omega, omega_dot, thrust
        )
        _, reaction_torque = result.reaction_on_uav()
        rotational = (
            inertia @ omega_dot + gyroscopic - uav_torque - reaction_torque
        )
        manipulator = result.joint_torques - joint_torque
        return np.concatenate([rotational, manipulator])

    size = 3 + n
    offset = residual(np.zeros(size))
    jacobian = np.column_stack(
        [residual(np.eye(size)[j]) - offset for j in range(size)]
    )
    # The coupled mass matrix is non-singular for any physical configuration, so
    # a singular Jacobian means the state has already left the physical regime
    # (usually an arm rate large enough to overflow). Report that as a diverged
    # simulation rather than as a linear-algebra failure, which says nothing
    # about the cause.
    try:
        solution = np.linalg.solve(jacobian, -offset)
    except np.linalg.LinAlgError as error:
        raise DivergedError(
            "coupled mass matrix became singular; the state has already left the "
            f"physical regime (|omega| = {np.linalg.norm(omega):.3g} rad/s, "
            f"|Theta_dot| = {np.linalg.norm(state.joint_rates):.3g} rad/s)"
        ) from error
    omega_dot, joint_ddot = solution[:3], solution[3:]

    # Translational motion, Eq. (12a). The reaction force is evaluated at the
    # accelerations just solved for, so it is consistent with the rotation.
    reaction_force, _ = newton_euler(
        state.joints, state.joint_rates, joint_ddot, omega, omega_dot, thrust
    ).reaction_on_uav()
    body_force = reaction_force + np.array([0.0, 0.0, thrust])
    rotation = rotation_body_to_inertial(*state.attitude)
    linear = rotation @ body_force / UAV.mass - np.array([0.0, 0.0, GRAVITY])

    if disturbance_force is not None:
        linear = linear + disturbance_force / UAV.mass
    if disturbance_torque is not None:
        omega_dot = omega_dot + np.linalg.solve(inertia, disturbance_torque)

    return Accelerations(linear=linear, body_rate_dot=omega_dot, joint_dot=joint_ddot)


def integrate(state: UamState, accelerations: Accelerations, dt: float) -> UamState:
    """Semi-implicit Euler step.

    Velocities advance first and positions use the updated velocity, which keeps
    the oscillatory attitude modes from gaining energy the way explicit Euler
    would at these step sizes.
    """
    velocity = state.velocity + dt * accelerations.linear
    body_rate = state.body_rate + dt * accelerations.body_rate_dot
    joint_rates = state.joint_rates + dt * accelerations.joint_dot

    phi, theta = state.attitude[0], state.attitude[1]
    euler_rates = np.linalg.solve(euler_rate_to_body_rate(phi, theta), body_rate)

    return replace(
        state,
        position=state.position + dt * velocity,
        velocity=velocity,
        attitude=state.attitude + dt * euler_rates,
        body_rate=body_rate,
        joints=state.joints + dt * joint_rates,
        joint_rates=joint_rates,
    )
