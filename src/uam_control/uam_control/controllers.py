"""The three controllers of Section III, plus the ERTF baseline of Section V.

Layout follows Fig. 2 of the paper:

    translational reference -> constrained LMPC          (Section III-A-1)
                            -> attitude reference        (Section III-A-2)
                            -> tube-based LPV-MPC        (Section III-B)
                            -> UAV torques

    joint reference         -> tube-based LPV-MPC        (Section III-C)
                            -> joint torques
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from .coupling import max_joint_acceleration, CrossCoupling, manipulator_lpv, rotational_lpv
from .mpc import _INF, BoxConstraint, Tube, lqr_gain, riccati_terminal_cost, solve_mpc
from .newton_euler import newton_euler
from .params import (
    ARM,
    CONTROL,
    GRAVITY,
    LIMITS,
    LPV,
    TOTAL_MASS,
    UAV,
)


# Impose Theorem 1's terminal constraint. Off by default, and the reason is
# measured rather than assumed.
#
# The set itself is correct: `rpi_check.py` verifies invariance from all 64
# corners over 500 steps, and admissibility of the feedback that holds it. But it
# is small -- 0.66 % of the state box for the rotational loop, 1.5 % for the
# manipulator -- and a four-step horizon cannot steer into it from a tracking
# trajectory. Flown with it on, the nominal scenario went from 0.21 m of error to
# 28 m and the pitch reached 1.49 rad: the QP is over-constrained, not unstable.
#
# So the guarantee and the flying configuration are, at N = 4, alternatives
# rather than companions. Turning this on demonstrates Theorem 1 holding; leaving
# it off is what tracks. What would make them one thing is a longer horizon,
# which is a compute question, not a control one.
TERMINAL_SET = os.environ.get('UAM_TERMINAL_SET', '0') == '1'


def rotation_body_to_inertial(phi: float, theta: float, psi: float) -> np.ndarray:
    """Rotation matrix of Eq. (1)."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cps * cth, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
            [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
            [-sth, sph * cth, cph * cth],
        ]
    )


def euler_rate_to_body_rate(phi: float, theta: float) -> np.ndarray:
    """Transformation B_T_I of Eq. (2), mapping Euler rates to body rates."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, -sth],
            [0.0, cph, sph * cth],
            [0.0, -sph, cph * cth],
        ]
    )


def _rotation(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body-to-inertial rotation from the Z-Y-X Euler angles used throughout."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps * cth, cps * sth * sph - sps * cph, cps * sth * cph + sps * sph],
        [sps * cth, sps * sth * sph + cps * cph, sps * sth * cph - cps * sph],
        [-sth, cth * sph, cth * cph],
    ])


def attitude_rate_command(attitude: np.ndarray, roll_ref: float,
                          pitch_ref: float, yaw_ref: float,
                          gain: np.ndarray) -> np.ndarray:
    """Commanded body rate from an attitude error, without an Euler degeneracy.

    Why this replaces `B_T_I @ (gain * euler_error)`.

    That expression looks like a coordinate change and is really a gain that
    depends on the attitude. Its pitch row is [0, cos(phi), sin(phi) cos(theta)],
    so the commanded pitch rate carries a factor of cos(roll). While the vehicle
    is near level that factor is 1 and the difference does not show. In a tumble
    the roll passes through +/-pi/2, cos(roll) goes to zero, and the pitch command
    is annihilated no matter how large the pitch error is.

    Measured, in the run that crashed: pitch at 1.41 rad, body rate at 8.55 rad/s,
    and the commanded torque between -0.04 and +0.03 N m for half a second, all
    the way down from 3.4 m to the ground. The vehicle was not unrecoverable --
    the controller had stopped asking for anything. Every diverged run in this
    project ends with that signature, and it turns an excursion that the rate loop
    could absorb into an impact.

    The geometric error 0.5 vee(R_d^T R - R^T R_d) is the standard replacement. It
    is smooth and non-zero for every attitude short of exactly inverted, it needs
    no small-angle assumption, and it is already in the body frame, which is where
    the rate loop wants its reference.
    """
    R = _rotation(*attitude)
    R_d = _rotation(roll_ref, pitch_ref, yaw_ref)
    error_matrix = R_d.T @ R - R.T @ R_d
    # vee of the skew part: the axis whose rotation carries R onto R_d.
    error = 0.5 * np.array([error_matrix[2, 1],
                            error_matrix[0, 2],
                            error_matrix[1, 0]])
    return -gain * error


# --------------------------------------------------------------------------
# Outer position loop
#
# The paper puts a constrained linear MPC here, Eq. (13)-(18). It is replaced by
# a PD, for two reasons.
#
# It is the least distinctive stage of the paper. The three contributions the
# authors claim are the cross-coupled decomposition, the tube-based LPV-MPC for
# the rotational dynamics, and the tube-based LPV-MPC for the manipulator. This
# stage is none of them: the paper itself describes it as an ordinary
# constrained LMPC, with no LPV scheduling and no tube. Dropping it leaves all
# three contributions still exercised, and leaves one MPC on the vehicle and one
# on the arm.
#
# It also brought Eq. (19)-(21) with it, which is where roll and pitch were
# computed by different formulas and where roll diverged. See
# `attitude_reference` below.
#
# `TranslationalMPC` is kept for comparison but is no longer on the default
# path; `PositionPD` is.
# --------------------------------------------------------------------------


class PositionPD:
    """Position and velocity PD producing the inertial force command.

    Returns the same quantity the MPC returned, `[u_x, u_y, u_z + m g]` of
    Eq. (14): the total inertial force the rotors must produce, gravity
    included. That keeps the interface to `attitude_reference` unchanged.

    Gains are set from a target closed-loop natural frequency with critical
    damping, `kp = wn^2` and `kd = 2 wn`, so there is one number to tune per
    axis rather than a weight matrix and a horizon.

    The horizontal bandwidth has to stay well below the attitude loop beneath
    it. This is a cascade: position asks for an attitude, attitude asks for a
    body rate, the MPC delivers the torque. Each inner loop needs to be several
    times faster than the one commanding it.

    At 1.5 rad/s against an attitude gain of 2.0 the separation was 1.3x and the
    vehicle sat in a limit cycle -- attitude swinging 0.35 rad, body rates
    alternating sign at 1 rad/s every sample, position wandering about 2 m. The
    rotational MPC was not at fault: its nominal state tracked the real one to
    within 0.05 rad/s throughout, so the tube was working. It was faithfully
    following an oscillating reference handed down from above.

    0.5 gave 4x separation and held position dead still with the arm bolted
    fixed. With the arm free it is too slow to reject the arm's reaction: the
    vehicle wanders about 1.5 m horizontally around a stationary reference while
    holding altitude to 0.08 m, because thrust reaches z directly but horizontal
    force only through the attitude cascade. 0.8 keeps 2.5x separation and
    trades some of that margin for rejection.
    """

    def __init__(
        self,
        # UAV.mass, not TOTAL_MASS. Eq. (12a) divides by the vehicle mass and
        # carries the arm's share separately, inside the reaction force -- and
        # `plant.py` follows that convention. Demanding TOTAL_MASS * a here
        # while the plant divides by UAV.mass inflates every thrust command by
        # TOTAL_MASS / UAV.mass, about 10 %, permanently. With that error the
        # hover point sits 0.22 m above the commanded altitude and the initial
        # thrust command is 59.25 N against the 53.88 N the vehicle weighs.
        mass: float = UAV.mass,
        bandwidth: float = 0.8,
        vertical_bandwidth: float = 2.0,
    ) -> None:
        self.mass = mass
        self.kp = np.array([bandwidth**2, bandwidth**2, vertical_bandwidth**2])
        self.kd = np.array([2.0 * bandwidth, 2.0 * bandwidth, 2.0 * vertical_bandwidth])

    def __call__(
        self, state: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Force command from the current translational state.

        `state` is [x, xdot, y, ydot, z, zdot]; `reference` is the horizon array
        the MPC used, of which only the first row is needed here.
        """
        target = np.atleast_2d(reference)[0]
        position = np.array([state[0], state[2], state[4]])
        velocity = np.array([state[1], state[3], state[5]])
        position_ref = np.array([target[0], target[2], target[4]])
        velocity_ref = np.array([target[1], target[3], target[5]])

        acceleration = self.kp * (position_ref - position) + self.kd * (
            velocity_ref - velocity
        )
        acceleration = np.clip(acceleration, -LIMITS.acceleration, LIMITS.acceleration)

        force = self.mass * acceleration
        force[2] += self.mass * GRAVITY
        force[0] = float(np.clip(force[0], -LIMITS.force_xy, LIMITS.force_xy))
        force[1] = float(np.clip(force[1], -LIMITS.force_xy, LIMITS.force_xy))
        force[2] = float(np.clip(force[2], LIMITS.force_z_min, LIMITS.force_z_max))
        return force


class TranslationalMPC:
    """Constrained LMPC of Eq. (13)-(18) in the augmented form of Eq. (16).

    The plant state is x_m = [x, xdot, y, ydot, z, zdot]; the augmented state
    x_zeta = [delta x_m, y_m] of Eq. (17) stacks the state increment with the
    measured output so that the decision variable is delta u and the loop
    carries integral action. That is why Table II lists Q_zeta as 10 * I_12.
    """

    def __init__(self, mass: float = UAV.mass) -> None:
        self.mass = mass
        self.horizon = CONTROL.N_zeta
        dt = CONTROL.dt_zeta

        a_block = np.array([[1.0, dt], [0.0, 1.0]])
        A_m = linalg.block_diag(a_block, a_block, a_block)
        B_m = np.zeros((6, 3))
        for axis in range(3):
            B_m[2 * axis + 1, axis] = dt / mass
        C_m = np.eye(6)

        self.A = np.block([[A_m, np.zeros((6, 6))], [C_m @ A_m, np.eye(6)]])
        self.B = np.vstack([B_m, C_m @ B_m])

        self.Q = CONTROL.Q_zeta()
        self.R = CONTROL.R_zeta()
        # The terminal weight is simply Q, not a Riccati solution. With
        # C_zeta,m = I_6 the augmented model of Eq. (17) integrates position and
        # velocity alike, which puts two integrators on a two-state single-input
        # axis and leaves uncontrollable modes on the unit circle; the discrete
        # Riccati equation has no stabilising solution there. The paper does not
        # rely on one either, taking the translational stability argument from
        # reference [51] instead of from the tube construction.
        self.P = self.Q.copy()

        self._previous_state: np.ndarray | None = None
        self._previous_input = np.zeros(3)
        # u_zeta(1|k), the force the *same* solve predicts one step ahead. Not
        # a second controller call -- one QP already commits to a trajectory
        # over the whole horizon, and this is step 1 of it. See the docstring
        # on `predicted_force_next` for what it is for.
        self.predicted_force_next = np.array([0.0, 0.0, mass * GRAVITY])

        self.state_bounds = [
            BoxConstraint(
                lower=np.concatenate([np.full(6, -_INF), self._plant_lower()]),
                upper=np.concatenate([np.full(6, _INF), self._plant_upper()]),
            )
        ] * self.horizon
        # Bound on the input *increment*, which is the decision variable of the
        # velocity-form model.
        self.input_bounds = [BoxConstraint.symmetric(np.full(3, 25.0))] * self.horizon

        # The actuator's *absolute* bound, on the cumulative sum of the
        # increment decision variable -- see `solve_mpc`'s `cumulative_bounds`.
        # Net force pre-gravity: symmetric on x, y; on z, the rotors' [force_z_min,
        # force_z_max] range shifted by the weight this controller will add back.
        weight = self.mass * GRAVITY
        absolute = BoxConstraint(
            lower=np.array([-LIMITS.force_xy, -LIMITS.force_xy,
                             LIMITS.force_z_min - weight]),
            upper=np.array([LIMITS.force_xy, LIMITS.force_xy,
                             LIMITS.force_z_max - weight]),
        )
        self.cumulative_bounds = [absolute] * self.horizon

    @staticmethod
    def _plant_lower() -> np.ndarray:
        return np.array(
            [-LIMITS.position, -LIMITS.velocity] * 3, dtype=float
        )

    @staticmethod
    def _plant_upper() -> np.ndarray:
        return np.array([LIMITS.position, LIMITS.velocity] * 3, dtype=float)

    def __call__(self, plant_state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return the optimal force vector u_zeta = [u_x, u_y, u_z + m g].

        `reference` holds the desired plant state at each of the N_zeta
        prediction steps, shape (N_zeta, 6).

        Note what the internal decision variable is. Eq. (15) gives every
        velocity row of B_zeta,m the same dt / m, so the prediction model is
        v_dot = u / m on all three axes with no gravity term: `u` is the *net*
        force and is zero on the z-axis at hover. Eq. (13) then labels the third
        component of the input vector `u_z + m g`, because that is the force the
        rotors have to produce once gravity is added back. The two are not the
        same quantity, and the hand-off happens here: the optimiser works in net
        force, and gravity is restored on the way out to Eq. (14) and (19).
        """
        if self._previous_state is None:
            # Starting from a zero previous state would present the whole
            # initial state as a one-step increment and kick the loop.
            self._previous_state = plant_state.copy()

        augmented = np.concatenate([plant_state - self._previous_state, plant_state])
        target = np.hstack([np.zeros((self.horizon, 6)), reference])

        delta_u_sequence = solve_mpc(
            augmented,
            target,
            [self.A] * self.horizon,
            [self.B] * self.horizon,
            self.Q,
            self.R,
            self.P,
            self.state_bounds,
            self.input_bounds,
            self.cumulative_bounds,
            self._previous_input,
        )

        self._previous_state = plant_state.copy()
        # The QP now plans against the same absolute ceiling this applies, so
        # the clip below is a numerical safety net, not where the actuator
        # limit is actually enforced.
        self._previous_input = np.clip(
            self._previous_input + delta_u_sequence[0], -LIMITS.force_xy, LIMITS.force_xy
        )
        weight = self.mass * GRAVITY
        u_zeta = self._previous_input.copy()
        u_zeta[2] = np.clip(
            u_zeta[2] + weight, LIMITS.force_z_min, LIMITS.force_z_max
        )

        # Step 1 of the same predicted trajectory, same clipping, so a caller
        # differencing this against u_zeta gets the rate the solve actually
        # committed to rather than a discontinuity from replanning between
        # calls. See `predicted_force_next`'s note at __init__.
        next_force = np.clip(
            self._previous_input + delta_u_sequence[1], -LIMITS.force_xy, LIMITS.force_xy
        )
        next_force[2] = np.clip(
            next_force[2] + weight, LIMITS.force_z_min, LIMITS.force_z_max
        )
        self.predicted_force_next = next_force
        return u_zeta


# --------------------------------------------------------------------------
# Section III-A-2: attitude reference from the translational solution
# --------------------------------------------------------------------------


@dataclass
class AttitudeReference:
    """Result of solving Eq. (19)-(21)."""

    thrust: float
    roll: float
    pitch: float


def attitude_reference(
    u_optimal: np.ndarray,
    coupling: CrossCoupling,
    yaw_reference: float,
    attitude: np.ndarray | None = None,
) -> AttitudeReference:
    """Recover the thrust and the reference roll and pitch from a force command.

    This replaces Eq. (19)-(21). Those equations solve a quadratic for the
    thrust and then invert Eq. (14) with two `atan2` expressions, one per angle,
    and the two are not the same shape: roll carries a
    `sqrt(F_y^2 + F_z^2 - c_phi^2)` that is clamped when the argument goes
    negative, while pitch does not. That asymmetry is a defect, not a feature of
    the physics -- the trajectory is symmetric in x and y, and roll was the
    channel that diverged.

    The standard multirotor construction is used instead. It is symmetric in the
    two axes by build, has no discriminant to clamp, and needs one `asin` and one
    `atan2` rather than a quadratic and two `atan2`:

        z_b = f_cmd / |f_cmd|                       desired body z axis
        sin(phi)  =  z_b,x sin(psi) - z_b,y cos(psi)
        tan(theta) = (z_b,x cos(psi) + z_b,y sin(psi)) / z_b,z

    Both identities fall straight out of the third column of Eq. (1), which is
    exactly what the two lines above invert.

    The arm's reaction is still accounted for, so the cross-coupled model still
    drives this stage. Eq. (11) gives the reaction as `M_f f_z + f_bar`; the
    constant part is rotated into the inertial frame with the current attitude
    and subtracted from the demand, and the thrust is then scaled by
    `|M_f + e_z|` so that the rotors supply what is actually needed.
    """
    demand = np.asarray(u_optimal, dtype=float)

    # Eq. (11): the part of the reaction that does not scale with thrust acts as
    # a force the rotors have to cancel. Express it in the inertial frame.
    if attitude is not None:
        rotation = rotation_body_to_inertial(*attitude)
        demand = demand - rotation @ coupling.f_bar

    magnitude = float(np.linalg.norm(demand))
    if magnitude < 1e-9:
        return AttitudeReference(thrust=0.0, roll=0.0, pitch=0.0)

    z_body = demand / magnitude

    # A unit of thrust produces `M_f + e_z` of net force, not `e_z`, so the
    # commanded thrust is scaled by how much of it survives the reaction.
    direction = coupling.M_f + np.array([0.0, 0.0, 1.0])
    gain = float(np.linalg.norm(direction))
    f_z = magnitude / gain if gain > 1e-9 else magnitude

    sin_yaw, cos_yaw = np.sin(yaw_reference), np.cos(yaw_reference)
    roll = float(np.arcsin(np.clip(z_body[0] * sin_yaw - z_body[1] * cos_yaw, -1.0, 1.0)))
    pitch = float(np.arctan2(z_body[0] * cos_yaw + z_body[1] * sin_yaw, z_body[2]))

    roll = float(np.clip(roll, -LIMITS.tilt, LIMITS.tilt))
    pitch = float(np.clip(pitch, -LIMITS.tilt, LIMITS.tilt))
    return AttitudeReference(thrust=max(f_z, 0.0), roll=roll, pitch=pitch)


# --------------------------------------------------------------------------
# Section III-B: tube-based LPV-MPC for the rotational dynamics
# --------------------------------------------------------------------------


class RotationalTubeMPC:
    """Tube-based LPV-MPC of Eq. (22)-(38), literal 3-state per the paper.

    State x_eta = [p, q, r]; the attitude itself is not in the model, matching
    Eq. (22)-(23) exactly. It is regulated one layer up: the caller converts an
    attitude error into a body-rate reference xr_eta = [pr, qr, rr] (Algorithm 1
    step 4, via Eq. 2) before calling this. A 6-state variant that folded the
    attitude kinematics into this same model was tried and measurably tracked
    tilt tighter -- see the note in `coupling.rotational_lpv` -- but is not what
    Eq. (22)-(23) print, so it is not what runs here.

    The feedback gain K_eta and the terminal cost are computed once, offline, at
    the hover operating point, exactly as Algorithm 1 prescribes. The LPV
    matrices themselves are rebuilt at every step from the measured scheduling
    signal rho = [p, q, r].
    """

    def __init__(self) -> None:
        self.horizon = CONTROL.N_eta
        self.dt = CONTROL.dt_eta
        self.Q = CONTROL.Q_eta()
        self.R = CONTROL.R_eta()

        # Offline design point: hover, arm at rest, no rotation.
        rest = CrossCoupling(
            M_d=np.zeros((3, 3)), M_c=np.zeros((3, 3)), M_s=np.zeros((3, 3)),
            M_l=np.zeros((3, 3)), tau_bar=np.zeros(3), M_f=np.zeros(3),
            f_bar=np.zeros(3))
        nominal, nominal_input = rotational_lpv(rest, np.zeros(3), self.dt)
        self.gain = lqr_gain(nominal, nominal_input, self.Q, self.R)
        self.P = riccati_terminal_cost(nominal, nominal_input, self.Q, self.R)
        self.tube = Tube(
            gain=self.gain,
            relative=LPV.relative,
            absolute=LPV.absolute_eta,
            state_margin=np.full(3, 0.5),
            input_margin=np.full(3, 1.0),
        )

        self.state_box = BoxConstraint.symmetric(np.full(3, LIMITS.body_rate))
        self.input_box = BoxConstraint.symmetric(LIMITS.torque_vector())
        self.nominal_state: np.ndarray | None = None
        self._previous_input = np.zeros(3)
        self.predicted_body_rate_dot = np.zeros(3)

        # Terminal set, Theorem 1, computed once at the design point -- which is
        # what the paper does: chapter 3 lists the RPI set as an offline
        # computation. Solving it per step was tried and costs about 6 ms of a
        # 25 ms tick, pushing the 95th percentile to 110 ms on `disturbance`,
        # which is past the control period. That is doing more than the paper
        # asks for and paying the loop's whole margin for it.
        #
        # Recorded rather than only used, because whether it is feasible is the
        # condition Proposition 1 turns on, and the 75 % tightening cap otherwise
        # hides an empty set behind visible saturation.
        self.terminal_rpi = self.tube.minimal_rpi(
            nominal, nominal_input, np.zeros(3), np.zeros(3), self.horizon
        )
        self.terminal_feasible = Tube.terminal_is_feasible(
            self.state_box, self.terminal_rpi
        )

    def __call__(
        self,
        body_rate: np.ndarray,
        reference: np.ndarray,
        coupling: CrossCoupling,
    ) -> np.ndarray:
        """Return the UAV torque vector [tau_x, tau_y, tau_z].

        `body_rate` is [p, q, r], the state this regulates. `reference` is the
        desired body rate at each prediction step, shape (N_eta, 3).

        The residual reaction torque tau_bar of Eq. (10) enters the model as part
        of the input, so it is subtracted from the optimal input to recover the
        torque the rotors must actually produce.
        """
        state = np.asarray(body_rate, dtype=float)
        reference = np.atleast_2d(reference)

        A, B = rotational_lpv(coupling, body_rate, self.dt)

        if self.nominal_state is None:
            # Same seeding as the manipulator: the rotational input is
            # tau + tau_bar, so the input that holds the current state is the
            # residual, not zero.
            self.nominal_state = state.copy()
            self._previous_input = coupling.tau_bar.copy()

        # The LPV input is u = tau + tau_bar, and only tau is actuated. The
        # admissible input set is therefore the actuator box *shifted by
        # tau_bar*, not a box centred on zero. With the residual reaction
        # reaching several times the rotor authority this distinction decides
        # whether the optimiser is asked for something the rotors can produce.
        limit = LIMITS.torque_vector()
        input_box = BoxConstraint(
            lower=coupling.tau_bar - limit,
            upper=coupling.tau_bar + limit,
        )
        # Same for the rotational loop. Its A varies far less, so the frozen
        # gain was nearly valid, but there is no reason to keep the exception.
        self.gain = lqr_gain(A, B, self.Q, self.R)
        self.tube.gain = self.gain
        reachable = self.tube.reachable_sets(
            A, B, state, self._previous_input, self.horizon
        )
        state_bounds, input_bounds = self.tube.tighten(
            self.state_box, input_box, reachable, self.terminal_rpi
        )
        self.nominal_state = self.tube.confine(
            state, self.nominal_state, reachable[1]
        )
        # A reference outside the state constraint set makes the tightened
        # problem chase a target it is not allowed to reach, and the optimiser
        # answers by sitting on the actuator limit.
        reference = np.clip(reference, self.state_box.lower, self.state_box.upper)

        # Theorem 1's terminal constraint, translated onto the terminal
        # reference. Recorded rather than only applied: an empty intersection is
        # exactly Proposition 1's infeasibility condition, and it is the thing the
        # 75 % tightening cap otherwise hides.
        self.terminal_applied = TERMINAL_SET and self.tube.apply_terminal_set(
            A, B, state_bounds, input_bounds, reference)

        nominal_input = solve_mpc(
            self.nominal_state,
            reference,
            [A] * self.horizon,
            [B] * self.horizon,
            self.Q,
            self.R,
            self.P,
            state_bounds,
            input_bounds,
        )[0]

        # Eq. (32): nominal command plus feedback on the tube error.
        error = state - self.nominal_state
        total_input = self.gain @ error + nominal_input

        # Saturate in actuator space, then carry the *realised* input back into
        # the nominal propagation of Eq. (31). Advancing the nominal system with
        # a command the rotors never produced is what lets the two models drift
        # apart until the tube error stops meaning anything.
        torque = np.clip(total_input - coupling.tau_bar, -limit, limit)
        realised = torque + coupling.tau_bar
        self.nominal_state = A @ self.nominal_state + B @ np.clip(
            nominal_input, input_box.lower, input_box.upper
        )
        self._previous_input = realised

        # The angular acceleration this command is about to produce. The
        # manipulator controller schedules on it, and measuring it instead means
        # reacting a sample late -- by which point the light third link has
        # already picked up speed it cannot be talked out of. Publishing the
        # planned value removes that lag, and it costs nothing: the rotational
        # loop has already solved for it.
        # Bounded by what the rotors can actually produce. This number leaves the
        # rotational loop and enters both the manipulator's model and the
        # decomposition, so an unbounded value becomes a force on the vehicle by
        # way of f_bar: it was seen at 7300 rad/s^2, which is 1300 N m about an
        # axis the rotors drive with 7. The bound is the plain one -- authority
        # over inertia -- and past it the LPV extrapolation is describing an
        # aircraft that is not there.
        reachable = LIMITS.torque_vector() / np.diag(UAV.inertia_tensor())
        self.predicted_body_rate_dot = np.clip(
            (A @ state + B @ realised - body_rate) / self.dt,
            -reachable, reachable,
        )
        return torque


# --------------------------------------------------------------------------
# Section III-C: tube-based LPV-MPC for the manipulator
# --------------------------------------------------------------------------


class ManipulatorTubeMPC:
    """Tube-based LPV-MPC of Eq. (39)-(48)."""

    def __init__(self) -> None:
        self.horizon = CONTROL.N_gamma
        self.dt = CONTROL.dt_gamma
        self.Q = CONTROL.Q_gamma()
        self.R = CONTROL.R_gamma()

        n = ARM.n_links
        # Offline design point: arm at rest, hovering vehicle. A double
        # integrator with the hover mass matrix is enough to fix K and P.
        rest = np.zeros(n)
        A0, B0, _ = manipulator_lpv(
            rest, rest, np.zeros(3), np.zeros(3), TOTAL_MASS * GRAVITY, self.dt
        )
        self.gain = lqr_gain(A0, B0, self.Q, self.R)
        # K is re-solved every step, against the A and B of the current
        # configuration. Freezing it at the design point does not work here.
        #
        # The tube propagates its half-widths through the entrywise magnitude of
        # the closed loop, |A + B K|, so that matrix has to be contractive at the
        # configuration it is actually applied to. With the arm hanging, the
        # design-point gain gives a row sum of 1.008 and the set stays put. With
        # the arm folded to [1.64, pi, pi] the same gain gives 39.44 -- three
        # million over five steps -- while the gain solved for that pose gives
        # 1.000. The reachable set then swallowed the whole input box, the
        # tightened set of Eq. (46) came out empty at the first horizon step, and
        # the solver returned its bound: the joints sat at [6, 6, 6] for good.
        #
        # Eq. (32) defines K as the LQR gain of the LPV system, which is
        # scheduled, so re-solving is what the paper asks for; the frozen gain
        # was the shortcut. solve_discrete_are on 6x6 costs about 1 ms against a
        # 25 ms period.

        self.P = np.zeros((2 * n, 2 * n))
        self.P[: n * 2, : n * 2] = riccati_terminal_cost(A0, B0, self.Q, self.R)
        self.tube = Tube(
            gain=self.gain,
            relative=LPV.relative,
            absolute=LPV.absolute_gamma,
            state_margin=np.concatenate([np.full(n, 0.4), np.full(n, 1.5)]),
            # A fraction of the authority, not an absolute torque. At 1.0 N m
            # against a 0.5 N m budget the margin was twice the entire input box
            # it was tightening, and the tightened set of Eq. (46) had nothing
            # left in it: on the rigid-base test the arm missed its reference by
            # 47 % of the amplitude at a 1.5 N m budget while never once reaching
            # the limit, which is the signature of a constraint set that closed
            # rather than an actuator that ran out.
            #
            # A robust margin is meant to reserve part of the authority for
            # rejecting what the model got wrong. Written as a constant it stops
            # being a fraction the moment the authority changes, and that is what
            # happened here.
            input_margin=np.full(n, 0.2 * LIMITS.joint_torque),
        )

        self.state_box = BoxConstraint.symmetric(
            np.concatenate(
                [LIMITS.joint_angle_vector(), np.full(n, LIMITS.joint_rate)]
            )
        )
        self.input_box = BoxConstraint.symmetric(np.full(n, LIMITS.joint_torque))
        # Same terminal set as the rotational loop, offline at the design point,
        # for the same reasons.
        self.terminal_rpi = self.tube.minimal_rpi(
            A0, B0, np.zeros(2 * n), np.zeros(n), self.horizon
        )
        self.terminal_feasible = Tube.terminal_is_feasible(
            self.state_box, self.terminal_rpi
        )
        self.nominal_state: np.ndarray | None = None
        self._previous_input = np.zeros(n)
        self._previous_torque = np.zeros(n)
        self.predicted_joint_acceleration = np.zeros(n)

    def __call__(
        self,
        joint_state: np.ndarray,
        reference: np.ndarray,
        body_rate: np.ndarray,
        body_rate_dot: np.ndarray,
        thrust: float,
    ) -> np.ndarray:
        """Return the joint torque vector."""
        n = ARM.n_links
        q, qd = joint_state[:n], joint_state[n:]
        A, B, residual = manipulator_lpv(
            q, qd, body_rate, body_rate_dot, thrust, self.dt
        )

        if self.nominal_state is None:
            # Seed the tube so that the first sample is already consistent.
            #
            # Copying the measured state alone leaves _previous_input at zero,
            # and zero is not the equilibrium: the LPV input of Eq. (40) is
            # tau + T, so holding still means u = 0 and tau = -T, not tau = 0.
            # Starting from zero the optimiser sees a step of the size of the
            # gravity term, and the very first K @ error came out at 24 N m
            # against a 6 N m limit. Two seconds of saturation followed, which
            # was long enough to drag the arm from its resting pose out to the
            # far end of its travel, and it never recovered.
            #
            # Seeding both halves -- nominal state at the measurement, previous
            # input at the value that holds it -- makes the first solve a
            # continuation rather than a step.
            self.nominal_state = joint_state.copy()
            self._previous_input = residual.copy()

        # Same shift as the rotational loop: the LPV input is tau + T and only
        # tau is actuated.
        input_box = BoxConstraint(
            lower=residual - LIMITS.joint_torque,
            upper=residual + LIMITS.joint_torque,
        )
        # Re-solve the tube gain at this configuration before propagating.
        self.gain = lqr_gain(A, B, self.Q, self.R)
        self.tube.gain = self.gain
        reachable = self.tube.reachable_sets(
            A, B, joint_state, self._previous_input, self.horizon
        )
        state_bounds, input_bounds = self.tube.tighten(
            self.state_box, input_box, reachable, self.terminal_rpi
        )
        self.nominal_state = self.tube.confine(
            joint_state, self.nominal_state, reachable[1]
        )
        reference = np.clip(reference, self.state_box.lower, self.state_box.upper)

        # Theorem 1's terminal constraint, translated onto the terminal
        # reference. Recorded rather than only applied: an empty intersection is
        # exactly Proposition 1's infeasibility condition, and it is the thing the
        # 75 % tightening cap otherwise hides.
        self.terminal_applied = TERMINAL_SET and self.tube.apply_terminal_set(
            A, B, state_bounds, input_bounds, reference)

        nominal_input = solve_mpc(
            self.nominal_state,
            reference,
            [A] * self.horizon,
            [B] * self.horizon,
            self.Q,
            self.R,
            self.P,
            state_bounds,
            input_bounds,
        )[0]

        error = joint_state - self.nominal_state
        total_input = self.gain @ error + nominal_input
        # Split of Eq. (32), exposed for diagnosis: the feedback part and the
        # optimiser part are meant to be comparable in size. If the nominal
        # state drifts away from the measurement the first term stops being a
        # tube correction and becomes the accumulated divergence.
        self.last_error = error.copy()
        self.last_feedback = self.gain @ error
        self.last_nominal_input = np.asarray(nominal_input).copy()
        self.last_input_box = (input_bounds[0].lower.copy(),
                               input_bounds[0].upper.copy())

        torque = np.clip(
            total_input - residual, -LIMITS.joint_torque, LIMITS.joint_torque
        )

        # Rate limit on the commanded torque, which is a different constraint
        # from the magnitude limit above and solves a different problem.
        #
        # The two ends of this system want opposite things from the magnitude.
        # Against a rigid base the arm tracks Section V's reference to 1% of its
        # amplitude with 6 N m and to 72% with 0.5, because holding 0.33 rad
        # costs 1.153 N m at the shoulder and it simply cannot be done for less.
        # Flying, the same 6 N m throws 49 N at a 54 N vehicle and the run leaves
        # within seconds, while 0.5 keeps it. No magnitude satisfies both.
        #
        # What actually hurts the vehicle is not the size of the torque but how
        # fast it arrives: these links buy about 90 rad/s^2 per N m, so a step
        # from zero to full authority moves the joint through its whole rate
        # range inside one control period and the reaction is a hammer blow. A
        # rate limit leaves the static authority the arm needs to hold a pose and
        # takes away the hammer. The reference itself only asks for 0.83 rad/s at
        # its steepest, so a limit that reaches full torque in a couple of tenths
        # of a second does not stand in the way of tracking.
        if LIMITS.joint_torque_rate > 0.0:
            step = LIMITS.joint_torque_rate * self.dt
            torque = np.clip(torque, self._previous_torque - step,
                             self._previous_torque + step)
        self._previous_torque = torque.copy()

        self.nominal_state = A @ self.nominal_state + B @ np.clip(
            nominal_input, input_box.lower, input_box.upper
        )
        self._previous_input = torque + residual

        # The joint acceleration this command implies, taken from the model
        # rather than from the plant.
        #
        # This matters more than it looks. Eq. (10) makes tau_bar a function of
        # Theta_ddot, and the coefficient is the arm's inertia, so tau_bar is
        # large and responds instantly to joint acceleration. Feeding back the
        # *measured* Theta_ddot therefore closes an algebraic loop through one
        # sample of delay: the commanded torque sets the acceleration, the
        # acceleration sets tau_bar, tau_bar shifts the admissible input set, and
        # the next sample answers with the opposite sign. The loop gain exceeds
        # one and the result is a full-scale reversal every sample, with tau_bar
        # swinging tens of N m while the joints have barely moved.
        #
        # The optimiser's own predicted acceleration carries the same
        # information without the delay, and is smooth because it comes from the
        # solved trajectory instead of from a differentiated plant response.
        #
        # Bounded, because this number leaves the manipulator and lands in the
        # vehicle's force demand through f_bar. Unbounded it reached values worth
        # 29 kN of reaction on a 54 N aircraft -- a prediction of something the
        # plant cannot do, since the joint rate limit caps how much velocity a
        # joint can gain in one period at LIMITS.joint_rate / dt. Past that the
        # LPV extrapolation is describing a machine that does not exist, and the
        # attitude loop has no way to know.
        reachable = max_joint_acceleration(q, qd, body_rate, body_rate_dot, thrust)
        self.predicted_joint_acceleration = np.clip(
            ((A @ joint_state + B @ self._previous_input)[n:] - qd) / self.dt,
            -reachable, reachable,
        )
        return torque


# --------------------------------------------------------------------------
# Section V: the ERTF baseline
# --------------------------------------------------------------------------


@dataclass
class PID:
    """Minimal discrete PID used by the baseline."""

    kp: float
    ki: float
    kd: float
    limit: float
    _integral: np.ndarray = field(default=None, repr=False)
    _previous: np.ndarray = field(default=None, repr=False)

    def __call__(self, error: np.ndarray, dt: float) -> np.ndarray:
        if self._integral is None:
            self._integral = np.zeros_like(error)
            self._previous = np.zeros_like(error)
        self._integral = np.clip(self._integral + error * dt, -self.limit, self.limit)
        derivative = (error - self._previous) / dt
        self._previous = error.copy()
        command = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(command, -self.limit, self.limit)


class ErtfBaseline:
    """Estimating-Reaction-Torque-and-Force baseline.

    The comparison the paper draws in Section V: the reaction torque and force
    are measured with the recursive Newton-Euler equations and then cancelled by
    a feedforward term, with PID closing the attitude and joint loops.

    The structural weakness the paper identifies is reproduced faithfully. The
    estimate uses the *previous* step's angular acceleration, because the
    current one is not available until the torque has been chosen; so the
    M_d @ omega_dot coupling of Eq. (10) is always one step stale. When the arm
    is slow that lag is irrelevant. When the arm accelerates it is not, and the
    attitude loop is left chasing a term it never sees in time.
    """

    def __init__(self) -> None:
        # The limit is an angular acceleration, because that is what this PID
        # produces: its output is multiplied by the inertia tensor below to become
        # a torque. It was LIMITS.torque -- a torque limit of 7.0 applied to an
        # acceleration signal -- so I @ feedback could never exceed 0.18 * 7 =
        # 1.26 N m, 18 % of the authority the rotors actually have. Measured, the
        # PID pinned at its bound on the first call and stayed there, and the run
        # fell out of the air at 13.2 s. With the units right it flies the full
        # 48 s.
        #
        # This matters beyond tidiness. The paper's claim is that ERTF tracks
        # worse than LPV-MPC, 20-30 % against 2-8 %, not that it falls out of the
        # sky. A baseline crippled by a units error in our own code is a strawman,
        # and any comparison built on it proves nothing.
        acceleration_limit = float(
            LIMITS.torque / min(np.diag(UAV.inertia_tensor())))
        self.attitude_pid = PID(kp=float(os.environ.get("UAM_ERTF_KP", 30.0)), ki=0.6, kd=float(os.environ.get("UAM_ERTF_KD", 0.05)), limit=acceleration_limit)
        self.joint_pid = PID(kp=14.0, ki=1.2, kd=2.4, limit=LIMITS.joint_torque)
        self.previous_body_rate_dot = np.zeros(3)

    def rotational(
        self,
        body_rate: np.ndarray,
        reference: np.ndarray,
        joint_state: np.ndarray,
        joint_acceleration: np.ndarray,
        thrust: float,
        dt: float,
    ) -> np.ndarray:
        n = ARM.n_links
        q, qd = joint_state[:n], joint_state[n:]
        reaction = newton_euler(
            q,
            qd,
            joint_acceleration,
            body_rate,
            self.previous_body_rate_dot,
            thrust,
        )
        _, reaction_torque = reaction.reaction_on_uav()

        gyroscopic = np.cross(body_rate, UAV.inertia_tensor() @ body_rate)
        feedback = self.attitude_pid(reference - body_rate, dt)
        if os.environ.get("UAM_ERTF_NO_FF") == "1":
            reaction_torque = np.zeros(3)
        torque = UAV.inertia_tensor() @ feedback + gyroscopic - reaction_torque
        limit = LIMITS.torque_vector()
        return np.clip(torque, -limit, limit)

    def manipulator(
        self,
        joint_state: np.ndarray,
        reference: np.ndarray,
        body_rate: np.ndarray,
        thrust: float,
        dt: float,
    ) -> np.ndarray:
        n = ARM.n_links
        q, qd = joint_state[:n], joint_state[n:]
        desired_acceleration = self.joint_pid(reference[:n] - q, dt) - 2.0 * qd
        torque = newton_euler(
            q,
            qd,
            desired_acceleration,
            body_rate,
            self.previous_body_rate_dot,
            thrust,
        ).joint_torques
        return np.clip(torque, -LIMITS.joint_torque, LIMITS.joint_torque)
