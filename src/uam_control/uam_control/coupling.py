"""Cross-coupled decomposition and the LPV models built from it.

This is the contribution of the reference paper. Instead of estimating the arm's
reaction as a lumped disturbance (the ERTF baseline), the reaction torque is
factorised in terms of the UAV's own states, Eq. (10):

    tau_arm_rea = M_d @ omega_dot
                + M_c @ [p q, q r, p r]
                + M_s @ [p^2, q^2, r^2]
                + M_l @ [p, q, r]
                + tau_bar

and the reaction force is factorised in terms of the thrust, Eq. (11):

    f_arm_rea = M_f * f_z + f_bar

How the matrices are obtained
-----------------------------
The paper derives them symbolically with SymPy and reports that the result is a
large expression in sines and cosines of the joint angles. That derivation is
not reproduced here; instead the same matrices are recovered numerically, which
is exact rather than approximate.

The reason it is exact: tracing the recursions of Eq. (6)-(7), the base torque
tau_1 is affine in omega_dot, exactly quadratic in omega (no omega_dot-omega
cross terms appear), and affine in f_z. A function with that structure is fully
determined by a finite number of evaluations, so probing Newton-Euler with unit
basis vectors reads the matrices off directly. Fifteen evaluations per control
step recover the whole decomposition to machine precision.

Discretisation follows Eq. (24) and Eq. (41): forward Euler, so
A = I + dt * P_1 and B = dt * P_2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.linalg import expm

from .newton_euler import newton_euler
from .params import ARM, UAV, ArmParams, UavParams

# Closed-form dynamics, if they have been generated. Everything below has a
# probing fallback that computes the same quantities from `newton_euler`, so the
# module works without them -- it just costs 43 dynamics evaluations per control
# tick instead of ten expressions, which measured 33 ms of a 43 ms tick.
#
# Regenerate with `python3 derive_dynamics.py` after changing anything in
# `params.py` that the arm's dynamics depend on, then run `test_generated.py`.
# The generated file is only trusted because that test compares it against the
# probing on random states.
try:
    from . import generated_dynamics as _GENERATED
except ImportError:  # pragma: no cover - the fallback path is the old code
    _GENERATED = None

_EPS = 1e-6


def max_joint_acceleration(q, qd, omega, omega_dot, f_z, arm: ArmParams = ARM):
    """Largest joint acceleration the arm can physically produce, per joint.

    A bound on Theta_ddot is needed wherever that quantity leaves the
    manipulator and becomes a force on the vehicle through f_bar, because neither
    the optimiser's extrapolation nor the free response is naturally bounded and
    f_bar has been seen at 131 N against a 54 N aircraft.

    The bound used before was `joint_rate / dt`, and it was wrong in kind rather
    than in value: a physical acceleration limit cannot depend on the control
    period. Moving from 40 Hz to 100 Hz silently multiplied it by 2.5, which is
    how a bound meant to keep f_bar honest ended up permitting 1000 rad/s^2.

    This is the real thing: with M q_ddot = tau - bias, no joint can exceed
    |M^-1| (|tau_max| + |bias|). It costs one closed-form M and one inverse.
    """
    q = np.asarray(q, dtype=float)
    if _GENERATED is not None:
        a = (*q, *np.asarray(qd, dtype=float), *np.zeros(arm.n_links),
             *np.asarray(omega, dtype=float), *np.asarray(omega_dot, dtype=float),
             float(f_z))
        M = _GENERATED.get_M(*a)
        bias = _GENERATED.get_tau_arm(*a).ravel()
    else:
        from .newton_euler import mass_matrix
        M = mass_matrix(q, arm=arm)
        bias = newton_euler(q, qd, np.zeros(arm.n_links), omega, omega_dot, f_z,
                            arm=arm).joint_torques
    from .params import LIMITS
    return np.abs(np.linalg.inv(M)) @ (
        np.full(arm.n_links, LIMITS.joint_torque) + np.abs(bias))


def zero_order_hold(A_c: np.ndarray, B_c: np.ndarray, dt: float):
    """Exact discretisation of xdot = A_c x + B_c u over one sample.

    Eq. (24) and Eq. (41) discretise with forward Euler -- the `1 / dt + phi`
    structure of those matrices is exactly `I + dt * P`. For the rotational
    model that is harmless, but for the manipulator it is not.

    The arm of Table I has a badly spread mass matrix: its eigenvalues run from
    6.8e-4 to 2.1e-1 kg m^2, a ratio of about 310. The stiff mode accelerates at
    several thousand rad/s^2 under the available joint torque, so over the
    dt_gamma = 0.025 s of Table II forward Euler is not merely inaccurate, it is
    unstable on its own: the discrete open-loop A has a spectral radius of 1.095,
    growing 9.5 % per sample with no input at all. The optimiser inherits that
    and commands full-scale torque reversals on consecutive samples.

    Zero-order hold removes the problem without touching the model or the sample
    rate, because it is the exact solution of the same continuous LPV system
    over one sample rather than its first-order truncation. The spectral radius
    drops back to essentially 1 and the arm's fast mode is represented properly.
    """
    n, m = A_c.shape[0], B_c.shape[1]
    block = np.zeros((n + m, n + m))
    block[:n, :n] = A_c
    block[:n, n:] = B_c
    discrete = expm(block * dt)
    return discrete[:n, :n], discrete[:n, n:]


@dataclass
class CrossCoupling:
    """The five reaction-torque terms of Eq. (10) plus the force terms of (11).

    All quantities are expressed in the UAV body frame and carry the sign of the
    reaction *acting on the UAV*, i.e. the negation of the inward-pass result.
    """

    M_d: np.ndarray  # (3, 3), multiplies omega_dot
    M_c: np.ndarray  # (3, 3), multiplies [p q, q r, p r]
    M_s: np.ndarray  # (3, 3), multiplies [p^2, q^2, r^2]
    M_l: np.ndarray  # (3, 3), multiplies [p, q, r]
    tau_bar: np.ndarray  # (3,), residual
    M_f: np.ndarray  # (3,), multiplies f_z
    f_bar: np.ndarray  # (3,), residual

    def torque(self, omega: np.ndarray, omega_dot: np.ndarray) -> np.ndarray:
        """Reconstruct the reaction torque from the decomposition."""
        p, q, r = omega
        return (
            self.M_d @ omega_dot
            + self.M_c @ np.array([p * q, q * r, p * r])
            + self.M_s @ np.array([p * p, q * q, r * r])
            + self.M_l @ omega
            + self.tau_bar
        )

    def force(self, f_z: float) -> np.ndarray:
        """Reconstruct the reaction force from the decomposition."""
        return self.M_f * f_z + self.f_bar


def decompose(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    omega: np.ndarray,
    omega_dot: np.ndarray,
    f_z: float,
    arm: ArmParams = ARM,
) -> CrossCoupling:
    """Recover Eq. (10) and Eq. (11) at the current manipulator state.

    `omega` and `omega_dot` only affect the force decomposition; the torque
    matrices depend on the manipulator state alone, as the paper states.

    Note on `tau_bar`: the paper describes the residual as depending solely on
    (Theta, Theta_dot, Theta_ddot). It also depends on f_z, because the base
    acceleration of Eq. (6c) propagates through the inertial forces into
    tau_1. That dependence is absorbed into `tau_bar`, evaluated at the
    measured f_z, which is exact at the current step.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)
    zeros = np.zeros(3)

    if _GENERATED is not None:
        # Closed form. The probing below is what this replaces: fifteen full
        # Newton-Euler passes to recover matrices that are one expression each.
        # `derive_dynamics.py` produced them by running this project's own
        # recursion symbolically, and `test_generated.py` checks all seven
        # against the probing on 200 random states -- agreement is 1e-15.
        #
        # The sign is here and nowhere else: the inward pass returns what the
        # base exerts on link 1, and Eq. (10)-(11) want the reaction on the
        # vehicle, which is its opposite.
        a = (*q, *qd, *qdd, *np.asarray(omega, dtype=float),
             *np.asarray(omega_dot, dtype=float), float(f_z))
        return CrossCoupling(
            M_d=-_GENERATED.get_M_d(*a),
            M_c=-_GENERATED.get_M_c(*a),
            M_s=-_GENERATED.get_M_s(*a),
            M_l=-_GENERATED.get_M_l(*a),
            tau_bar=-_GENERATED.get_tau_bar(*a).ravel(),
            M_f=-_GENERATED.get_M_f(*a).ravel(),
            f_bar=-_GENERATED.get_f_bar(*a).ravel(),
        )

    def base_torque(w: np.ndarray, wd: np.ndarray, thrust: float) -> np.ndarray:
        # Negated: the inward pass returns what the base exerts on link 1, the
        # reaction on the UAV is its opposite.
        return -newton_euler(q, qd, qdd, w, wd, thrust, arm=arm).base_torque

    def base_force(w: np.ndarray, wd: np.ndarray, thrust: float) -> np.ndarray:
        return -newton_euler(q, qd, qdd, w, wd, thrust, arm=arm).base_force

    # Residual: no UAV rotation at all.
    tau_bar = base_torque(zeros, zeros, f_z)

    # Angular-acceleration coupling: one probe per axis.
    M_d = np.column_stack(
        [base_torque(zeros, np.eye(3)[j], f_z) - tau_bar for j in range(3)]
    )

    # Linear and squared rate coupling: probe each axis at +1 and -1 so the odd
    # and even parts separate.
    M_l_cols, M_s_cols = [], []
    for j in range(3):
        e = np.eye(3)[j]
        plus = base_torque(e, zeros, f_z) - tau_bar
        minus = base_torque(-e, zeros, f_z) - tau_bar
        M_l_cols.append(0.5 * (plus - minus))
        M_s_cols.append(0.5 * (plus + minus))
    M_l = np.column_stack(M_l_cols)
    M_s = np.column_stack(M_s_cols)

    # Product coupling: excite two axes at once and subtract the parts already
    # accounted for. Column order matches the monomial order [p q, q r, p r].
    M_c_cols = []
    for a, b in ((0, 1), (1, 2), (0, 2)):
        e = np.eye(3)[a] + np.eye(3)[b]
        mixed = base_torque(e, zeros, f_z) - tau_bar
        M_c_cols.append(
            mixed - M_l[:, a] - M_l[:, b] - M_s[:, a] - M_s[:, b]
        )
    M_c = np.column_stack(M_c_cols)

    # Force decomposition at the current rotational state.
    f_bar = base_force(omega, omega_dot, 0.0)
    M_f = base_force(omega, omega_dot, 1.0) - f_bar

    return CrossCoupling(
        M_d=M_d, M_c=M_c, M_s=M_s, M_l=M_l, tau_bar=tau_bar, M_f=M_f, f_bar=f_bar
    )


def uav_gyroscopic_matrices(uav: UavParams = UAV) -> tuple[np.ndarray, np.ndarray]:
    """The M_c and M_s of Eq. (5b) that come from the UAV alone.

    Expanding Eq. (3b) gives I_t omega_dot = -omega x I_t omega + tau. With a
    diagonal inertia tensor,

        -omega x I_t omega = [(Iyy - Izz) q r, (Izz - Ixx) p r, (Ixx - Iyy) p q]

    which is purely a product term, so M_s is zero. Written against the monomial
    ordering [p q, q r, p r] this gives the M_c below.
    """
    ixx, iyy, izz = uav.inertia_xx, uav.inertia_yy, uav.inertia_zz
    M_c = np.array(
        [
            [0.0, iyy - izz, 0.0],
            [0.0, 0.0, izz - ixx],
            [ixx - iyy, 0.0, 0.0],
        ]
    )
    return M_c, np.zeros((3, 3))


def rotational_lpv(
    coupling: CrossCoupling,
    rho: np.ndarray,
    dt: float,
    uav: UavParams = UAV,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time LPV model of the UAV's rotational dynamics, Eq. (24)-(25).

    `rho` is the scheduling signal [p, q, r] taken from the current measurement.
    The quadratic rate terms are turned into a state matrix by folding one
    factor of the rate into the scheduled parameter, which is what the two
    diagonal matrices of Eq. (25) accomplish:

        M_c @ diag(rho_q, rho_r, rho_p) @ [p, q, r] = M_c @ [p q, q r, p r]
        M_s @ diag(rho_p, rho_q, rho_r) @ [p, q, r] = M_s @ [p^2, q^2, r^2]

    Returns (A_eta, B_eta) for the state [p, q, r] and the input
    [tau_x + tau_bar_x, tau_y + tau_bar_y, tau_z + tau_bar_z].
    """
    p, q, r = rho[:3]
    M_c_uav, M_s_uav = uav_gyroscopic_matrices(uav)

    effective_inertia = uav.inertia_tensor() - coupling.M_d
    inverse = np.linalg.inv(effective_inertia)

    P1 = inverse @ (
        (M_c_uav + coupling.M_c) @ np.diag([q, r, p])
        + (M_s_uav + coupling.M_s) @ np.diag([p, q, r])
        + coupling.M_l
    )
    P2 = inverse

    A_rate, B_rate = zero_order_hold(P1, P2, dt)

    # Literal 3-state, [p, q, r], as Eq. (22)-(23) print it. A 6-state variant
    # (attitude folded into this same model) was tried to put the tilt itself
    # inside the tube's constraint set: state-only tilt capping let the achieved
    # pitch overshoot the command by about 70 % on the `fast` scenario, since
    # nothing bounded the tilt that was actually *reached*. Reverted for
    # paper-fidelity; if that overshoot resurfaces, that is where to look.
    return A_rate, B_rate


def manipulator_lpv(
    q: np.ndarray,
    qd: np.ndarray,
    omega: np.ndarray,
    omega_dot: np.ndarray,
    f_z: float,
    dt: float,
    arm: ArmParams = ARM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete-time LPV model of the manipulator, Eq. (39)-(42).

    The paper rewrites the gravity-like vector as G = Q @ Theta + T by replacing
    sin(q_i) with sinc(q_i) * q_i, then sets

        P_1 = -M^-1 C,  P_2 = -M^-1 Q,  P_3 = M^-1.

    Two departures from the printed equations, both deliberate:

    1. Eq. (41) places the P_1 block in the column that multiplies Theta and the
       P_2 block in the column that multiplies Theta_dot, which is the opposite
       of what Eq. (42) defines (C multiplies velocity, Q multiplies position).
       The physically consistent placement is used here.
    2. Rather than the sinc substitution, Q is taken as the Jacobian of G with
       respect to Theta at the current configuration and T absorbs the exact
       remainder. This is the same factorisation the sinc trick performs for
       pure sine terms, generalises to the full coupled expression, and leaves
       no approximation error at the current operating point. Whatever drift it
       produces over the horizon is exactly the parameter variation the
       tube-based controller is designed to absorb.

    C is recovered from the fact that the velocity-dependent part of the
    Newton-Euler bias is a quadratic form in Theta_dot: if h(qd) is that part,
    then C = (1/2) * dh/d(qd) satisfies C @ qd = h(qd) identically.

    Returns (A_gamma, B_gamma, T) with state [Theta, Theta_dot] and input
    [tau_1 + T_1, tau_2 + T_2, tau_3 + T_3].
    """
    n = arm.n_links
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    zeros_n = np.zeros(n)

    if _GENERATED is not None:
        # Closed form for M, C, Q and T, generated from this same definition of
        # C -- the one that satisfies C qd = bias - G and so carries the Coriolis
        # coupling with the vehicle's rotation and the joint damping along with
        # the centrifugal terms. The probing below computes the identical
        # quantities with 28 Newton-Euler passes; this costs four expressions.
        a = (*q, *qd, *zeros_n, *np.asarray(omega, dtype=float),
             *np.asarray(omega_dot, dtype=float), float(f_z))
        M_inv = np.linalg.inv(_GENERATED.get_M(*a))
        C = _GENERATED.get_C_lpv(*a)
        Q_mat = _GENERATED.get_Q_lpv(*a)
        T = _GENERATED.get_T_lpv(*a).ravel()
        A_c = np.block([[np.zeros((n, n)), np.eye(n)],
                        [-M_inv @ Q_mat, -M_inv @ C]])
        B_c = np.vstack([np.zeros((n, n)), M_inv])
        A, B = zero_order_hold(A_c, B_c, dt)
        return A, B, T

    def bias(joint_angles: np.ndarray, joint_rates: np.ndarray) -> np.ndarray:
        return newton_euler(
            joint_angles, joint_rates, zeros_n, omega, omega_dot, f_z, arm=arm
        ).joint_torques

    # The gravity-like bias, computed once. It used to be evaluated four times
    # with identical arguments -- three inside the mass-matrix comprehension and
    # once more below as G -- and each evaluation is a full Newton-Euler pass.
    # This function makes 31 of them per call at 0.207 ms each, against a control
    # period the node was already overrunning by a factor of two.
    G = bias(q, zeros_n)

    # Mass matrix: probe with unit joint accelerations. Newton-Euler is linear in
    # Theta_ddot, so column j is the response to a unit acceleration on joint j
    # with the bias removed.
    M = np.column_stack(
        [
            newton_euler(
                q, zeros_n, np.eye(n)[j], omega, omega_dot, f_z, arm=arm
            ).joint_torques
            - G
            for j in range(n)
        ]
    )
    M_inv = np.linalg.inv(M)

    def velocity_part(joint_rates: np.ndarray) -> np.ndarray:
        return bias(q, joint_rates) - G

    # The velocity term is not a pure quadratic: whenever the UAV is rotating,
    # the Coriolis coupling between omega and Theta_dot contributes terms that
    # are linear in Theta_dot. Split the two by parity before differentiating,
    # otherwise the half-gradient halves the linear contribution.
    def quadratic_part(joint_rates: np.ndarray) -> np.ndarray:
        return 0.5 * (velocity_part(joint_rates) + velocity_part(-joint_rates))

    # Linear block: exact, one probe per basis direction.
    C_linear = np.column_stack(
        [
            0.5 * (velocity_part(np.eye(n)[j]) - velocity_part(-np.eye(n)[j]))
            for j in range(n)
        ]
    )
    # Quadratic block: half the gradient, exact because central differences are
    # exact on a quadratic.
    C_quadratic = np.column_stack(
        [
            (quadratic_part(qd + _EPS * np.eye(n)[j]) - quadratic_part(qd - _EPS * np.eye(n)[j]))
            / (4.0 * _EPS)
            for j in range(n)
        ]
    )
    C = C_linear + C_quadratic

    # Q = dG/dTheta, with T taking up the exact remainder. Eq. (39) carries the
    # residual as +T on the input side, so the split is G = Q @ Theta - T.
    Q_mat = np.column_stack(
        [
            (bias(q + _EPS * np.eye(n)[j], zeros_n) - bias(q - _EPS * np.eye(n)[j], zeros_n))
            / (2.0 * _EPS)
            for j in range(n)
        ]
    )
    T = Q_mat @ q - G

    P1 = -M_inv @ C  # multiplies Theta_dot
    P2 = -M_inv @ Q_mat  # multiplies Theta
    P3 = M_inv

    A_c = np.block([[np.zeros((n, n)), np.eye(n)], [P2, P1]])
    B_c = np.vstack([np.zeros((n, n)), P3])
    A, B = zero_order_hold(A_c, B_c, dt)
    return A, B, T
