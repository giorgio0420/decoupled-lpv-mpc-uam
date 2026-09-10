"""Checks on the dynamics and the decomposition.

These are the properties the rest of the package leans on. Each one caught a
real defect while the package was being written, so they are worth keeping.
"""

import numpy as np
import pytest

from uam_control.controllers import attitude_reference
from uam_control.coupling import decompose, manipulator_lpv, rotational_lpv
from uam_control.newton_euler import joint_accelerations, mass_matrix, newton_euler
from uam_control.params import ARM, GRAVITY, TOTAL_MASS, UAV
from uam_control.plant import UamState, solve_accelerations

RNG = np.random.default_rng(0)


def random_state():
    return dict(
        q=RNG.uniform(-2, 2, 3),
        qd=RNG.uniform(-3, 3, 3),
        qdd=RNG.uniform(-5, 5, 3),
        omega=RNG.uniform(-2, 2, 3),
        omega_dot=RNG.uniform(-4, 4, 3),
        f_z=RNG.uniform(10, 100),
    )


@pytest.mark.parametrize("trial", range(25))
def test_decomposition_reconstructs_the_reaction(trial):
    """Eq. (10) and (11) must reproduce Newton-Euler exactly, not approximately.

    The decomposition is obtained by probing rather than by symbolic algebra, so
    this is the check that the probing captures every term. It holds to machine
    precision because tau_1 really is affine in omega_dot, exactly quadratic in
    omega, and affine in f_z.
    """
    s = random_state()
    coupling = decompose(**s)
    force, torque = newton_euler(**s).reaction_on_uav()

    assert coupling.torque(s["omega"], s["omega_dot"]) == pytest.approx(torque, abs=1e-9)
    assert coupling.force(s["f_z"]) == pytest.approx(force, abs=1e-9)


def test_forward_dynamics_inverts_newton_euler():
    s = random_state()
    tau = newton_euler(**s).joint_torques
    recovered = joint_accelerations(
        s["q"], s["qd"], tau, s["omega"], s["omega_dot"], s["f_z"]
    )
    assert recovered == pytest.approx(s["qdd"], abs=1e-9)


def test_mass_matrix_is_symmetric_positive_definite():
    for _ in range(10):
        M = mass_matrix(RNG.uniform(-2, 2, 3))
        assert M == pytest.approx(M.T, abs=1e-9)
        assert np.linalg.eigvalsh(M).min() > 0.0


def test_arm_adds_inertia_rather_than_removing_it():
    """(I_t - M_d) must exceed I_t.

    A sign slip in the reaction would show up here as an effective inertia
    *smaller* than the bare airframe, which is physically backwards: an arm
    bolted to the vehicle can only make it harder to rotate.
    """
    coupling = decompose(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        TOTAL_MASS * GRAVITY,
    )
    effective = UAV.inertia_tensor() - coupling.M_d
    assert np.all(np.diag(effective) >= np.diag(UAV.inertia_tensor()) - 1e-12)
    assert np.linalg.eigvalsh(effective).min() > 0.0


def test_hover_thrust_equals_total_weight():
    """The two masses of the paper must not be interchanged.

    Eq. (12a) divides by the airframe mass because the arm already enters
    through the reaction force, while Eq. (6c) uses the combined mass because it
    describes the whole vehicle's acceleration. Getting this backwards puts the
    hover thrust about 10 % high, which this pins down.
    """
    coupling = decompose(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        TOTAL_MASS * GRAVITY,
    )
    result = attitude_reference(
        np.array([0.0, 0.0, UAV.mass * GRAVITY]), coupling, 0.0
    )
    assert result.thrust == pytest.approx(TOTAL_MASS * GRAVITY, rel=1e-6)
    assert result.roll == pytest.approx(0.0, abs=1e-9)
    assert result.pitch == pytest.approx(0.0, abs=1e-9)


def test_hovering_vehicle_stays_put():
    state = UamState.at_rest()
    accelerations = solve_accelerations(
        state, TOTAL_MASS * GRAVITY, np.zeros(3), np.zeros(ARM.n_links)
    )
    assert accelerations.linear == pytest.approx(np.zeros(3), abs=1e-9)
    assert accelerations.body_rate_dot == pytest.approx(np.zeros(3), abs=1e-9)
    assert accelerations.joint_dot == pytest.approx(np.zeros(3), abs=1e-9)


def test_rotational_lpv_matches_the_plant():
    """The controller's model and the truth model must agree to machine precision."""
    for _ in range(10):
        state = UamState.at_rest()
        state.joints = RNG.uniform(-0.4, 0.4, 3)
        state.joint_rates = RNG.uniform(-1, 1, 3)
        state.body_rate = RNG.uniform(-0.4, 0.4, 3)
        torque = RNG.uniform(-1.5, 1.5, 3)

        thrust = TOTAL_MASS * GRAVITY
        exact = solve_accelerations(state, thrust, torque, np.zeros(3))
        coupling = decompose(
            state.joints, state.joint_rates, exact.joint_dot,
            state.body_rate, exact.body_rate_dot, thrust,
        )
        # Comparing a one-sample zero-order-hold step against the instantaneous
        # derivative leaves an O(dt) remainder, so the tolerance has to sit
        # above dt rather than at machine precision.
        dt = 1e-4
        # Six states, [phi, theta, psi, p, q, r]. Only the rate block is compared:
        # the attitude rows are the kinematics eta_dot = omega, which is a
        # modelling choice rather than a claim about the plant, and comparing them
        # against the truth model would be checking the small-angle approximation
        # instead of the dynamics.
        A, B = rotational_lpv(coupling, state.body_rate, dt)
        full = np.concatenate([state.attitude, state.body_rate])
        predicted = (
            (A @ full + B @ (torque + coupling.tau_bar))[3:] - state.body_rate
        ) / dt
        assert predicted == pytest.approx(exact.body_rate_dot, rel=1e-2, abs=1e-4)


def test_manipulator_lpv_converges_to_the_true_acceleration():
    """Zero-order hold must reproduce the continuous dynamics as dt shrinks."""
    s = random_state()
    tau = newton_euler(**s).joint_torques
    errors = []
    for dt in (1e-3, 1e-4):
        A, B, residual = manipulator_lpv(
            s["q"], s["qd"], s["omega"], s["omega_dot"], s["f_z"], dt
        )
        state = np.concatenate([s["q"], s["qd"]])
        implied = ((A @ state + B @ (tau + residual))[3:] - s["qd"]) / dt
        errors.append(np.max(np.abs(implied - s["qdd"])))
    assert errors[1] < errors[0]
    assert errors[1] < 1e-2 * max(1.0, np.max(np.abs(s["qdd"])))


def test_manipulator_discretisation_is_not_self_exciting():
    """Forward Euler is unstable here; the discretisation must not be.

    With the arm of Table I and dt_gamma = 0.025 s the forward-Euler model of
    Eq. (41) has a spectral radius of 1.095 with no input at all. Zero-order
    hold keeps it on the unit circle, where an undamped pendulum belongs.
    """
    A, _, _ = manipulator_lpv(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        TOTAL_MASS * GRAVITY, 0.025,
    )
    assert np.abs(np.linalg.eigvals(A)).max() < 1.0 + 1e-6
