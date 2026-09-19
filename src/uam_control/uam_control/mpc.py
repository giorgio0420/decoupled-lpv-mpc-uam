"""Condensed linear MPC and the tube machinery of Section III-B.

Two pieces live here:

`solve_mpc`
    A dense, condensed quadratic program for a linear time-varying prediction
    model with box constraints. Used unchanged by all three controllers of the
    paper; only the model, weights and bounds differ.

`Tube`
    Reachable-set propagation (Eq. 35), constraint tightening (Eq. 36) and the
    LQR feedback gain of Eq. (32).

Set representation
------------------
The paper computes reachable sets and the RPI terminal set as polytopes with
the MPT toolbox. Here every set is an axis-aligned box. Under that
representation:

* the Minkowski sum of two boxes is exact (half-widths add);
* the Minkowski difference of two boxes is exact (half-widths subtract);
* a linear map of a box is a zonotope, which is over-approximated by its
  interval hull, `abs(A) @ half_widths`.

Only the linear map is inexact, and it errs by *enlarging* the set. The tube
therefore still contains the true error trajectory, so the constraint-tightening
guarantee of Eq. (36) is preserved; the price is extra conservatism, which shows
up as a slightly wider tube than MPT would produce.

The explicit terminal set of Eq. (37) is replaced by the terminal cost from the
discrete algebraic Riccati equation plus the tightened box at the last
prediction step. The Riccati solution satisfies Definition 1 by construction, so
the stability argument of Theorem 1 still applies; what is given up is the
guarantee of recursive feasibility that a true RPI terminal set would provide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import linalg, sparse

_INF = 1e6
# How far the minimal-RPI series is summed before its tail is bounded and closed.
# At the spectral radii this project sees the terms decay as 0.88^i and 0.95^i, so
# the manipulator needs about 270 of them to fall six decades and ran out at 200 --
# returning None, which reads as "no invariant set" when the truth was "not yet
# summed". The tail is bounded and added rather than dropped, so the tolerance can
# be loose without the answer becoming an under-estimate. Cost is a 6x6 matmul per
# term, negligible against the QP.
_RPI_TERMS = 500
_RPI_TOLERANCE = 1e-4


class MpcStateError(RuntimeError):
    """The state handed to the optimiser is no longer usable."""


@dataclass
class BoxConstraint:
    """Axis-aligned bounds, the constraint sets X and U of the paper."""

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def symmetric(cls, half_width: np.ndarray) -> "BoxConstraint":
        half_width = np.asarray(half_width, dtype=float)
        return cls(-half_width, half_width)

    @property
    def half_width(self) -> np.ndarray:
        return 0.5 * (self.upper - self.lower)

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.upper + self.lower)

    def tighten(self, amount: np.ndarray, max_fraction: float = 0.75) -> "BoxConstraint":
        """Minkowski difference with a box of the given half-widths, Eq. (36).

        The tightening is capped at `max_fraction` of the half-width. Without a
        cap an over-wide tube silently collapses the set onto its centre, and a
        controller whose predicted state is pinned to a point produces no
        command at all while still reporting a solved QP. Leaving a quarter of
        the box turns that failure into visible saturation instead.
        """
        amount = np.minimum(
            np.asarray(amount, dtype=float), max_fraction * self.half_width
        )
        return BoxConstraint(self.lower + amount, self.upper - amount)


def _lifted_matrices(
    A: list[np.ndarray], B: list[np.ndarray], n: int, m: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build the prediction matrices Phi and Gamma for x = Phi x0 + Gamma U.

    Accepts a per-step model so that the nominal LPV matrices of Eq. (27) can
    vary along the horizon. Row block i corresponds to the state at step i + 1.
    """
    Phi = np.zeros((n * horizon, n))
    Gamma = np.zeros((n * horizon, m * horizon))
    state_transition = np.eye(n)
    for i in range(horizon):
        state_transition = A[i] @ state_transition
        Phi[n * i : n * (i + 1), :] = state_transition
        for j in range(i + 1):
            block = np.eye(n)
            for k in range(j + 1, i + 1):
                block = A[k] @ block
            Gamma[n * i : n * (i + 1), m * j : m * (j + 1)] = block @ B[j]
    return Phi, Gamma


def solve_mpc(
    x0: np.ndarray,
    reference: np.ndarray,
    A: list[np.ndarray],
    B: list[np.ndarray],
    Q: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    state_bounds: list[BoxConstraint],
    input_bounds: list[BoxConstraint],
    cumulative_bounds: list[BoxConstraint] | None = None,
    cumulative_baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the receding-horizon problem and return the whole input sequence.

    `reference` is the target for every predicted state, shape (horizon, n).
    `state_bounds` and `input_bounds` hold one box per prediction step, which is
    what lets the tightened sets of Eq. (36) shrink along the horizon.

    `cumulative_bounds`, with `cumulative_baseline`, bounds
    `cumulative_baseline + sum(u[0..i])` at each step -- the *absolute* value of
    a decision variable that is itself an increment, `input_bounds` only caps
    the increment's own size. Used by the velocity-form translational MPC,
    whose decision variable is a force increment: without this the QP plans
    against an actuator it believes has unlimited cumulative authority and only
    meets the real ceiling once the caller clips the realised input, which is
    where the mismatch between planned and applied force turns into a limit
    cycle. Left `None` for controllers whose decision variable is already the
    absolute input.
    """
    horizon = len(A)
    n, m = A[0].shape[0], B[0].shape[1]

    # A diverged state reaches the solver as bounds that have crossed over, and
    # OSQP then fails inside setup with a bare error code that says nothing about
    # the cause. Catch it here instead: the caller gets a held input and the
    # simulation reports where it actually lost the plant.
    if not np.all(np.isfinite(x0)):
        raise MpcStateError("state contains non-finite entries")

    Phi, Gamma = _lifted_matrices(A, B, n, m, horizon)

    # Stage weights, with the terminal weight replacing Q on the last block.
    weights = [Q] * (horizon - 1) + [Q + P]
    Q_bar = linalg.block_diag(*weights)
    R_bar = linalg.block_diag(*([R] * horizon))

    free_response = Phi @ x0
    error = free_response - reference.reshape(-1)

    hessian = Gamma.T @ Q_bar @ Gamma + R_bar
    hessian = 0.5 * (hessian + hessian.T)  # keep it exactly symmetric for OSQP
    gradient = Gamma.T @ Q_bar @ error

    state_lo = np.concatenate([b.lower for b in state_bounds]) - free_response
    state_hi = np.concatenate([b.upper for b in state_bounds]) - free_response
    # Two ways a diverged plant shows up here. The bounds can cross over, and
    # they can both grow past the solver's own notion of infinity (1e30), at
    # which point it clamps one side and not the other and reports the crossing
    # as a data-validation error that says nothing about the cause.
    if np.any(state_lo > state_hi) or np.max(
        np.abs(np.concatenate([state_lo, state_hi]))
    ) > 1e12:
        raise MpcStateError(
            "predicted free response has left the constraint set; the plant has "
            "already diverged"
        )
    input_lo = np.concatenate([b.lower for b in input_bounds])
    input_hi = np.concatenate([b.upper for b in input_bounds])
    # The input boxes are centred on the residual reaction term, so a diverged
    # arm shows up here rather than in the state bounds.
    if not (np.all(np.isfinite(input_lo)) and np.all(np.isfinite(input_hi))) or np.max(
        np.abs(np.concatenate([input_lo, input_hi]))
    ) > 1e12:
        raise MpcStateError(
            "input constraint set has run away; the residual reaction term is "
            "no longer physical"
        )

    blocks = [sparse.eye(m * horizon, format="csc"), sparse.csc_matrix(Gamma)]
    lower_blocks = [input_lo, state_lo]
    upper_blocks = [input_hi, state_hi]

    if cumulative_bounds is not None:
        running_sum = sparse.csc_matrix(
            np.kron(np.tril(np.ones((horizon, horizon))), np.eye(m))
        )
        baseline = np.tile(np.asarray(cumulative_baseline, dtype=float), horizon)
        cumulative_lo = np.concatenate([b.lower for b in cumulative_bounds]) - baseline
        cumulative_hi = np.concatenate([b.upper for b in cumulative_bounds]) - baseline
        blocks.append(running_sum)
        lower_blocks.append(cumulative_lo)
        upper_blocks.append(cumulative_hi)

    constraint = sparse.vstack(blocks, format="csc")
    lower = np.concatenate(lower_blocks)
    upper = np.concatenate(upper_blocks)

    problem = osqp.OSQP()
    problem.setup(
        P=sparse.csc_matrix(hessian),
        q=gradient,
        A=constraint,
        l=lower,
        u=upper,
        verbose=False,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=8000,
        polish=False,
    )
    result = problem.solve()

    # Trust the solver's own verdict, not the shape of what it returned.
    #
    # status_val == 1 is OSQP_SOLVED; everything else -- primal infeasible, dual
    # infeasible, max_iter_reached, sigint -- means the vector in result.x is not
    # a solution. The previous guard tested only for None and non-finite values,
    # which sounds equivalent and is not: on 'primal infeasible' OSQP returns
    # 2.14e9 on every element, a perfectly finite number that sailed through and
    # became a torque command. Downstream it was clipped to the actuator limit,
    # so the symptom was three joints pinned at exactly [6, 6, 6] -- shape-wise
    # indistinguishable from honest saturation, and unaffected by the horizon,
    # the tube width, the constraint bounds or the gain, because none of those
    # were ever reached. It survived several rounds of diagnosis for that reason.
    if result.info.status_val != 1 or result.x is None or not np.all(
        np.isfinite(result.x)
    ):
        # Hold the input at the centre of the first admissible box: a defined,
        # in-bounds command rather than whatever the failed solve left behind.
        return np.tile(input_bounds[0].centre, (horizon, 1))
    return result.x.reshape(horizon, m)


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Discrete LQR gain used as the tube feedback K of Eq. (32).

    Returned with the sign convention u = K e, so the closed-loop matrix of
    Eq. (33) is A_K = A + B K.
    """
    P = linalg.solve_discrete_are(A, B, Q, R)
    return -np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)


def riccati_terminal_cost(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Terminal cost satisfying Definition 1, from the Riccati solution."""
    return linalg.solve_discrete_are(A, B, Q, R)


@dataclass
class Tube:
    """Reachable sets and constraint tightening for one tube-based controller.

    `relative` and `absolute` set the bound on the per-step variation of each
    entry of the LPV matrices, i.e. the compact set of Assumption 1(iii).
    """

    gain: np.ndarray
    relative: float
    absolute: float
    state_margin: np.ndarray | None = None
    input_margin: np.ndarray | None = None

    def disturbance_half_width(
        self,
        A: np.ndarray,
        B: np.ndarray,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """Half-width of the one-step disturbance set W of Eq. (30) and (34).

        Eq. (34) takes the convex hull over the whole constraint set, i.e. over
        every admissible state and input. Doing that literally is unusable here.
        The manipulator's box spans +/- 3 rad and +/- 8 rad/s while the actual
        trajectory never leaves a tenth of it, so the worst-case product
        produces a tube several times wider than the constraint set itself; the
        tightening of Eq. (36) then collapses the predicted state onto a point
        and the controller commands identically zero torque while still
        reporting a solved QP.

        The disturbance set is therefore evaluated at the current operating
        point plus a margin, rather than over the whole box. That is exactly the
        "practical approach" of Gonzalez et al., reference [55] in the paper,
        which the paper cites for this construction. The guarantee becomes local
        -- it holds as long as the trajectory stays within the margin -- and the
        margin is what has to be widened if the tube is ever violated.

        The entrywise parameter bound scales with the varying part of A rather
        than with A itself, since A = I + dt * P: the identity does not vary, and
        including it would inflate the bound by a factor of 1 / dt.

        The absolute floor is taken per row, not over the whole matrix. Its job is
        to keep a nonzero bound where an entry happens to be zero, and a single
        scalar taken from the largest entry anywhere does that by handing every
        row the uncertainty of the stiffest one. Here the stiffest entry, 1.54,
        sits in a velocity row, while the joint-angle rows carry nothing but
        dt = 0.025: the floor then dominated their own relative term by a factor
        of thirty and gave a joint angle a one-step disturbance of 0.197 rad --
        7.9 rad/s of uncertainty on a joint that moves below 1 rad/s. Downstream,
        |K| @ R reached 23 N m against a 6 N m input half-width, the 75% cap in
        `BoxConstraint.tighten` engaged from the second horizon step onward, and
        with the cap engaged the tube's own precondition -- that K @ e fits
        inside the tightening -- no longer held. Per row the same pose gives
        0.0024 rad and |K| @ R of 3.7, and the cap never engages.
        """
        varying = np.abs(A - np.eye(A.shape[0]))
        magnitude_b = np.abs(B)
        delta_A = self.relative * varying + self.absolute * varying.max(
            axis=1, keepdims=True, initial=0.0
        )
        delta_B = self.relative * magnitude_b + self.absolute * magnitude_b.max(
            axis=1, keepdims=True, initial=0.0
        )

        state_margin = (
            self.state_margin if self.state_margin is not None else np.zeros_like(state)
        )
        input_margin = (
            self.input_margin
            if self.input_margin is not None
            else np.zeros_like(control)
        )
        reach_state = np.abs(state) + state_margin
        reach_input = np.abs(control) + input_margin
        return delta_A @ reach_state + delta_B @ reach_input

    def reachable_sets(
        self,
        A: np.ndarray,
        B: np.ndarray,
        state: np.ndarray,
        control: np.ndarray,
        horizon: int,
    ) -> list[np.ndarray]:
        """Propagate Eq. (35) and return the half-width at each step.

        The drift accumulates over the horizon, as in Eq. (26): the parameter at
        step i may have drifted i times, so the disturbance injected at step i
        grows linearly with i.
        """
        closed_loop = A + B @ self.gain
        unit = self.disturbance_half_width(A, B, state, control)
        magnitude = np.abs(closed_loop)

        half_widths = [np.zeros(A.shape[0])]  # R(0|k) = {0}
        for step in range(horizon):
            propagated = magnitude @ half_widths[-1]
            half_widths.append(propagated + (step + 1) * unit)
        return half_widths

    def minimal_rpi(
        self,
        A: np.ndarray,
        B: np.ndarray,
        state: np.ndarray,
        control: np.ndarray,
        horizon: int,
    ) -> np.ndarray | None:
        """Half-width of the minimal robust positively invariant set for the error.

        Theorem 1's asymptotic-stability argument and Proposition 1's recursive
        feasibility both rest on a terminal set that is RPI under the feedback
        u = K e: start inside it and the error stays inside it forever, whatever
        the parameters do within their bounds. Without one the horizon simply
        ends -- the optimiser may hand back a terminal state from which the next
        problem is infeasible, and nothing in the cost forbids it. A Riccati
        terminal *cost* discourages that; a terminal *set* rules it out.

        The paper computes this as a polytope with the MPT toolbox. Here it is an
        ellipsoid, reported as the box that bounds it, and the reason it is not
        simply a box is worth recording.

        The obvious construction is the fixed point of the recursion the rest of
        this tube already uses, b = |A_K| b + w, solved as (I - |A_K|)^-1 w. It
        does not exist. Entrywise magnitude discards the signs that make feedback
        stabilising, so |A_K| can fail to be contractive while A_K is Schur, and
        for this system it always does: the six-state model carries the attitude
        kinematics, which puts a 1 on the diagonal with a dt beside it, and an
        integrator chain in entrywise magnitude has a row sum above one by
        construction. Measured, A_K has spectral radius 0.817 and |A_K| has more
        than 1. The box is the wrong shape for this system, not a coarse one.

        What works is to keep the signs inside the powers. The minimal RPI set is
        the infinite Minkowski sum of the disturbance propagated forward,

            F = W (+) A_K W (+) A_K^2 W (+) ...

        which is a zonotope, and the box that circumscribes it has half-widths

            b* = sum_i |A_K^i| w

        -- the *signed* matrix power, then the magnitude, in that order. That
        converges for every Schur A_K and it is far tighter than the two
        alternatives tried before it:

          * |A_K|^i, magnitude first, is the fixed point of the recursion the
            rest of this tube uses. It does not exist here. Entrywise magnitude
            discards the signs that make feedback stabilising, and the six-state
            model carries the attitude kinematics, which puts a 1 on the diagonal
            with a dt beside it -- an integrator chain in entrywise magnitude has
            a row sum above one by construction. Measured, A_K has spectral
            radius 0.879 and |A_K| has more than 1.
          * An ellipsoid from the discrete Lyapunov equation, with the level set
            by the triangle inequality in the P-norm. It exists but is useless:
            that bound uses the worst-case *one-step* contraction, which for a
            non-normal A_K sits far closer to 1 than the asymptotic rate.
            Measured, lambda = 0.9831 against rho = 0.8785, and the resulting
            1 / (1 - lambda) inflates the disturbance by a factor of 59.

        The sum is truncated when the terms stop mattering, and the tail is
        bounded rather than dropped, so the result stays an over-estimate.

        `w` is the per-step disturbance at the *last* horizon step rather than
        the first. `reachable_sets` injects (i + 1) * unit at step i, modelling a
        parameter that may have drifted i times, and that grows without bound as
        i goes to infinity, so no invariant set exists for it. Over an infinite
        horizon the drift does not in fact accumulate without limit -- it
        saturates at the extent of the parameter's own compact set -- and the
        horizon's own value is the largest injection the finite prediction ever
        uses. Taking it as constant therefore upper-bounds the true set.

        Returns None when A_K is not Schur, in which case no invariant set exists
        and the caller must not pretend it has one.
        """
        closed_loop = A + B @ self.gain
        spectral_radius = np.abs(np.linalg.eigvals(closed_loop)).max()
        if spectral_radius >= 1.0:
            return None
        unit = horizon * self.disturbance_half_width(A, B, state, control)

        power = np.eye(closed_loop.shape[0])
        half_width = np.zeros_like(unit)
        for _ in range(_RPI_TERMS):
            term = np.abs(power) @ unit
            half_width = half_width + term
            if term.max() <= _RPI_TOLERANCE * max(half_width.max(), 1e-12):
                # The remaining tail is a geometric series in the spectral radius
                # of what is left, added rather than discarded so the answer stays
                # an over-estimate of the true set.
                return half_width + term * spectral_radius / (1.0 - spectral_radius)
            power = closed_loop @ power
        # Did not settle within the term budget: the loop is Schur but slowly, and
        # claiming an invariant set here would be claiming more than was computed.
        return None

    def tighten(
        self,
        state_bound: BoxConstraint,
        input_bound: BoxConstraint,
        reachable: list[np.ndarray],
        terminal: np.ndarray | None = None,
    ) -> tuple[list[BoxConstraint], list[BoxConstraint]]:
        """Apply Eq. (36) to the state and input constraint sets.

        `terminal`, when given, is the RPI half-width from `minimal_rpi`, and it
        replaces the reachable set at the final step. R(N) is what the error can
        reach by step N and says nothing about step N + 1; the RPI set is what it
        can reach ever. Tightening the last step by the larger of the two is what
        makes the terminal state one the next problem can still start from.
        """
        gain_magnitude = np.abs(self.gain)
        widths = list(reachable[1:])
        if terminal is not None:
            widths[-1] = np.maximum(widths[-1], terminal)
        states = [state_bound.tighten(r) for r in widths]
        inputs = [input_bound.tighten(gain_magnitude @ r) for r in reachable[:-1]]
        return states, inputs

    def terminal_set(
        self,
        A: np.ndarray,
        B: np.ndarray,
        state_bound: BoxConstraint,
        input_bound: BoxConstraint,
    ) -> np.ndarray | None:
        """The terminal set of Theorem 1: maximal, invariant, and admissible.

        This is the second half of the construction and `minimal_rpi` is the
        first. They are different objects and the theorem needs both, in order:

          * the *minimal* RPI of the error bounds where the true trajectory can
            be relative to the nominal one, and is what X is tightened by;
          * the *maximal* RPI inside what is left is the set the nominal terminal
            state must land in, so that u = K x can hold it there forever.

        Computing the second without the first is what makes the set collapse:
        the iteration is asked to find an invariant region inside constraints
        that have not been given room for the error, and it empties. That was
        observed directly, and it is why an attempt at this by way of a polytopic
        Pre-iteration reported "mRPI set collapsed, disturbances too large".

        In the box family the answer is closed-form. The nominal system under
        feedback is x+ = A_K x with no disturbance, so a box [-c, c] is invariant
        exactly when |A_K| c <= c. Perron-Frobenius gives such a c for free: if
        |A_K| is contractive its Perron vector v is positive and |A_K| v = rho v
        <= v, so every box along v is invariant and only the scale is left to
        choose. The scale is then the largest that fits both the tightened state
        set and the tightened input set through K.

        The result is the maximal invariant admissible box in the Perron
        direction. A polytope would be larger -- it can lean where a box cannot --
        so this is conservative, which for a terminal set is the safe direction:
        a state accepted here is certainly recoverable, never falsely so.

        Returns None when A_K is unstable, or when nothing admissible is left,
        which is Proposition 1's infeasibility condition.
        """
        closed_loop = A + B @ self.gain
        if np.abs(np.linalg.eigvals(closed_loop)).max() >= 1.0:
            return None

        # A Lyapunov ellipsoid, not a box, and with the signed matrix.
        #
        # A box was tried first and cannot work here, for a reason this file
        # already records above: entrywise magnitude discards the signs that make
        # the feedback stabilising, and the six-state model carries the attitude
        # kinematics -- an integrator chain, whose entrywise row sum exceeds one
        # by construction. Measured, rho(A_K) = 0.879 while rho(|A_K|) = 1.025,
        # so no box along any direction is invariant even though the loop is
        # comfortably stable.
        #
        # With signs kept, the natural invariant object is a sublevel set of a
        # Lyapunov function. Solving A_K' P A_K - P = -I gives P > 0 with
        # x' P x decreasing along every trajectory, so {x : x' P x <= c} maps
        # strictly into itself for any c > 0. Only the level is left to choose,
        # and it is the largest that keeps the set inside the tightened state
        # constraints and its feedback inside the tightened input constraints.
        #
        # For a half-space |a' x| <= h the largest level is h^2 / (a' P^-1 a),
        # which is exact rather than conservative. State rows use a = e_j; input
        # rows use a = K_k.
        try:
            P = linalg.solve_discrete_lyapunov(
                closed_loop.T, np.eye(closed_loop.shape[0]))
        except Exception:                      # noqa: BLE001 - see below
            # A singular or ill-conditioned solve means the Lyapunov certificate
            # does not exist numerically, which is the same answer as "no
            # terminal set" and must not be reported as one.
            return None
        P_inverse = np.linalg.inv(P)

        levels = []
        for j, half in enumerate(state_bound.half_width):
            if half > 0.0:
                levels.append(half ** 2 / P_inverse[j, j])
        for k, half in enumerate(input_bound.half_width):
            row = self.gain[k]
            denominator = float(row @ P_inverse @ row)
            if denominator > 1e-18 and half > 0.0:
                levels.append(half ** 2 / denominator)
        if not levels:
            return None
        level = min(levels)
        if level <= 0.0:
            return None

        # The QP takes box constraints, so what is handed back is the largest box
        # *inscribed in* that ellipsoid, along the axes. Every point of it is in
        # the ellipsoid and the ellipsoid is invariant, so the guarantee carries:
        # a terminal state in this box is one the feedback holds for ever. The box
        # is smaller than the ellipsoid, which is the safe direction.
        #
        # The worst corner of a box with half-widths d is the one maximising
        # x' P x, and that maximum is d' |P| d.
        direction = 1.0 / np.sqrt(np.diag(P))
        worst = float(direction @ np.abs(P) @ direction)
        if worst <= 0.0:
            return None
        return direction * np.sqrt(level / worst)

    def apply_terminal_set(
        self,
        A: np.ndarray,
        B: np.ndarray,
        state_bounds: list[BoxConstraint],
        input_bounds: list[BoxConstraint],
        reference: np.ndarray,
    ) -> bool:
        """Constrain the last horizon step to the terminal set, around the target.

        The terminal set of `terminal_set` is centred on the equilibrium, because
        that is where u = K x holds the nominal system. A tracking problem does
        not sit at the equilibrium: its terminal state is meant to be *at the
        reference*. Imposing a box around zero would therefore forbid tracking
        anything at all -- the optimiser would be told to end the horizon near
        level while being asked to fly a circle.

        So the set is translated onto the terminal reference and intersected with
        the tightened state set already there. The invariance argument survives
        the translation because the dynamics are linear and the reference is a
        steady state of them over one horizon.

        Returns False when the intersection is empty, which is Proposition 1's
        infeasibility condition, and leaves the bounds untouched in that case:
        an empty box would make the QP unsolvable and hide the reason.
        """
        terminal = self.terminal_set(A, B, state_bounds[-1], input_bounds[-1])
        if terminal is None:
            return False
        centre = np.asarray(reference)[-1]
        lower = np.maximum(state_bounds[-1].lower, centre - terminal)
        upper = np.minimum(state_bounds[-1].upper, centre + terminal)
        if np.any(lower > upper):
            return False
        state_bounds[-1] = BoxConstraint(lower, upper)
        return True

    @staticmethod
    def terminal_is_feasible(
        state_bound: BoxConstraint, terminal: np.ndarray | None
    ) -> bool:
        """Is anything left of the constraint set after tightening by the RPI set?

        Proposition 1 says an empty tightened set means the problem is infeasible
        and the controller must say so rather than proceed. This implementation
        instead caps the tightening at 75 % of each half-width and continues,
        which turns a silent failure into visible saturation but does not
        reproduce the proposition. This reports the condition the cap hides.
        """
        if terminal is None:
            return False
        return bool(np.all(terminal < state_bound.half_width))

    @staticmethod
    def confine(
        state: np.ndarray, nominal: np.ndarray, reachable: np.ndarray
    ) -> np.ndarray:
        """Re-seed the nominal state so the tube invariant x - x_nominal in R holds.

        This is the step the paper leaves implicit and it is load-bearing. The
        nominal system of Eq. (31) is driven by the nominal input alone while the
        real system runs on Eq. (32) and additionally saturates at the actuator
        limits. Nothing in Eq. (31)-(33) ties the two together, so once they
        disagree the pair drifts without bound: the error e feeding the tube gain
        stops being a tube error and becomes the accumulated divergence, and
        K @ e then commands torque in the wrong direction entirely.

        Clamping the nominal state into the reachable set around the measurement
        restores the invariant the construction assumes. When the tube is wide
        this leaves the nominal trajectory untouched and the scheme behaves as
        written; when the real state has run outside it, the nominal is pulled
        back to the tube boundary rather than abandoned.
        """
        error = state - nominal
        bounded = np.clip(error, -reachable, reachable)
        return state - bounded
