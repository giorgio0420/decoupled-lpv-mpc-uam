"""Offline reproduction of Section V: Algorithm 1, the ERTF comparison, Table III.

Run with

    python -m uam_control.simulate --scenario nominal
    python -m uam_control.simulate --scenario fast --controller lpv
    python -m uam_control.simulate --scenario disturbance --plot

The loop is multi-rate, as Section IV requires: the rotational and manipulator
controllers run at dt_eta = dt_gamma = 0.025 s and the translational controller
at dt_zeta = 0.05 s, while the plant integrates on a finer step.

One addition to Algorithm 1 is documented rather than hidden. Step 4 of the
algorithm converts the reference Euler *rates* into a body-rate reference, and
the rotational controller then tracks body rate only. Rate tracking alone leaves
the attitude itself unconstrained, so any integration error accumulates until
the slow translational loop notices it as a position error. The attitude
responses of Fig. 6 are far tighter than that would allow, so the reference
Euler rate is generated proportionally from the attitude error instead.

Note what is deliberately *not* done: the reference Euler rate is not obtained by
differencing successive attitude references. That feedforward is the obvious
reading of step 4, and it destabilises the cascade. The attitude reference comes
out of a constrained optimiser and moves in jumps, so its one-step difference is
a spike of amplitude (delta angle) / dt_zeta -- a 0.1 rad change becomes a
4 rad/s rate demand, which sits at the body-rate constraint and drives the inner
loop into saturation. The proportional term alone tracks the attitude to
numerical precision.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np

from .controllers import (
    AttitudeReference,
    ErtfBaseline,
    ManipulatorTubeMPC,
    RotationalTubeMPC,
    PositionPD,
    TranslationalMPC,
    attitude_reference,
    attitude_rate_command,
    euler_rate_to_body_rate,
)
from .coupling import decompose
from .mpc import MpcStateError
from .params import ARM, CONTROL, GRAVITY, TOTAL_MASS
from .plant import DivergedError, UamState, integrate, solve_accelerations
from .trajectory import (
    SCENARIOS,
    Scenario,
    disturbance,
    horizon_reference,
    joint_reference,
    translational_reference,
)

PLANT_STEP = 0.0025
# Per-axis: roll/pitch fast, yaw slow. A shared scalar gain applied to yaw as
# well was what previously sent yaw torque to its actuator limit -- this
# airframe's k_m / k_f = 0.02, so yaw needs about fifty times less gain than
# roll or pitch for the same rotor authority. Matches hover_node.py's tuned
# ATTITUDE_GAIN.
ATTITUDE_GAIN = np.array([4.0, 4.0, 0.05])
STATE_LABELS = ("x", "y", "z", "phi", "theta", "psi", "q1", "q2", "q3")


@dataclass
class Trace:
    """Recorded histories, one row per plant step."""

    time: list[float] = field(default_factory=list)
    position: list[np.ndarray] = field(default_factory=list)
    position_ref: list[np.ndarray] = field(default_factory=list)
    attitude: list[np.ndarray] = field(default_factory=list)
    attitude_ref: list[np.ndarray] = field(default_factory=list)
    joints: list[np.ndarray] = field(default_factory=list)
    joints_ref: list[np.ndarray] = field(default_factory=list)
    thrust: list[float] = field(default_factory=list)
    torque: list[np.ndarray] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {name: np.array(value) for name, value in self.__dict__.items()}

    def integral_absolute_error(self, dt: float) -> dict[str, float]:
        """Table III: the integral of the absolute tracking error."""
        errors = np.hstack(
            [
                np.array(self.position) - np.array(self.position_ref),
                np.array(self.attitude) - np.array(self.attitude_ref),
                np.array(self.joints) - np.array(self.joints_ref),
            ]
        )
        totals = np.sum(np.abs(errors), axis=0) * dt
        return dict(zip(STATE_LABELS, totals))


def run(
    scenario: Scenario,
    controller: str = "lpv",
    verbose: bool = True,
) -> Trace:
    """Execute Algorithm 1 (or the ERTF baseline) over the whole scenario."""
    n = ARM.n_links
    # Start on the trajectory. The reference of Eq. Section V begins at
    # y = A, not at the origin, so launching from [0, 0, h] would present the
    # loop with a step of the full amplitude at t = 0 and saturate the tilt
    # command before the run has begun.
    start = translational_reference(scenario, 0.0)
    state = UamState.at_rest(height=scenario.height_start)
    state.position = np.array([start[0], start[2], start[4]])
    state.velocity = np.array([start[1], start[3], start[5]])
    trace = Trace()

    # UAM_TRANS_MPC=1 mirrors the switch already wired in hover_node.py, so the
    # paper's own translational stage (Eq. 13-18) can be reproduced offline,
    # where a run costs seconds instead of a Gazebo session.
    translational = (
        TranslationalMPC(mass=TOTAL_MASS)
        if os.environ.get("UAM_TRANS_MPC") == "1"
        else PositionPD()
    )
    # UAM_TILT_SLEW [rad/s], unset = unbounded. A passive rate limit on the
    # roll/pitch command between the translational and rotational stages:
    # unlike a gain, it can only slow the demand down, never amplify it, so it
    # cannot itself destabilise the loop the way retuning K or Q/R did in
    # testing (see README 4.4). Tries to buy back the phase margin the
    # 1-sample delay costs without touching either controller's own tuning.
    tilt_slew = float(os.environ.get("UAM_TILT_SLEW", "inf"))
    previous_tilt = np.zeros(2)
    use_lpv = controller == "lpv"
    rotational = RotationalTubeMPC() if use_lpv else None
    manipulator = ManipulatorTubeMPC() if use_lpv else None
    baseline = None if use_lpv else ErtfBaseline()

    # Hold-over commands between control updates.
    attitude_target = AttitudeReference(thrust=TOTAL_MASS * GRAVITY, roll=0.0, pitch=0.0)
    body_rate_target = np.zeros(3)
    uav_torque = np.zeros(3)
    joint_torque = np.zeros(n)
    body_rate_dot = np.zeros(3)
    joint_ddot = np.zeros(n)
    previous_attitude_ref = np.zeros(3)

    steps = int(round(scenario.duration / PLANT_STEP))
    fast_every = int(round(CONTROL.dt_eta / PLANT_STEP))
    slow_every = int(round(CONTROL.dt_zeta / PLANT_STEP))

    coupling = decompose(
        state.joints,
        state.joint_rates,
        joint_ddot,
        state.body_rate,
        body_rate_dot,
        attitude_target.thrust,
    )

    for step in range(steps):
        t = step * PLANT_STEP

        # The decomposition is what the controllers schedule on, so it is
        # refreshed at the control rate rather than at the integration rate.
        if step % fast_every == 0:
            # Schedule on the accelerations the controllers intend, not on the
            # ones just differentiated out of the plant: see the note in
            # ManipulatorTubeMPC on why the measured values close an unstable
            # algebraic loop through tau_bar.
            coupling = decompose(
                state.joints,
                state.joint_rates,
                manipulator.predicted_joint_acceleration if use_lpv else joint_ddot,
                state.body_rate,
                rotational.predicted_body_rate_dot if use_lpv else body_rate_dot,
                attitude_target.thrust,
            )

        # Algorithm 1, steps 2-4: translational optimisation, then the attitude
        # reference it implies.
        if step % slow_every == 0:
            reference = horizon_reference(
                translational_reference, scenario, t, CONTROL.N_zeta, CONTROL.dt_zeta
            )
            try:
                optimal_force = translational(state.translational_vector(), reference)
            except MpcStateError as error:
                if verbose:
                    print(f"  diverged at t = {t:.2f} s: {error}")
                break
            attitude_target = attitude_reference(
                optimal_force, coupling, 0.0, state.attitude
            )
            if np.isfinite(tilt_slew):
                demanded = np.array([attitude_target.roll, attitude_target.pitch])
                step_limit = tilt_slew * CONTROL.dt_zeta
                limited = previous_tilt + np.clip(
                    demanded - previous_tilt, -step_limit, step_limit
                )
                previous_tilt = limited
                attitude_target = AttitudeReference(
                    thrust=attitude_target.thrust, roll=limited[0], pitch=limited[1]
                )

            desired = np.array([attitude_target.roll, attitude_target.pitch, 0.0])
            previous_attitude_ref = desired

            if hasattr(translational, "predicted_force_next"):
                # Step 4, read literally: the paper says "using the obtained
                # reference Euler rates" without ever showing the equation
                # that produces them from the angles of Eq. (20)-(21) -- a
                # real gap in the text, not an omitted triviality. Differencing
                # across separate calls (the obvious reading) was tried before
                # today and destabilises, because a receding-horizon replan can
                # jump the commanded angle between calls with nothing tying the
                # two together. This differences *within* the single solve that
                # just ran instead: `predicted_force_next` is step 1 of the
                # same predicted trajectory step 0 (`optimal_force`) came from,
                # so both angles come from one continuous plan and the rate
                # between them is the one the optimiser actually committed to.
                next_target = attitude_reference(
                    translational.predicted_force_next, coupling, 0.0, state.attitude
                )
                eta_dot_r = np.array([
                    next_target.roll - attitude_target.roll,
                    next_target.pitch - attitude_target.pitch,
                    0.0,
                ]) / CONTROL.dt_zeta
                body_rate_target = euler_rate_to_body_rate(
                    attitude_target.roll, attitude_target.pitch
                ) @ eta_dot_r
            else:
                # SO(3) geometric error, not B_T_I @ (gain * euler_error): the
                # latter's pitch row carries a factor of cos(roll) and
                # annihilates the pitch command at roll = +/-pi/2. See
                # attitude_rate_command's docstring for the crash this avoids.
                body_rate_target = attitude_rate_command(
                    state.attitude, attitude_target.roll, attitude_target.pitch,
                    0.0, ATTITUDE_GAIN)

        # Algorithm 1, steps 5-8: the two tube-based controllers.
        if step % fast_every == 0:
          try:
              # Step 4: the body-rate reference the rotational loop tracks,
              # xr_eta = [pr, qr, rr] of Eq. (22). Held from the last
              # translational update, same as the ERTF baseline's.
              rate_reference = np.tile(body_rate_target, (CONTROL.N_eta, 1))
              joint_target = horizon_reference(
                  joint_reference, scenario, t, CONTROL.N_gamma, CONTROL.dt_gamma
              )

              if use_lpv:
                  uav_torque = rotational(
                      state.body_rate, rate_reference, coupling)
                  # Schedule the arm on the angular acceleration the rotational
                  # loop has just committed to, not on the lagged measurement.
                  joint_torque = manipulator(
                      state.joint_vector(),
                      joint_target,
                      state.body_rate,
                      rotational.predicted_body_rate_dot,
                      attitude_target.thrust,
                  )
              else:
                  uav_torque = baseline.rotational(
                      state.body_rate,
                      body_rate_target,
                      state.joint_vector(),
                      joint_ddot,
                      attitude_target.thrust,
                      CONTROL.dt_eta,
                  )
                  joint_torque = baseline.manipulator(
                      state.joint_vector(),
                      joint_target[0],
                      state.body_rate,
                      attitude_target.thrust,
                      CONTROL.dt_gamma,
                  )
                  baseline.previous_body_rate_dot = body_rate_dot.copy()
          except MpcStateError as error:
              if verbose:
                  print(f"  diverged at t = {t:.2f} s: {error}")
              break

        # Algorithm 1, step 9: apply and integrate.
        force_disturbance, torque_disturbance = disturbance(scenario, t)
        try:
            accelerations = solve_accelerations(
                state,
                attitude_target.thrust,
                uav_torque,
                joint_torque,
                force_disturbance,
                torque_disturbance,
            )
        except (DivergedError, MpcStateError) as error:
            if verbose:
                print(f"  diverged at t = {t:.2f} s: {error}")
            break
        body_rate_dot = accelerations.body_rate_dot
        joint_ddot = accelerations.joint_dot
        state = integrate(state, accelerations, PLANT_STEP)

        position_ref = translational_reference(scenario, t)
        trace.time.append(t)
        trace.position.append(state.position.copy())
        trace.position_ref.append(position_ref[[0, 2, 4]])
        trace.attitude.append(state.attitude.copy())
        trace.attitude_ref.append(
            np.array([attitude_target.roll, attitude_target.pitch, 0.0])
        )
        trace.joints.append(state.joints.copy())
        trace.joints_ref.append(joint_reference(scenario, t)[:n])
        trace.thrust.append(attitude_target.thrust)
        trace.torque.append(uav_torque.copy())

        if not np.all(np.isfinite(state.position)):
            if verbose:
                print(f"  diverged at t = {t:.2f} s")
            break

    return trace


def format_table(results: dict[str, dict[str, float]]) -> str:
    """Render the IAE comparison in the layout of Table III."""
    names = list(results)
    width = max(len(n) for n in names) + 2
    lines = ["", "IAE performance index", "-" * (10 + width * len(names))]
    lines.append("state".ljust(10) + "".join(n.upper().ljust(width) for n in names))
    for label in STATE_LABELS:
        row = label.ljust(10)
        for name in names:
            value = results[name].get(label, float("nan"))
            row += f"{value:.3f}".ljust(width)
        lines.append(row)
    return "\n".join(lines)


def plot(traces: dict[str, Trace], scenario: Scenario, path: str) -> None:
    """Reproduce the figure set of Section V."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 9))
    colours = {"lpv": "tab:red", "ertf": "tab:blue"}

    axis_3d = figure.add_subplot(2, 3, 1, projection="3d")
    for name, trace in traces.items():
        data = np.array(trace.position)
        axis_3d.plot(data[:, 0], data[:, 1], data[:, 2], color=colours.get(name), lw=1.2, label=name.upper())
    reference = np.array(next(iter(traces.values())).position_ref)
    axis_3d.plot(reference[:, 0], reference[:, 1], reference[:, 2], "g--", lw=1.2, label="RT")
    axis_3d.set_title("3-D trajectory")
    axis_3d.set_xlabel("x (m)")
    axis_3d.set_ylabel("y (m)")
    axis_3d.set_zlabel("z (m)")
    axis_3d.legend(fontsize=7)

    panels = [
        (2, "position", 0, "x (m)"),
        (3, "position", 2, "z (m)"),
        (5, "attitude", 0, "phi (rad)"),
        (6, "attitude", 1, "theta (rad)"),
    ]
    for index, key, column, ylabel in panels:
        axis = figure.add_subplot(2, 3, index)
        for name, trace in traces.items():
            values = np.array(getattr(trace, key))
            axis.plot(trace.time, values[:, column], color=colours.get(name), lw=1.0, label=name.upper())
        reference = np.array(getattr(next(iter(traces.values())), f"{key}_ref"))
        axis.plot(
            next(iter(traces.values())).time,
            reference[:, column],
            "g--",
            lw=1.0,
            label="RT",
        )
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    axis = figure.add_subplot(2, 3, 4)
    for name, trace in traces.items():
        values = np.array(trace.joints)
        axis.plot(trace.time, values[:, 0], color=colours.get(name), lw=1.0, label=name.upper())
    axis.plot(
        next(iter(traces.values())).time,
        np.array(next(iter(traces.values())).joints_ref)[:, 0],
        "g--",
        lw=1.0,
        label="RT",
    )
    axis.set_xlabel("time (s)")
    axis.set_ylabel("q1 (rad)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)

    figure.suptitle(f"UAM trajectory tracking - {scenario.name} scenario")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    print(f"figures written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="nominal", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--controller",
        default="both",
        choices=("lpv", "ertf", "both"),
        help="which control scheme to run",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default="tracking.png")
    arguments = parser.parse_args()

    scenario = SCENARIOS[arguments.scenario]
    wanted = ("lpv", "ertf") if arguments.controller == "both" else (arguments.controller,)

    traces, results = {}, {}
    for name in wanted:
        print(f"running {name} on the {scenario.name} scenario ...")
        trace = run(scenario, controller=name)
        traces[name] = trace
        results[name] = trace.integral_absolute_error(PLANT_STEP)

    print(format_table(results))
    if arguments.plot:
        plot(traces, scenario, arguments.output)


if __name__ == "__main__":
    main()
