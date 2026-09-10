"""Reference trajectories for the three scenarios of Section V.

Scenario 1 ("nominal")
    Sinusoidal reference along x and y, ramp along z, and a periodic step
    reference on each joint with amplitude 0.33 rad and period 5 s. Read off
    Figs. 4-7: roughly two horizontal cycles over the 30 s run, z rising from
    1.5 m to about 2.6 m, and joint motion starting at the 5 s mark.

Scenario 2 ("fast")
    The paper doubles the amplitude, raises the frequency by 50 %, and drives
    the joints to 0.55 rad, per Figs. 10-13.

Scenario 3 ("disturbance")
    A rectangular horizontal path with impulsive force and moment disturbances
    at t = 7.5 s, 15 s and 22.5 s, Eq. (50).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ARM


@dataclass(frozen=True)
class Scenario:
    name: str
    duration: float
    horizontal_amplitude: float
    horizontal_period: float
    height_start: float
    height_end: float
    joint_amplitude: float
    joint_period: float
    joint_start: float
    rectangular: bool = False
    disturbances: bool = False


NOMINAL = Scenario(
    name="nominal",
    duration=30.0,
    horizontal_amplitude=2.5,
    horizontal_period=15.0,
    height_start=1.5,
    height_end=2.6,
    joint_amplitude=0.33,
    joint_period=5.0,
    joint_start=5.0,
)

FAST = Scenario(
    name="fast",
    duration=30.0,
    horizontal_amplitude=5.0,
    horizontal_period=10.0,
    height_start=1.5,
    height_end=3.0,
    joint_amplitude=0.55,
    joint_period=5.0,
    joint_start=5.0,
)

DISTURBANCE = Scenario(
    name="disturbance",
    duration=30.0,
    horizontal_amplitude=2.5,
    horizontal_period=30.0,
    height_start=1.5,
    height_end=2.6,
    joint_amplitude=0.33,
    joint_period=5.0,
    joint_start=5.0,
    rectangular=True,
    disturbances=True,
)

#: Not one of the paper's three. Everything is held still, so whatever the
#: vehicle does here is its own loops rather than the reference, which is the
#: only way to tell a tracking error from a limit cycle.
HOVER = Scenario(
    name="hover",
    duration=30.0,
    horizontal_amplitude=0.0,
    horizontal_period=15.0,
    height_start=1.5,
    height_end=1.5,
    joint_amplitude=0.0,
    joint_period=5.0,
    joint_start=5.0,
)

SCENARIOS = {s.name: s for s in (HOVER, NOMINAL, FAST, DISTURBANCE)}


def _rectangular_path(scenario: Scenario, t: float) -> tuple[float, float]:
    """Corner-to-corner rectangle, smoothed so the velocity stays bounded."""
    a = scenario.horizontal_amplitude
    phase = (t % scenario.horizontal_period) / scenario.horizontal_period
    corners = np.array([[-a, -a], [a, -a], [a, a], [-a, a], [-a, -a]])
    index = min(int(phase * 4), 3)
    local = phase * 4 - index
    # Smoothstep along each edge keeps the reference C1 at the corners.
    blend = local * local * (3.0 - 2.0 * local)
    point = corners[index] + blend * (corners[index + 1] - corners[index])
    return float(point[0]), float(point[1])


def translational_reference(scenario: Scenario, t: float) -> np.ndarray:
    """Desired [x, xdot, y, ydot, z, zdot] at time t."""
    climb = (scenario.height_end - scenario.height_start) / scenario.duration
    z = scenario.height_start + climb * min(t, scenario.duration)
    zdot = climb if t < scenario.duration else 0.0

    if scenario.rectangular:
        step = 1e-3
        x0, y0 = _rectangular_path(scenario, t)
        x1, y1 = _rectangular_path(scenario, t + step)
        return np.array([x0, (x1 - x0) / step, y0, (y1 - y0) / step, z, zdot])

    a = scenario.horizontal_amplitude
    w = 2.0 * np.pi / scenario.horizontal_period
    return np.array(
        [
            a * np.sin(w * t),
            a * w * np.cos(w * t),
            a * np.cos(w * t),
            -a * w * np.sin(w * t),
            z,
            zdot,
        ]
    )


def joint_reference(scenario: Scenario, t: float) -> np.ndarray:
    """Desired [Theta, Theta_dot] at time t.

    The reference is a square wave of alternating sign, as in Fig. 7. It is
    smoothed with a short tanh transition so the reference velocity is finite;
    an ideal step would demand infinite joint acceleration and neither
    controller could be blamed for missing it.
    """
    n = ARM.n_links
    if t < scenario.joint_start:
        return np.zeros(2 * n)

    elapsed = t - scenario.joint_start
    w = 2.0 * np.pi / scenario.joint_period
    # How sharp the transition is, which the paper does not specify: Fig. 7 shows
    # periodic steps but not their rise time. The value matters more than it
    # looks, because it sets the peak joint rate the reference demands and hence
    # the reaction the arm dumps on the vehicle at every reversal.
    #
    # 6.0 asks for 2.49 rad/s and holds the joints against their torque limit
    # through the whole transition; 2.0 asks for 0.83 rad/s and cuts the steady
    # tracking error from 0.12 rad to 0.02 rad. The reaction torque saturates at
    # the joint limit either way, so this alone does not buy headroom at the
    # rotors -- it buys tracking accuracy.
    sharpness = 2.0
    wave = np.tanh(sharpness * np.sin(w * elapsed))
    wave_rate = (
        sharpness * w * np.cos(w * elapsed) * (1.0 - wave * wave)
    )

    # Ramp the wave in over a quarter period, with an envelope whose value and
    # slope are both zero at the switch-on instant.
    #
    # Without this the reference is continuous in position but not in velocity.
    # tanh(k sin(w e)) is zero at e = 0 and that is precisely where its slope is
    # steepest, so the *rate* reference jumps from 0 to 0.83 rad/s in one sample
    # the moment the arm is told to move. The optimiser answers a step in its
    # target velocity with full torque, the arm's reaction saturates, and the
    # vehicle is hit with the whole thing 0.15 s after the manoeuvre begins --
    # which is exactly where the closed loop was failing, and it failed there
    # regardless of how gentle the rest of the waveform was.
    ramp_duration = 0.25 * scenario.joint_period
    u = min(elapsed / ramp_duration, 1.0)
    envelope = u * u * (3.0 - 2.0 * u)
    envelope_rate = (
        6.0 * u * (1.0 - u) / ramp_duration if u < 1.0 else 0.0
    )

    angles = np.full(n, scenario.joint_amplitude * envelope * wave)
    rates = np.full(
        n,
        scenario.joint_amplitude * (envelope_rate * wave + envelope * wave_rate),
    )
    return np.concatenate([angles, rates])


def disturbance(scenario: Scenario, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Impulsive wind disturbance of Eq. (50).

    The paper applies the three impulses at 7.5 s, 15 s and 22.5 s with
    magnitudes in the ratio 1/2 : 1 : -1. They are applied here over a 0.25 s
    window rather than as true impulses, so that the integrator sees them.
    """
    if not scenario.disturbances:
        return np.zeros(3), np.zeros(3)

    width = 0.25
    schedule = ((7.5, 0.5), (15.0, 1.0), (22.5, -1.0))
    base_force = np.array([3.0, 3.0, -3.0])
    base_torque = np.array([0.2, 0.2, -0.2])

    for instant, scale in schedule:
        if instant <= t < instant + width:
            return scale * base_force, scale * base_torque
    return np.zeros(3), np.zeros(3)


def horizon_reference(
    generator, scenario: Scenario, t: float, steps: int, dt: float
) -> np.ndarray:
    """Stack a reference over a prediction horizon."""
    return np.array([generator(scenario, t + (k + 1) * dt) for k in range(steps)])
