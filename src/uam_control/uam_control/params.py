"""Model and controller parameters.

Values come from Table I (UAM model parameters) and Table II (controller
parameters) of Eskandarpour et al., "Decoupled Dynamic Modeling by Decomposing
the Cross-Coupled Dynamics and Tube-Based LPV-MPC Control Scheme for Aerial
Manipulation", IEEE TAES 61(5), 2025.

Rotor geometry is not part of the paper; it comes from the local Gazebo model
(`hexacopter_sim/model/robot.xacro`) and is used only by the control-allocation
stage that maps the wrench produced by the controllers onto six rotor speeds.
"""

import os
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.8  # Table I uses 9.8, not 9.81. Kept as-is for comparability.


@dataclass(frozen=True)
class ArmParams:
    """Three-link manipulator, Table I.

    The link frames follow Craig's convention with the link extending along the
    frame's x-axis and the joint rotating about the frame's z-axis. Table I
    reports the moments of inertia as (15e-4, 15e-4, 55e-6) about
    (x_Mi, y_Mi, z_Mi); those axes are permuted with respect to the convention
    used here, so the values are re-ordered to keep the slender-rod physics
    consistent: the small moment is about the link's own axis and
    m * l**2 / 12 = 0.166 * 0.33**2 / 12 = 1.5e-3 about the two transverse axes.
    """

    n_links: int = 3
    mass: float = 0.166  # m_i [kg]
    length: float = 0.33  # l_i [m]
    inertia_axial: float = 55e-6  # about the link axis [kg m^2]
    inertia_transverse: float = 15e-4  # about the two transverse axes [kg m^2]
    # Viscous damping at each joint, read from robot.xacro:
    #   <dynamics damping="0.05" friction="0.0" />
    #
    # Gazebo applies it and the model did not, so the two disagreed by
    # 0.05 * qdot N m on every joint -- 0.125 N m at 2.5 rad/s, against joint
    # torques that are often below 1 N m. That is a real term, not a rounding
    # difference, and it belongs in the model the MPC predicts with.
    damping: float = 0.05  # [N m s / rad]

    def link_inertia(self) -> np.ndarray:
        """Diagonal inertia tensor of one link, expressed in that link's frame."""
        return np.diag(
            [self.inertia_axial, self.inertia_transverse, self.inertia_transverse]
        )

    def com_offset(self) -> np.ndarray:
        """Position of a link's centre of mass in that link's own frame."""
        return np.array([0.5 * self.length, 0.0, 0.0])

    def link_offset(self) -> np.ndarray:
        """Position of the next frame's origin in the current link's frame."""
        return np.array([self.length, 0.0, 0.0])


@dataclass(frozen=True)
class UavParams:
    """Hexarotor, Table I."""

    mass: float = 5.0  # m_uav [kg]
    inertia_xx: float = 0.18  # I_xx about x_b [kg m^2]
    inertia_yy: float = 0.18  # I_yy about y_b [kg m^2]
    inertia_zz: float = 0.3  # I_zz about z_b [kg m^2]

    def inertia_tensor(self) -> np.ndarray:
        """UAV inertia tensor I_t of Eq. (3b)."""
        return np.diag([self.inertia_xx, self.inertia_yy, self.inertia_zz])


@dataclass(frozen=True)
class RotorGeometry:
    """Rotor layout of the local Gazebo model, used by control allocation.

    Positions are expressed in `robot_base` and were read off the joint origins
    in `robot.xacro`. The rotor plane sits at z = 2.021 m in that frame, but the
    height does not enter the allocation: with a thrust vector along +z the
    torque r x F only depends on the in-plane offsets.

    `spin` is +1 for one rotation sense and -1 for the other, alternating around
    the hexagon. The absolute sign is provisional until a low-speed yaw test in
    Gazebo confirms which sense the propeller geometry actually produces.
    """

    x: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.000000, 0.173203, 0.173205, 0.000000, -0.173203, -0.173205]
        )
    )
    y: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.1, -0.1, -0.2, -0.1, 0.1])
    )
    spin: np.ndarray = field(
        default_factory=lambda: np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    )
    # The draft `hexacopter_control/config/hexacopter.yaml` carries k_f = 8.0e-6
    # with max_omega = 900, and its README flags both as provisional pending a
    # static-thrust test. They do not survive one: six rotors at those numbers
    # produce 38.9 N against a vehicle weighing 53.9 N, so the aircraft cannot
    # lift itself, let alone hover with margin.
    #
    # k_f below is sized instead for a thrust-to-weight ratio of 2.5, which is
    # ordinary for a multirotor and puts the hover point at 63 % of maximum rotor
    # speed. k_m keeps the original k_m / k_f ratio of 0.02.
    #
    # These are still not measured values. They are a self-consistent stand-in so
    # that the control design is exercised against an airframe that can fly; the
    # real ones have to come from a static-thrust test on the actual hardware.
    thrust_coefficient: float = 2.772e-5  # k_f [N s^2]
    moment_coefficient: float = 5.544e-7  # k_m [N m s^2]
    max_omega: float = 900.0  # [rad/s]

    def allocation_matrix(self) -> np.ndarray:
        """Map squared rotor speeds to the wrench [T, tau_x, tau_y, tau_z].

        With F_i = k_f * omega_i**2 along +z and r_i = (x_i, y_i, *), the torque
        contribution is r_i x F_i = (y_i * F_i, -x_i * F_i, 0), plus the drag
        reaction s_i * k_m * omega_i**2 about z.
        """
        kf, km = self.thrust_coefficient, self.moment_coefficient
        return np.vstack(
            [
                np.full(6, kf),
                self.y * kf,
                -self.x * kf,
                self.spin * km,
            ]
        )


@dataclass(frozen=True)
class ControllerParams:
    """Table II. Horizons, sample times and weights for the three controllers.

    Subscripts follow the paper: zeta = translational, eta = rotational,
    gamma = manipulator.
    """

    # Prediction horizons
    N_zeta: int = 8
    N_eta: int = int(os.environ.get("UAM_N_ETA", 4))
    N_gamma: int = 4  # Table II

    # Sample times [s]
    #
    # Not Table II's 0.01. That value was set here once the closed form brought a
    # tick from 43 ms to 23, on the reasoning that 23 < 25 so 100 Hz had become
    # executable. It had not: 23 ms was the cost of the three controllers alone,
    # and the tick in situ measures 19 to 45 ms with a median of 25 and a 95th
    # percentile of 45. A 10 ms timer against a 25 ms tick means the loop closes
    # at whatever period it manages, while the gain is designed for the period it
    # was told, and the two disagree by a factor of two to four.
    #
    # That disagreement is not a detail. The rate loop's own eigenvalues, with the
    # one sample of actuation delay the implementation really has:
    #
    #     dt         10 ms    20 ms    25 ms    30 ms    40 ms
    #     rho       0.979    0.957    0.947    0.987    1.140
    #
    # -- stable through 30 ms and divergent at 40, which is inside the spread the
    # node actually produces. Measured, it oscillated with growing amplitude from
    # 0.18 s and sat on the +/-7 N m limit from 0.6 s onward.
    #
    # 50 ms is chosen so the budget exceeds the cost with margin rather than by
    # luck: 23 ms of work in a 50 ms period leaves the loop regular, the gap
    # between ticks equal to the period, and one sample of delay instead of two or
    # three. The LQR gain is re-solved at whichever dt is in force, so 20 Hz is
    # the same design procedure at a slower rate, not a detuning -- K falls from
    # 5.82 to 2.97 and rho to 0.915.
    #
    # Table II's 100 Hz needs a tick under about 8 ms. That is a compiled
    # controller, not this one, and it is the honest reason for the difference.
    dt_zeta: float = 0.05
    dt_eta: float = 0.05
    dt_gamma: float = 0.05

    # State weights
    Q_zeta_gain: float = 10.0  # 10 * I_12 on the augmented translational state
    Q_eta_gain: float = 15.0  # 15 * I_3
    Q_gamma_gain: float = 100.0  # 100 * I_6

    # Input weights
    R_zeta_gain: float = 0.1  # 0.1 * I_3
    # Do not raise this. Against Q_eta = 15 it gives K_yy = -12.3 N m per rad/s
    # of body-rate error, which is 68 rad/s^2 per rad/s -- past the 1 / dt_eta
    # = 40 that nulls the error in one sample, so on paper the loop overshoots
    # every step and softening it looks obviously right. It is not.
    #
    # Measured at hover, arm off, nothing commanded to move:
    #
    #     R_eta 0.01, ATTITUDE_GAIN 2.0     x drift  -0.33 .. +0.34 m
    #     R_eta 1.0,  ATTITUDE_GAIN 0.7     x drift -30.37 .. +36.46 m
    #     R_eta 1.0,  ATTITUDE_GAIN 0.35    x drift -99.71 .. +155.13 m
    #
    # The fast rate loop is what holds position: at 0.01 the vehicle keeps
    # station inside a third of a metre. Softening it lets pitch overshoot its
    # own command by 60% -- 0.90 rad reached against 0.55 asked -- with the
    # rotor torque never once at its limit, and the vehicle then flies off on
    # the tilt. Lowering ATTITUDE_GAIN to match makes it worse, not better.
    R_eta_gain: float = 0.3  # prova assetto
    # Raising this to 1.0 was tried against the 10^4 ratio with Q_gamma and moved
    # nothing: median |Theta_ddot| went 197.6 -> 185.5 and the vehicle still ran
    # away. R enters `lqr_gain` as well as the QP, so a hundredfold rise softened
    # K by ten and the tracking error grew by about as much, leaving the product
    # K @ e where it was. Back to Table II's value.
    R_gamma_gain: float = 0.01  # 0.01 * I_3

    # Terminal weights
    P_eta_gain: float = 15.0  # 15 * I_3
    P_gamma_gain: float = 100.0  # 100 * I_3

    def Q_zeta(self) -> np.ndarray:
        return self.Q_zeta_gain * np.eye(12)

    def R_zeta(self) -> np.ndarray:
        return self.R_zeta_gain * np.eye(3)

    # Six states now, [phi, theta, psi, p, q, r]. The angles carry the weight
    # because they are what is regulated; the rates are there to be damped, and
    # weighting them as heavily as the angles asks the optimiser to hold the
    # vehicle still rather than to point it.
    #
    # Chosen so the closed loop keeps the stability margin the flying cascade
    # had, measured on the six-state model with the one sample of actuation delay
    # the implementation really has:
    #
    #     angle weight    240      150      100       60       25
    #     K_angle       10.81     8.70     7.20     5.65     3.71
    #     rho          1.0171   0.9949   0.9791   0.9631   0.9435
    #
    # The cascade that flies -- attitude gain 4.0 into a rate loop with K = 2.97,
    # so an effective angle gain of 11.86 -- scores 0.9782 on this same model, so
    # 100 is the apples-to-apples choice: identical margin, with the attitude now
    # inside a constraint set instead of outside every one.
    #
    # Note that 0.9782, not the 0.9149 recorded earlier. That figure was computed
    # on a three-state model which omits the direct torque-to-angle feedthrough
    # within a step, and on an attitude gain of 2.0 rather than the 4.0 actually
    # flown. The six-state model was more complete and put the flying
    # configuration closer to the edge than was thought -- reverted to literal
    # 3-state (Eq. 22-23), so that margin is not re-measured here.

    def Q_eta(self) -> np.ndarray:
        return self.Q_eta_gain * np.eye(3)

    def R_eta(self) -> np.ndarray:
        return self.R_eta_gain * np.eye(3)

    def P_eta(self) -> np.ndarray:
        return self.P_eta_gain * np.eye(3)

    def Q_gamma(self) -> np.ndarray:
        return self.Q_gamma_gain * np.eye(6)

    def R_gamma(self) -> np.ndarray:
        return self.R_gamma_gain * np.eye(3)

    def P_gamma(self) -> np.ndarray:
        # Table II gives P_gamma as 100 * I_3; it weights the three joint
        # positions of the terminal state, so it is embedded in a 6x6 block.
        return np.diag([self.P_gamma_gain] * 3 + [0.0] * 3)


@dataclass(frozen=True)
class Constraints:
    """State and input constraint sets.

    The paper states that the constraint sets exist and are polytopes but does
    not tabulate them. These bounds are chosen to be consistent with the
    reported trajectories (position within a few metres, attitude well inside
    the small-angle region, thrust able to lift 5 kg with margin) and are the
    quantities to revisit when tuning against hardware.
    """

    # Translational
    position: float = 12.0  # [m]
    velocity: float = 6.0  # [m/s]
    force_xy: float = 30.0  # [N]
    # Cap on the acceleration the outer position loop may ask for. Sized so the
    # tilt it implies stays inside LIMITS.tilt: atan(4.0 / 9.8) = 0.39 rad.
    acceleration: float = 4.0  # [m/s^2]
    force_z_min: float = 5.0  # [N]
    force_z_max: float = 110.0  # [N]

    # Rotational
    body_rate: float = 3.0  # [rad/s]
    # The two actuator limits below have to be sized against each other, not
    # independently: whatever angular acceleration the rotors produce, the arm's
    # joints have to be able to hold the arm against it, or the arm gets away and
    # its reaction torque then exceeds anything the rotors can answer with.
    #
    # Holding the arm costs roughly 0.6 N m at joint 1 per rad/s^2 of base
    # angular acceleration, so the joint limit and the rotor limit are coupled
    # through the effective inertia.
    #
    # These are not free choices: they come from what the rotors can actually
    # deliver while holding hover thrust, computed from the allocation matrix.
    # Roll and pitch reach about 7.7 N m; yaw only 1.08 N m, because yaw is
    # produced by rotor drag rather than by differential thrust and k_m / k_f is
    # small. Treating yaw as if it had the same authority as roll asks the
    # allocator for a wrench that does not exist.
    # The rule two paragraphs up binds, and the pair 7 / 6 violates it: 7 N m on
    # I_yy = 0.18 is 39 rad/s^2 of base angular acceleration, holding the arm
    # against that costs about 23 N m at joint 1, and the joint has 6. Driving
    # the arm alone with that acceleration and nothing else saturates its joints
    # on 91% of steps; with the acceleration removed and the thrust swing and
    # body rate left in, they never saturate at all.
    #
    # Cutting this to 1.5 to satisfy the rule from the vehicle's side was tried
    # and is worse: at 1.5 N m the vehicle cannot hold its own attitude, tumbles,
    # and thrust demand reached 4758 N. Rotor authority buys attitude control
    # before it buys disturbance, so the pair has to be satisfied from the arm's
    # side instead -- see LIMITS.joint_torque.
    torque: float = 7.0  # [N m], roll and pitch
    torque_yaw: float = 1.0  # [N m]

    # Attitude reference produced by the translational stage
    # 0.6 rad is 34 degrees, which is a lot of tilt for a hovering manipulator
    # and it is what let the arm's reaction throw the vehicle: 34 degrees is
    # 6.6 m/s^2 of horizontal acceleration, and the attitude command was seen
    # pinned at this cap with its sign flipping on 41% of ticks. The cap is what
    # bounds how far a moment disturbance can push the vehicle sideways before
    # the position loop gets a say. At 0.15 rad the same disturbance buys
    # 1.5 m/s^2. It also stays above what the position loop itself asks for,
    # since LIMITS.acceleration = 4.0 m/s^2 is only reachable at 0.39 rad -- so
    # this now binds the disturbance path and clips the position loop's own
    # demand, which is the trade being made.
    tilt: float = 0.6  # [rad], caps |phi_r| and |theta_r|

    # Manipulator. Taken from the joint limits declared in robot.xacro
    # (lower=-3.1416 upper=3.1416, effort=6.0, velocity=10.0), not chosen here.
    #
    # joint_angle carries 8 % over the mechanical stop on purpose. The MPC state
    # box exists to keep the *predicted* trajectory sensible; the hard limit is
    # enforced by the simulator regardless, so a box tighter than physics buys
    # nothing and costs a great deal. At 3.0 it was tighter than pi, and the arm
    # rests at pi: every reference built around the rest pose was silently
    # clipped below the current position, the residual error saturated the
    # joints at [6, 6, 6], and the reaction threw the vehicle off. A box whose
    # boundary coincides exactly with the rest pose has the same failure in
    # milder form, because Eq. (46) then tightens it to nothing.
    joint_angle: float = 3.4  # [rad], stop is pi
    # The shoulder cannot do what the other two can. The arm hangs under the
    # vehicle, so rotating joint 1 past a quarter turn drives link 1 up into the
    # rotor plane and the frame; the XACRO's +/-pi came from a CAD export, not
    # from a mechanism. Beyond the realism, the wide range is what let the
    # optimiser reach poses with the arm's centre of mass far off the vehicle's
    # axis, and an off-axis centre of mass is a moment the vehicle can only
    # cancel by tilting, which is how it ends up translating away.
    #
    # Elbow and wrist keep their range: those fold without hitting anything.
    joint_angle_first: float = 1.5708  # [rad], pi/2 alla spalla

    def joint_angle_vector(self) -> np.ndarray:
        """Per-joint angle bound; the shoulder is mechanically tighter."""
        return np.array(
            [self.joint_angle_first] + [self.joint_angle] * (ARM.n_links - 1)
        )
    joint_rate: float = 10.0  # [rad/s], matches the xacro velocity limit
    # What the controller is allowed to ask for, which is not what the joint can
    # physically deliver: the XACRO keeps effort = 6.0 so the arm can still hold
    # itself out horizontally, which costs 2.42 N m at hover thrust.
    #
    # The controller gets less because the links are light: 1 N m buys about
    # 90 rad/s^2 at joint 1 and twice that at joint 2, so a full 6 N m command
    # moves the joint through its whole 10 rad/s rate range inside one 25 ms
    # control period. At that point the sample time, not the controller, decides
    # what the arm does, and the reaction it throws at the vehicle reached 49 N
    # against a 54 N weight. 2.5 N m keeps holding authority at hover with
    # margin and caps the reaction near a fifth of the vehicle's weight.
    #
    # What the controller may ask for, which is not what the joint can deliver.
    # The XACRO keeps effort = 6.0, so the arm can still physically hold itself
    # out horizontally against 2.42 N m; this is the budget the MPC is given.
    #
    # 6.0 is what a CAD export defaulted to, and on these links it is enormous:
    # 1 N m buys about 90 rad/s^2 at joint 1. Swept at hover, with the arm the
    # only thing changed and the vehicle otherwise holding station to +/-0.34 m:
    #
    #     arm off      x  -0.33 .. + 0.34 m    |q - q_ref| 0.046 rad
    #     6.0          x            -98 m      reaction 49 N on a 54 N vehicle
    #     2.5          x -11.70 .. +12.85 m    1.716 rad
    #     1.5          x  -4.85 .. + 0.82 m    1.206 rad
    #     1.0          x  -0.18 .. + 0.19 m    0.322 rad
    #     0.5          x  -0.08 .. + 0.08 m    0.160 rad
    #
    # The transition is sharp between 1.0 and 1.5, and below it the arm damps the
    # vehicle rather than throwing it: at 1.0 the vehicle holds station better
    # with the arm running than with it switched off.
    #
    # 1.0 is not enough, and only a long run shows it. At 1.0 the sweep numbers
    # above hold for about 40 s of simulated time and then the run departs: 42 to
    # 56 s across the four levels, with no ordering by difficulty. Nothing grows
    # and nothing drifts -- the amplitudes are flat for 35 s and then it goes --
    # so it is escape from a limit cycle, not an instability. The arm does not
    # widen the vehicle's attitude oscillation, it narrows it, but it triples the
    # torque activity needed to hold it: `tau_y` standard deviation 2.10 N m
    # against 0.88 with the arm off. That is the amplitude that eventually
    # escapes.
    #
    # At 0.5 it does not. Every level runs its full window with no departure, and
    # the worst position error is smaller than with the arm switched off:
    #
    #     livello   durata   err max   IAE pos [m s]   IAE giunti [rad s]
    #     hover      49.9 s   0.06 m       0.87              9.86
    #     easy       55.2 s   0.07 m       1.06             13.64
    #     medium     45.3 s   0.21 m       3.89             16.99
    #     hard       57.1 s   0.68 m      19.98             34.63
    #     arm off    64.8 s   0.37 m         --                --
    #
    # The cost is joint tracking, and it is a static limit rather than a control
    # one: holding joint 1 out costs about 2.3 N m per radian, so 0.5 N m holds
    # roughly 0.2 rad. `easy` asks 0.15 rad and is inside it; `hard` asks 0.5 and
    # is not, which is most of its joint IAE.
    #
    # ponytail: a magnitude cap where the problem is really the rate. The arm
    # reaches its full torque inside one control period, so a rate limit on the
    # commanded torque would buy the same calm without giving up the static
    # authority. Worth trying before widening this back.
    # Swept against Section V's own reference, which asks all three joints to
    # step to 0.33 rad. Holding that pose costs 1.153 N m at the shoulder, so
    # 0.5 cannot hold it statically -- and giving the arm enough to hold it makes
    # everything worse rather than better:
    #
    #     budget   fuga    buono   err posizione   err giunti  (riferimento 0.318)
    #     0.5      62 s    46 s      0.608 m        0.355 rad
    #     1.0      37 s    21 s      0.583 m        0.468 rad
    #     1.2      31 s    15 s      0.805 m        0.537 rad
    #
    # The arm is not tracking its reference at any of these. It is oscillating
    # inside a limit cycle driven by the vehicle's own motion, and more authority
    # only widens that oscillation, which is why less torque tracks better. The
    # open problem is the limit cycle, not this number.
    # That sweep is void and is kept only so it is not repeated as if it were
    # evidence. It was taken with the world running from launch, which means the
    # vehicle was on the ground with the arm folded against the floor for every
    # row of it, on a 10 ms period the node could not hold, with a clamp that was
    # itself driving the joints into their stops. The "limit cycle driven by the
    # vehicle's own motion" it concludes with was the floor.
    #
    # Measured again, on the rigid-base test where the vehicle cannot contribute
    # anything and only the arm's own loop is on trial. Same controller, same
    # reference, only this number changing:
    #
    #     budget    error vs the 0.33 rad amplitude    steps at the limit
    #     0.5 N m              73 %                          64 %
    #     1.5                  47 %                          12 %
    #     2.5                  23 %                           0 %
    #     4.0                   3 %                           0 %
    #
    # So the manipulator was never failing to track: it was starved. Holding
    # joint 1 at 0.33 rad costs 1.153 N m and it was given 0.5, so the reference
    # asked for a pose the budget forbids. The XACRO gives the actuator 6.0.
    #
    # Note the row at 2.5: nothing saturates and the error is still 23 %. That
    # part is the tube, whose input_margin is 1.0 N m -- twice the entire budget
    # it was tightening. The margins were sized for a much larger authority than
    # the limit allowed.
    #
    # This was held back to 1.5 for a while, on a measurement that said 4.0 threw
    # the vehicle. That measurement was taken before the loop was driven by the
    # simulator and before the attitude error was geometric, so it came off a rig
    # whose runs diverged for reasons that had nothing to do with the arm.
    # Re-measured on the current one, 4.0 costs the vehicle nothing -- 0.207 m on
    # `nominal` against 0.207 at 1.5, peak pitch 0.076 rad either way -- and takes
    # the joints from 28 % of their reference amplitude to 4 %, which is inside
    # the 2-8 % band the paper reports for its own controller.
    #
    # It also improved the vehicle on `fast`, 18.1 % to 13.0 %, and that part is
    # worth understanding: an arm that follows its reference delivers a reaction
    # the decomposition can predict, while a starved arm delivers whatever the
    # vehicle's own motion shakes out of it. More torque from a tracking arm is
    # easier to carry than less from a flailing one.
    #
    # The cross-coupled feedforward is what makes it carryable, and at this budget
    # that is measurable rather than assumed. Same 4.0 N m, `nominal`, two runs
    # each, with UAM_NO_COUPLING=1 as the only difference:
    #
    #                          with coupling      without
    #     vehicle error          0.226 m          0.667
    #     joint error            0.011 rad        0.570
    #     joint correlation      1.00/0.99/1.00   0.00/0.35/0.92
    #     peak pitch             0.076 rad        0.159
    #
    # Fifty times better at the joints and three at the vehicle. That is Eq. (10)
    # and (11) earning their place, and it is the paper's central claim tested
    # directly.
    joint_torque: float = float(os.environ.get('UAM_JOINT_TAU', 4.0))

    # How fast that torque may change, which is the constraint that actually
    # separates the two ends of this system -- see ManipulatorTubeMPC for why the
    # magnitude limit above cannot satisfy both at once. 30 N m/s reaches the
    # XACRO's full 6 N m in 0.2 s, while the reference asks for 0.83 rad/s at its
    # steepest, so this sits well clear of what tracking needs. 0 disables it.
    joint_torque_rate: float = 0.0


    def torque_vector(self) -> np.ndarray:
        """Per-axis rotor torque authority [roll, pitch, yaw]."""
        return np.array([self.torque, self.torque, self.torque_yaw])


@dataclass(frozen=True)
class LpvBounds:
    """Bounds on the rate of variation of the scheduled parameters.

    Assumption 1(iii) requires a compact set for Delta_theta. These enter the
    disturbance set of Eq. (30) and therefore set the width of the robust tube.
    They are expressed as a relative bound: the per-step change of each entry of
    the LPV matrices is bounded by `relative` times the magnitude of that entry,
    plus an absolute floor so that near-zero entries still carry some
    uncertainty.
    """

    # Measured, not guessed. At 0.25 the tightened input set of Eq. (46) is
    # already empty at the first horizon step: the yaw channel of the rotational
    # controller has -0.286 N m of room left, so the solver has nothing to
    # choose from and returns the bound. That is the tau_z pinned at exactly
    # +/-1.000 N m on every sample, flipping sign, that drove rotors to zero in
    # an alternating 1-3-5 / 2-4-6 pattern.
    #
    # A sweep puts the rotational controller back in business at 0.01, where the
    # worst remaining margin over the horizon is +0.570 N m.
    #
    # It does NOT rescue the manipulator, and no value does: at 0.0001, 2500
    # times smaller, its worst margin is still -41307 N m. There the reachable
    # set itself is the problem, growing from 14 to 108197 over five steps
    # because |A + B K| compounds -- the LQR gain is large since B is small on
    # the arm's stiff modes. That needs the disturbance evaluated along the
    # predicted trajectory instead of at the corner of the constraint box.
    relative: float = 0.01
    absolute_eta: float = 0.005
    absolute_gamma: float = 0.005


UAV = UavParams()

#: The arm the whole package uses.
#:
#: There used to be a second set here, `GAZEBO_ARM`, holding the isotropic
#: `ixx = iyy = izz = 0.166` that `hexacopter_sim/model/robot.xacro` once
#: declared -- a placeholder equal to the link mass, 110 times the physical
#: value. The XACRO now declares `ixx = iyy = 15e-4`, `izz = 55e-6` in an
#: `<inertial>` frame rotated by `rpy = (-pi/2, pi, -pi)`, which maps to
#: `diag(15e-4, 55e-6, 15e-4)` in the link frame: the small moment about the
#: link's own axis (+y there, +x under Craig's convention used here). Gazebo
#: and this file now integrate the same physics, so the second set and its
#: `UAM_ARM` switch are gone. If they ever diverge again, compare in the link
#: frame -- the XACRO numbers are not stated in it.
ARM = ArmParams()
ROTORS = RotorGeometry()
CONTROL = ControllerParams()
LIMITS = Constraints()
LPV = LpvBounds()

# Two different masses appear in the paper's equations and both are written `m`,
# which is worth being explicit about because swapping them double-counts the
# arm and puts the hover thrust about 10 % high.
#
# TOTAL_MASS is the mass of the whole vehicle. It belongs in Eq. (6c), where
# v_dot_0 = [0, 0, f_z / m] is the proper acceleration of the *system* under the
# rotor thrust, which is what drives the manipulator recursion.
#
# UAV.mass is the mass of the airframe alone, and it is the one that belongs in
# Eq. (3a) and (12a). Those equations already carry the arm's contribution
# through the reaction force f_arm_rea, so dividing by the combined mass would
# count the arm twice.
#
# Check at hover: the arm pulls down on the base with -m_arm g = -4.88 N, the
# rotors produce m_total g = 53.88 N, and the airframe balances at
# (53.88 - 4.88) / 5.0 = 9.8 m/s^2 = g. Using the combined mass instead leaves a
# residual of -0.89 m/s^2 and the vehicle slowly sinks.
TOTAL_MASS = UAV.mass + ARM.n_links * ARM.mass
