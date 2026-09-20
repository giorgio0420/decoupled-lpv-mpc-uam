#!/usr/bin/env python3
"""Phase 3: Gazebo is the plant, Python is the controller.

Closes position and attitude only. The arm is fixed in the model for now, so the
cross-coupled terms are evaluated at rest and the manipulator MPC is not run.

State comes from /uav/odom, which the rotor plugin publishes in the link frame --
the same frame it applies the wrench in. Going through /world/empty/dynamic_pose
instead would need a 180 degree correction and numerical differentiation for the
velocities; the plugin already holds both, so it just sends them.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray

from uam_control.controllers import (
    AttitudeReference, ErtfBaseline, ManipulatorTubeMPC, PositionPD,
    RotationalTubeMPC, TranslationalMPC, attitude_rate_command,
    attitude_reference,
)
from uam_control.coupling import CrossCoupling, decompose, max_joint_acceleration
from uam_control.newton_euler import joint_accelerations, mass_matrix
from uam_control.params import (
    ARM, CONTROL, GRAVITY, LIMITS, ROTORS, TOTAL_MASS, UAV,
)
from uam_control.trajectory import (
    SCENARIOS, translational_reference, joint_reference as paper_joint_reference,
    horizon_reference as joint_horizon, disturbance,
)

# Per-axis, not one number for all three. Yaw is produced by rotor drag, roll and
# pitch by differential thrust, and on this airframe k_m / k_f = 0.02 -- so the
# same angular error costs fifty times more rotor spread about z than about x or
# y. With a common gain the yaw term alone drove rotors to zero in an alternating
# 1-3-5 / 2-4-6 pattern, which is the drag axis' signature, and took thrust and
# attitude down with it. Yaw is also the axis nothing here needs to be quick on.
# Which of Section V's scenarios to fly, plus a `hover` that holds everything
# still so the vehicle's own loops can be told apart from a tracking error.
#
#   hover        nothing moves
#   nominal      sinusoid in x and y, ramp in z, joints stepping 0.33 rad at 5 s
#   fast         amplitude doubled, frequency up 50 %, joints 0.55 rad
#   disturbance  rectangular path with wind impulses at 7.5, 15 and 22.5 s
#
# The shapes live in `trajectory.py` and were written from the paper; this node
# used to import them and then fly a cosine of its own instead.
SCENARIO = SCENARIOS[os.environ.get('UAM_SCENARIO', 'nominal')]
EASE_TIME = 6.0                # [s] blend from rest onto the trajectory

# Attitude error [rad] -> commanded euler rate [rad/s]. This is what sets the
# vehicle's angular acceleration, and that acceleration is what the arm has to
# hold itself against: 0.6 N m at joint 1 per rad/s^2, against 2.5 N m of joint
# authority, so anything past about 4 rad/s^2 saturates the arm. Left as a knob
# because the right value is a measurement, not a derivation.
# 6.0, measured, with the position bandwidth at 1.5 and the loop at 20 Hz. The
# whole cascade has to stay separated -- position 1.5, attitude 6, rate K/I =
# 16.5 rad/s -- and this is the middle rung. Swept on the nominal scenario, mean
# tracking error:
#
#     gain      4.0     6.0     9.0     12.0
#     error    1.31    0.99    3.83   216.4
#
# 6.0 is the best single number and the wrong choice. Nine scenario runs at 6.0
# came out bimodal: six tracked to 0.9-1.2 m and three left at 11, 64 and 95 m,
# every failure with joint 1 pinned at its 1.5708 stop and the allocation giving
# up two thirds of the commanded torque. The margin to divergence is only 1.5x,
# and what crosses it is not noise in the gain but the arm: 6.0 against a 0.3 rad
# error asks 1.8 rad/s, which the 16.5 rad/s rate loop delivers as about 30
# rad/s^2, and past roughly 4 rad/s^2 the shoulder cannot hold itself. More joint
# authority does not buy it back -- measured at 1.5 and 2.5 N m, the error got
# worse, 1.43 and 1.73 against 1.16.
#
#     gain      4.0     6.0     9.0     12.0
#     error    1.31    0.99    3.83   216.4
#
# So 4.0: a third worse when both work, and it works every time. Room above it
# comes from a faster tick, which raises the rate loop, not from this number.
ATTITUDE_GAIN = np.array(
    [float(os.environ.get('UAM_ATT_GAIN', 4.0))] * 2 + [0.05])
# Stiffness used only when the arm is disabled, to hold it at the hanging pose so
# the run measures the vehicle carrying a rigid mass.
#
# One gain per joint cannot work here, and that is not a tuning opinion. The
# mass matrix at the hanging pose has diagonal
#
#     [0.16268, 0.04819, 0.00602]  kg m^2
#
# so the same damping coefficient is twenty-seven times stiffer, in sampled
# terms, at the wrist than at the shoulder. A discrete damper needs
# KD dt / I < 2 to decay at all, and the flat KD = 2.0 that was here gave
# [0.123, 0.415, 3.323]: stable at the first two joints and divergent at the
# third. It was measured doing exactly that -- the joints oscillated at about
# 10 Hz with growing amplitude for a second and then, inside one 10 ms sample,
# went from [0.12, -0.17, -0.71] to [-1.571, 3.142, 2.534], which is every joint
# against its stop. The stop impact put 7 rad/s of body rate into the vehicle and
# the attitude never came back. The run that produced this trace was the one
# meant to prove the vehicle could not hold attitude; what it proved is that this
# clamp cannot hold the arm.
#
# So: one bandwidth, and the gains follow the inertia. Critical damping at
# OMEGA gives KD dt / I = 2 OMEGA dt at every joint alike, which at 15 rad/s and
# 10 ms is 0.30 -- well inside the limit for all three, by construction rather
# than by luck.
# The bandwidth is derived from the period rather than written down, for the
# reason above: a number that is safe at one sample rate is not at another, and
# this clamp has already been the cause of one wrong conclusion. 2 OMEGA dt is
# 0.30 whatever dt_gamma becomes.
# 2 OMEGA dt = 0.8, so the same fraction of the stability limit at every joint and
# at any dt. 0.30 was tried first and is too soft to be a clamp: the arm swung
# with the vehicle and still reached its stops during flight, which is the thing
# this exists to prevent.
CLAMP_OMEGA = 0.4 / CONTROL.dt_gamma  # [rad/s]
_CLAMP_INERTIA = np.diag(mass_matrix(np.zeros(ARM.n_links)))
CLAMP_KP = _CLAMP_INERTIA * CLAMP_OMEGA ** 2
CLAMP_KD = 2.0 * _CLAMP_INERTIA * CLAMP_OMEGA
# The joint's own authority, not a fraction of it. A clamp that saturates is a
# bang-bang controller, and at 1.5 N m against these gains it saturated on any
# swing past 0.14 rad. Holding the arm still is the entire purpose of the arm-off
# configuration, so it gets what the actuator has.
CLAMP_LIMIT = 6.0  # [N m], the XACRO effort
# Close attitude only, level, at hover thrust, with the position loop bypassed.
ATTITUDE_ONLY = os.environ.get('UAM_ATTITUDE_ONLY', '0') == '1'
# Every reaction term forced to zero, so no path from the arm reaches any loop.
#
# The attitude test alone was not the isolation it looked like: the reaction was
# still entering the rotational MPC through its LPV model and the manipulator's
# through omega_dot, so a failure there could still be the coupling rather than
# the rate loop. With this set the vehicle is a plain hexarotor carrying a dead
# mass. If it cannot hold level even then, the defect is in the rotational loop
# or the allocation and the whole decomposition is innocent.
NO_COUPLING = os.environ.get('UAM_NO_COUPLING', '0') == '1'
ZERO_COUPLING = CrossCoupling(
    M_d=np.zeros((3, 3)), M_c=np.zeros((3, 3)), M_s=np.zeros((3, 3)),
    M_l=np.zeros((3, 3)), tau_bar=np.zeros(3), M_f=np.zeros(3),
    f_bar=np.zeros(3))


def quat_to_euler(x, y, z, w):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def quat_to_matrix(x, y, z, w):
    """Body-to-world rotation, the same sense the plugin's wrench uses."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# Where the vehicle's centre of mass sits in the link frame that /uav/odom
# reports, read from the plugin's own diagnostic:
#
#     [ROTORDIAG] link origin  = 0 0 5
#     [ROTORDIAG] link CoM     = 1e-06 -0 7
#
# Two metres. The whole airframe is built that far above its link origin in the
# XACRO -- rotors at z = 2.021, inertial origin at z = 2.000 -- so the point the
# odometry publishes is two metres below the aircraft.
#
# Regulating that point is what made the horizontal loop positive feedback. Tilt
# the body by +theta and a point 2 m below the axis swings to -2 theta, while the
# thrust tilts toward +x: to move right, the measured position must first go left.
# Measured open loop with tau_y = +0.3 N m, the match is exact --
#
#     t = 0.011 s   pitch +0.000091   x -0.000181   -2 pitch = -0.000181
#     t = 0.101 s   pitch +0.008264   x -0.016443   -2 pitch = -0.016528
#
# -- and the correlation between pitch and measured x is -0.997 at the link
# origin against +0.962 once transformed here. The position loop was asking for
# force one way and being shown motion the other, which is why no gain, weight,
# torque limit or tilt cap ever helped: on a non-minimum-phase response the gain
# only sets how fast it diverges.
#
# ponytail: this is the link's own centre of mass, not the composite one. The arm
# pulls the true centre down to about 1.955 m, leaving a 4.5 cm lever instead of
# 200. Track the composite one from the arm state if that residual ever matters.
LINK_TO_COM = np.array([0.0, 0.0, 2.0])


class HoverNode(Node):
    def __init__(self):
        super().__init__('uam_hover')
        self.declare_parameter('target_z', 1.0)
        # Fly with the arm hanging passive, joint torques held at zero.
        # Splits two candidates that the coupled run cannot tell apart:
        # whether the tracking error comes from the arm's reaction, or from
        # Gazebo's physics differing from the model the offline tests use.
        self.declare_parameter('arm_enabled', True)
        self.arm_enabled = bool(self.get_parameter('arm_enabled').value)
        self.target = np.array([0.0, 0.0,
                                self.get_parameter('target_z').value])

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)
        self.body_rate = np.zeros(3)
        self.have_state = False
        self.yaw_reference = None
        self.origin = None
        # Let the vehicle climb and settle before the trajectory begins.
        self.trajectory_start = 8.0
        self.last_thrust = TOTAL_MASS * GRAVITY

        # Exposed as a knob because the right value is now measurable. The nine
        # scenario runs are all stable and none of them tracks: the error scales
        # with the trajectory's frequency, and z -- the one axis with a 2.0 rad/s
        # bandwidth instead of 0.8 -- has an IAE of 1.2 m s against 40 on x and y.
        # Same trajectory, same loop, different bandwidth. That is phase lag, not
        # a control fault, and the fix is the number itself.
        #
        # Bandwidth 1.5 rad/s, measured on the nominal scenario against the
        # attitude gain of 6.0: mean error 2.10 m at 0.8, 1.57 at 1.2, 0.99 at
        # 1.5, 1.10 at 2.0. The old comment in PositionPD warns that 1.5 gave a
        # limit cycle, and it did -- against an attitude gain of 2.0, a separation
        # of 1.3x. The separation is what mattered, not the number.
        #
        # UAM_TRANS_MPC=1 puts the paper's own outer stage here instead, the
        # constrained linear MPC of Eq. (13)-(18). The PD was a stand-in, written
        # because the MPC is the least distinctive of the three stages, and the
        # substitution deserves a measurement rather than an argument: the
        # augmented form of Eq. (16) carries integral action, the PD does not, and
        # what is left on x and y is a standing lag. Both return the same quantity,
        # the total inertial force of Eq. (14), so this is the only line that
        # changes.
        #
        # TOTAL_MASS, not the class default of UAV.mass: the aircraft carries the
        # arm, and 5.0 against a real 5.498 is a 10 % error in the commanded force.
        if os.environ.get('UAM_TRANS_MPC') == '1':
            self.pd = TranslationalMPC(mass=TOTAL_MASS)
            self.get_logger().info('stadio traslazionale: MPC lineare vincolato')
        else:
            self.pd = PositionPD(
                bandwidth=float(os.environ.get('UAM_POS_BW', 2.0)))
        # UAM_USE_ERTF=1 flies Section V's baseline instead of the two tube
        # controllers, so the paper's headline comparison -- Table III, LPV-MPC
        # against Estimating-Reaction-Torque-and-Force -- can be run as two
        # scenarios differing in one variable.
        #
        # The baseline has no tube and no nominal state, so every place that
        # touches those has to know which scheme is running. That is what the
        # `use_ertf` guards below are for; there is no third case.
        self.use_ertf = os.environ.get('UAM_USE_ERTF', '0') == '1'
        if self.use_ertf:
            self.baseline = ErtfBaseline()
            self.rotational = None
            self.manipulator = None
            self.get_logger().info('schema di controllo: baseline ERTF')
        else:
            self.baseline = None
            self.rotational = RotationalTubeMPC()
            self.manipulator = ManipulatorTubeMPC()
            self.get_logger().info('schema di controllo: LPV-MPC con tubo')
        self.joints = np.zeros(ARM.n_links)
        self.joint_rates = np.zeros(ARM.n_links)
        self.joint_target = None
        # arm is fixed: evaluate the coupling once, at rest
        z3 = np.zeros(3)
        self.coupling = decompose(np.zeros(ARM.n_links), np.zeros(ARM.n_links),
                                  np.zeros(ARM.n_links), z3, z3,
                                  TOTAL_MASS * GRAVITY)
        self.allocation = np.linalg.pinv(ROTORS.allocation_matrix())
        self.hover_squared = None

        self.create_subscription(Odometry, '/uav/odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joints,
                                 qos_profile_sensor_data)
        self.reference_marker = self.create_publisher(
            Point, '/uav/reference', 1)
        self.rotors = self.create_publisher(Float64MultiArray, '/rotor_speeds', 10)
        self.joint_torques = self.create_publisher(Float64MultiArray, '/joint_torques', 10)
        # Eq. (50)'s wind impulses. Only `disturbance` scenario has nonzero
        # values here (trajectory.disturbance returns zero otherwise), so this
        # is silently a no-op on every other scenario and safe to publish
        # unconditionally every tick.
        self.disturbance_pub = self.create_publisher(
            Float64MultiArray, '/uav/disturbance', 1)
        # No timer. The simulator drives the controller.
        #
        # A ROS timer runs on the wall clock and the physics runs on its own, so
        # the two only agree by luck. Every timing pathology in this project came
        # from that: a node believing it ran at 40 Hz while running at 18.7, a
        # simulation advancing 20 ms against an assumed 25, a gain designed for
        # 10 ms closing a loop at 25 to 45, and stalls of 450 to 960 ms during
        # which the last command stayed latched on the plant.
        #
        # Odometry is published by the simulator, one message per physics batch,
        # carrying the simulated timestamp. Ticking from it makes the control
        # period exact in simulated time by construction rather than by
        # measurement: the loop runs when the simulated clock says it should, at
        # whatever wall-clock rate the machine manages. That is what
        # `sysCall_actuation` gives a CoppeliaSim script -- the simulator calls the
        # controller, one step at a time -- and it is the closest equivalent
        # available here without writing the controller as a Gazebo plugin.
        #
        # The decimation is on simulated time and not on message count, so a
        # change in the physics rate does not silently change the control period.
        self._next_tick_stamp = None
        self.ticks = 0
        self._engage_stamp = None
        self._last_joint_tau = np.zeros(ARM.n_links)
        self.odom_stamp = 0.0
        self._last_stamp = 0.0
        self._prof = {k: 0.0 for k in
                      ('decompose', 'posizione', 'rotazionale',
                       'allocazione', 'manipolatore', 'traccia')}
        self._prof_n = 0
        self._wall0 = time.perf_counter()
        self.scenario = SCENARIO
        self.get_logger().info(f"scenario '{self.scenario.name}'")
        self._trace = open('/tmp/trace.csv', 'w')
        self._trace.write(
            't,x,y,z,ref_x,ref_y,ref_z,'
            'pitch,pitch_cmd,q_cmd,q_meas,tau_y,sfit,'
            'omega_min,thrust,vx,force_x,force_z,fbar_x,'
            'q0,q1,q2,qref0,qref1,qref2,'
            'tau_y_real,thrust_real,tau_x_real,tau_x,wall,sim,tick_ms,'
            'roll,roll_cmd,p_meas,p_cmd,yaw,yaw_cmd,dist_on\n')
        self.get_logger().info(f'holding {self.target}')

    def on_odom(self, m):
        p, t = m.pose.pose, m.twist.twist
        self.odom_stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.attitude = quat_to_euler(p.orientation.x, p.orientation.y,
                                      p.orientation.z, p.orientation.w)
        self.body_rate = np.array([t.angular.x, t.angular.y, t.angular.z])

        # Move the measurement from the link origin to the centre of mass. See
        # LINK_TO_COM: the published point is two metres below the aircraft, and
        # regulating it inverts the sign of the horizontal response.
        rotation = quat_to_matrix(p.orientation.x, p.orientation.y,
                                  p.orientation.z, p.orientation.w)
        offset = rotation @ LINK_TO_COM
        self.position = np.array([p.position.x, p.position.y, p.position.z]) + offset
        # The velocity of that point carries the rigid-body term as well. The
        # plugin publishes twist.angular in the body frame, so it is rotated out
        # before taking the cross product with a world-frame lever.
        self.velocity = (np.array([t.linear.x, t.linear.y, t.linear.z])
                         + np.cross(rotation @ self.body_rate, offset))
        if self.yaw_reference is None:
            # The model spawns with a non-zero yaw: robot.xacro carries
            # rpy="-3.14 3.14 -3.14" on its origins. Regulating to world zero
            # would demand a 180 degree turn on the first tick, and yaw is the
            # weakest axis on this airframe -- k_m / k_f is 0.02, so the torque
            # that implies saturates the allocation and destroys roll and pitch
            # with it. Hold whatever yaw it started at.
            self.yaw_reference = self.attitude[2]
            self.get_logger().info(f'yaw at rest = {self.yaw_reference:+.4f} rad')
            self.origin = None  # latched later, once actually at rest
        self.have_state = True

        # Run the loop from here, on the simulated clock. The schedule is absolute
        # rather than incremental: the next due time advances by whole periods from
        # the previous due time, so a tick that arrives late does not push the
        # following ones late with it, and the average period stays exactly
        # dt_eta in simulated time. If the simulator has jumped by more than one
        # period, the schedule is resynchronised instead of firing repeatedly to
        # catch up -- a burst of ticks on one measurement is not control, and the
        # tube would see a series of zero-length steps.
        if self._next_tick_stamp is None:
            self._next_tick_stamp = self.odom_stamp + CONTROL.dt_eta
            return
        if self.odom_stamp + 1e-9 < self._next_tick_stamp:
            return
        overdue = self.odom_stamp - self._next_tick_stamp
        if overdue > CONTROL.dt_eta:
            self._next_tick_stamp = self.odom_stamp + CONTROL.dt_eta
        else:
            self._next_tick_stamp += CONTROL.dt_eta
        self.tick()

    def on_joints(self, m):
        if len(m.position) >= ARM.n_links:
            self.joints = np.array(m.position[:ARM.n_links])
        if self.joint_target is None:
            # The hanging pose, which is q = 0 since the joint origins were
            # rebuilt: links collinear, 0.990 m of reach, tip straight below the
            # shoulder. Verified in flight -- released with the vehicle airborne
            # the arm settles there on its own.
            #
            # Not the pose measured at rest. The vehicle rests sitting on the
            # floor, and there the arm is not hanging but crushed against the
            # ground, with joints 2 and 3 pushed into their stops at +/-pi.
            # Latching that reads [-0.001, -3.142, 3.031] and asks the MPC to
            # hold the arm folded into a mechanical limit for the whole run.
            self.joint_target = np.zeros(ARM.n_links)
            self.get_logger().info(
                'arm reference anchored at the hanging pose q = 0; '
                'measured rest pose was %s' % np.round(self.joints, 3))
        if len(m.velocity) >= ARM.n_links:
            self.joint_rates = np.array(m.velocity[:ARM.n_links])

    def _try_capture_origin(self):
        """Latch the starting pose, but only from a straight arm.

        The simulation is held paused until this succeeds, so the pose latched
        here is the spawn pose: vehicle level, arm hanging straight at q = 0.

        The check is not decoration. Runs that latched a folded arm produced
        entirely different behaviour from runs that caught it straight, and two
        such runs cannot be compared -- which invalidated a bandwidth
        measurement that looked conclusive at the time. Refusing to start is
        better than producing a run whose conclusion has to be withdrawn later.
        """
        if self.origin is not None:
            return True
        folded = float(np.abs(self.joints).max())
        if folded > 0.05:
            # Drive it straight rather than waiting for gravity to do it. Waiting
            # was a coin toss: q = 0 is the equilibrium, but the arm is released
            # from the built pose with the world already running, and whether it
            # settles there or swings past -pi and stays folded is a race with how
            # long the launch takes. Measured, one run in three engaged; the other
            # two sat at [0, -pi, 3.03] until the timeout and produced a header
            # and nothing else. The same clamp that holds the arm during an
            # arm-off run brings it there, so this adds a use of existing gains
            # rather than a mechanism.
            self.joint_torques.publish(Float64MultiArray(data=np.clip(
                -CLAMP_KP * self.joints - CLAMP_KD * self.joint_rates,
                -CLAMP_LIMIT, CLAMP_LIMIT).tolist()))
            self._rejects = getattr(self, '_rejects', 0) + 1
            if self._rejects % 40 == 1:
                self.get_logger().warn(
                    'raddrizzo: braccio a %s, |q| = %.3f > 0.05'
                    % (np.round(self.joints, 3), folded))
            return False
        self.origin = self.position.copy()
        self.joint_target = np.zeros(ARM.n_links)
        self.get_logger().info(
            'ENGAGED z=%+.3f arm=%s |q|=%.4f -- straight, run is comparable'
            % (self.origin[2], np.round(self.joints, 3), folded))
        return True

    def tick(self):
        tick_t0 = time.perf_counter()
        if not self.have_state:
            return
        if not self._try_capture_origin():
            return

        # Use the simulated time that actually passed, not the period the timer
        # was asked for. Measured on this machine the node runs at 18.7 Hz of wall
        # clock while believing it runs at 40, and the simulation advances 20 ms
        # per tick against the 25 the model assumes -- so every discretisation,
        # every nominal propagation and every reachable set was 25% wrong, all the
        # time, not now and then. The odometry carries a simulated timestamp; the
        # interval between two of them is the truth.
        gap = self.odom_stamp - self._last_stamp
        self._last_stamp = self.odom_stamp
        nominal_dt = CONTROL.dt_eta
        if gap > 3.0 * nominal_dt:
            # A stall, and they reach a second here. Propagating the tube across a
            # gap that size is worse than admitting the sample is lost: the
            # nominal state runs away from the real one, the tube error inflates,
            # and K @ e comes out as a burst of torque. This is the chain that
            # shows up in the window as "it freezes, then the vehicle spins".
            # Re-seeding costs one sample of tracking and keeps the tube honest.
            if not self.use_ertf:
                self.rotational.nominal_state = None
                self.manipulator.nominal_state = None
                self.get_logger().warn(
                    'salto di %.0f ms nel tempo simulato: tubi riseminati'
                    % (gap * 1e3))
            return
        dt = nominal_dt
        if gap > 1e-6:
            # The floor used to be half the nominal period, which quietly
            # rounded a real 10 ms step up to 12.5 and reintroduced the same 25%
            # error it was added to remove -- from the other side. The bound is
            # there to reject nonsense, not to enforce a rate, so it sits well
            # away from anything the simulation plausibly produces.
            dt = float(np.clip(gap, 0.002, 3.0 * nominal_dt))
            if not self.use_ertf:
                self.rotational.dt = dt
                self.manipulator.dt = dt
        self._current_dt = dt
        # The angular acceleration the arm is scheduled on. The MPC publishes the
        # value it has just solved for; the baseline has no such prediction, so it
        # carries forward what the rotors actually produced last tick.
        predicted_rate_dot = (
            self.baseline.previous_body_rate_dot if self.use_ertf
            else self.rotational.predicted_body_rate_dot)
        # The cross-coupled decomposition is what the paper is about, so it is
        # rebuilt every tick from the arm's measured state rather than frozen at
        # rest. The joint acceleration comes from the manipulator MPC's own
        # prediction: feeding back a differentiated measurement would close an
        # algebraic loop through the controller that produced it.
        #
        # It is passed unfiltered, and that was tested rather than assumed. A
        # low-pass on it cut the standard deviation of f_bar from 13.8 N to 2.5 N
        # and made the vehicle worse at every setting: |omega_dot| p95 went from
        # 20.4 to 41.7 rad/s^2 as the filter tightened. f_bar is not noise on the
        # measurement, it is feedforward for a push the arm really delivers, and
        # smoothing it means the rate loop has to chase that push after the fact.
        # With the arm disabled its torques are not published, so its acceleration
        # is the free response to gravity, not what the MPC predicts. The MPC is
        # still running and does not know its output is being thrown away: it saw
        # the passive arm swing to 2.5 rad, answered with an acceleration it was
        # never going to produce, and f_bar reached 29 kN against a 54 N vehicle.
        # Every arm-off comparison taken before this was contaminated by it.
        #
        # Zeroing it instead is worse, and that was measured too: the arm is half
        # a kilo of pendulum under five and a half of aircraft and its reaction is
        # real, so zero blinds the attitude loop to a disturbance that is still
        # arriving. The vehicle held station to 2 mm for twelve seconds and then
        # the swinging arm pitched it over 0.65 rad while it was being commanded
        # the other way.
        #
        # The free response is what the plant actually does with zero torque, and
        # `joint_accelerations` computes exactly that.
        # One path, not two, and it is the forward dynamics rather than the
        # optimiser's extrapolation: the acceleration the model says the arm is
        # producing under the torque that was actually sent last tick. That is the
        # same information the MPC's prediction was there to give, bounded by
        # physics instead of by a linearisation, and it stays correct whether the
        # arm is being driven by the MPC or held by the clamp. The model agrees
        # with Gazebo to within 3% on exactly this quantity, measured in free
        # fall, so it is not a guess.
        #
        # It is not the algebraic loop the earlier comment warned about: that one
        # came from differentiating the *measured* joint velocity, which reacts to
        # the torque inside the same sample. This is the response to a command
        # already committed.
        arm_acceleration = joint_accelerations(
            self.joints, self.joint_rates, self._last_joint_tau,
            self.body_rate, predicted_rate_dot,
            max(self.last_thrust, 1.0))
        # Bounded on both paths, for the same reason: this number becomes a force
        # on the vehicle through f_bar, and neither path is naturally bounded. The
        # free response of a chain with these inertias runs to hundreds of rad/s^2
        # once the joints are moving fast, and the MPC's extrapolation runs
        # further. A joint cannot gain more velocity in one period than its own
        # rate limit allows, so that ratio is the largest acceleration the plant
        # can actually deliver; past it the number describes a machine that is
        # not there.
        reachable = max_joint_acceleration(
            self.joints, self.joint_rates, self.body_rate,
            predicted_rate_dot, max(self.last_thrust, 1.0))
        arm_acceleration = np.clip(arm_acceleration, -reachable, reachable)
        _t = time.perf_counter()
        self.coupling = decompose(
            self.joints, self.joint_rates,
            arm_acceleration,
            self.body_rate, predicted_rate_dot,
            max(self.last_thrust, 1.0))
        self._prof['decompose'] += time.perf_counter() - _t
        if NO_COUPLING:
            # Computed and thrown away rather than skipped, so the tick costs the
            # same as a real one. Timing has been the difference between a stable
            # and an unstable run twice in this project; a cheaper tick here would
            # make the comparison meaningless.
            self.coupling = ZERO_COUPLING

        state = np.array([self.position[0], self.velocity[0],
                          self.position[1], self.velocity[1],
                          self.position[2], self.velocity[2]])
        # Section V trajectory: a 2.5 m circle in x and y over 15 s, plus a slow
        # ramp in z from 1.5 m to 2.6 m. Referred to the pose the vehicle rests
        # at, since the link frame reads about -1.01 m sitting on the floor
        # rather than zero.
        #
        # The height offset is the scenario's own r[4], not a hand-picked
        # constant. It was written as ,
        # which reduces to : half a metre below what the
        # paper asks for. Combined with an origin captured on the first odometry
        # message, before the pose had settled, the commanded altitude came out
        # at -0.98 against a resting -1.01 -- three centimetres of flight with a
        # 0.99 m arm underneath. The arm was pinned against the floor for the
        # whole run, which is why its joints never tracked and why the nominal
        # state ran away: the plant was in contact, and no torque the controller
        # commanded could move it.
        #
        # At the paper's 1.5 m the arm tip clears the ground by about 0.5 m.
        # Simulated time since the loop engaged, read from the odometry stamp
        # rather than counted in ticks. `ticks * dt_eta` is only the elapsed time
        # if every tick really took dt_eta, which was false by a factor of two to
        # four before the loop was driven by the simulator, and is still false
        # across a stall. The trajectory has to advance with the world, not with
        # the controller's opinion of how often it ran.
        if self._engage_stamp is None:
            self._engage_stamp = self.odom_stamp
        t = (self.odom_stamp - self._engage_stamp) - self.trajectory_start
        # A deliberately gentle pair of motions, to watch the whole loop work
        # before asking it for the paper's full scenario: the vehicle traces a
        # cosine in x at constant height, the arm a sine on all three joints.
        # Both smooth and slow, no steps and no ramp, so that whatever misbehaves
        # is the coupling rather than the reference.
        # Ease from where the vehicle actually is onto the trajectory.
        #
        # It settles on the ground and the trajectory lives 1.6 m above that, so
        # commanding it directly is a 1.6 m step at t = 0 on top of the cosine.
        # The position loop answers with a large tilt, the attitude loop chases
        # it, and the arm gets the whole transient through the reaction term
        # before it has ever tracked anything. Blending over EASE_TIME with a
        # smoothstep -- zero value and zero slope at both ends -- means the
        # commanded pose starts exactly at the measurement and arrives on the
        # trajectory with the velocity already matched.
        # Section V's own reference, from `trajectory.py`, instead of a shape
        # invented here. That module already carried all three of the paper's
        # scenarios -- sinusoid in x and y over a ramp in z, the doubled and
        # sped-up variant, and the rectangle with wind impulses -- and it was
        # imported and then ignored while a hand-written cosine was flown in its
        # place.
        #
        # Referred to the pose the vehicle actually holds. The paper's altitudes
        # are absolute, 1.5 m rising to 2.6 m, and the model spawns at 5 m, so
        # commanding them directly is a three-metre descent on top of the
        # trajectory. Only the horizontal shape and the climb rate are taken.
        # Before the trajectory starts the reference is a full stop, velocities
        # included. Section V's path is not at rest at t = 0 -- it opens with
        # a * w = 1.05 m/s along x -- so freezing the clock at zero while waiting
        # freezes that velocity too, and the vehicle spends the wait being told
        # to hold its position and to travel at a metre per second at the same
        # time. The position loop answers the velocity error, and it had left
        # before the run began.
        tt = max(t, 0.0)
        dist_force, dist_torque = disturbance(self.scenario, tt)
        self.disturbance_pub.publish(Float64MultiArray(
            data=[*dist_force.tolist(), *dist_torque.tolist()]))
        # Logged as a flag, not the six components, because what a plot needs
        # is "was it on", to shade the window it acted in -- not its exact
        # value, which the schedule already documents.
        self._log_disturbance = float(
            np.any(dist_force != 0.0) or np.any(dist_torque != 0.0))
        start = np.array([self.origin[0], 0.0, self.origin[1], 0.0,
                          self.origin[2], 0.0])
        origin = translational_reference(self.scenario, 0.0)

        def desired(when):
            """The blended reference at one instant, so the horizon can be built.

            The PD only ever needs `when = t`. The MPC needs the future, and it has
            to be the same reference the PD would have been given at each of those
            instants -- ease blend included, or the preview would disagree with the
            command in exactly the first few seconds where the blend is active.
            """
            if when < 0.0:
                return start
            paper = translational_reference(self.scenario, when)
            # Referred to the circle's centre, not to its value at t = 0.
            #
            # Section V's path is x = a sin(wt), y = a cos(wt): a circle, and
            # symmetric in amplitude and frequency by construction. Subtracting
            # paper(0) put the start at the circle's y extreme, so the commanded y
            # ran from the origin to origin - 2a while x ran from -a to +a about
            # it. Y therefore travelled twice the distance from the start that x
            # did -- 10 m against 5 on the `fast` scenario -- and the measured
            # error followed: |ex| 1.193 m against |ey| 6.351 m, a factor of five
            # on a trajectory with no asymmetry in it at all.
            #
            # Centring instead puts a step of a metres in y at t = 0, which is
            # exactly what the ease blend exists to absorb: it starts at the
            # measurement and arrives on the circle with the velocity matched.
            # Only the horizontal centre is taken from the vehicle; the altitude
            # still keeps its own offset, because the paper's z is a ramp rather
            # than an oscillation and its start is the pose to hold.
            target = np.array([
                self.origin[0] + paper[0], paper[1],
                self.origin[1] + paper[2], paper[3],
                self.origin[2] + paper[4] - origin[4], paper[5]])
            u = min(max(when / EASE_TIME, 0.0), 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            return start + blend * (target - start)

        target = desired(tt)
        # The ramp runs from the moment the trajectory starts, not from the start
        # of the run. Measured against `t + trajectory_start` it was already
        # fully spent by the time there was anything to ease onto, so the paper's
        # opening velocity of 1.05 m/s arrived as a step.
        # One row for the PD, the whole horizon for the MPC. `desired` already
        # applies the blend at whichever instant it is asked about.
        if isinstance(self.pd, TranslationalMPC):
            reference = np.array([desired(t + (k + 1) * CONTROL.dt_zeta)
                                  for k in range(CONTROL.N_zeta)])
        else:
            reference = np.array([desired(t)])

        _t = time.perf_counter()
        force = self.pd(state, reference)
        target = attitude_reference(force, self.coupling, 0.0, self.attitude)
        # The thrust demand needs the bound the constraint set already declares.
        # `attitude_reference` returns the magnitude of whatever force it was
        # handed, and with the arm's reaction in that sum it has been seen asking
        # for 80 kN from an aircraft that makes 135. Downstream the allocation
        # clips the rotors and delivers something else entirely, and no loop is
        # told. Clipping here keeps the command inside what the rotors can build.
        target = AttitudeReference(
            thrust=float(np.clip(target.thrust, LIMITS.force_z_min,
                                 LIMITS.force_z_max)),
            roll=target.roll, pitch=target.pitch)

        if ATTITUDE_ONLY:
            # Attitude alone, level, at hover thrust. No position loop and no
            # coupling correction reach the attitude command.
            #
            # The one test never run: every measurement so far has had position
            # and attitude closed together, so a failure in either looked like a
            # failure of the pair. If the vehicle cannot hold level here it is
            # the rotational loop, and the cascade is innocent; if it holds, the
            # cascade is where to look -- and then it is worth modelling, with
            # the law the controller actually uses rather than the one assumed.
            #
            # It will drift horizontally, with nothing regulating position. That
            # is expected and is not what is being measured.
            target = AttitudeReference(thrust=TOTAL_MASS * GRAVITY,
                                       roll=0.0, pitch=0.0)

        self._prof['posizione'] += time.perf_counter() - _t
        # Literal 3-state MPC (Eq. 22-23): the optimiser regulates body rate
        # only, not attitude, so the pointing command has to be turned into a
        # rate reference up here, same as the ERTF baseline already does below
        # with `attitude_rate_command` -- the SO(3) error, not the Euler one
        # that annihilates the pitch command at roll = +/-pi/2 (see that
        # function's docstring for the crash this avoids).
        _t = time.perf_counter()
        joint_state_full = np.concatenate([self.joints, self.joint_rates])
        # Both schemes are rate controllers now, so both need the SO(3) error
        # turned into a body-rate command up here.
        rate_command = attitude_rate_command(
            self.attitude, target.roll, target.pitch, self.yaw_reference,
            ATTITUDE_GAIN)
        if self.use_ertf:
            torque = self.baseline.rotational(
                self.body_rate, rate_command, joint_state_full,
                arm_acceleration, target.thrust, self._current_dt)
        else:
            rate_horizon = np.tile(rate_command, (CONTROL.N_eta, 1))
            torque = self.rotational(self.body_rate, rate_horizon, self.coupling)

        self._prof['rotazionale'] += time.perf_counter() - _t
        # Thrust first, torque scaled to whatever is left.
        #
        # pinv alone returns the minimum-norm omega^2, which is free to go
        # negative; clipping that at zero then silently delivers a different
        # wrench than the one asked for. It showed up as rotors alternating
        # 1-3-5 / 2-4-6 straight to zero, because yaw comes from rotor drag and
        # k_m / k_f is 0.02 here -- even a negligible tau_z demands an enormous
        # alternating split, and the clip ate the thrust along with it.
        #
        # Instead: put the thrust on all six equally, which is always feasible,
        # then add as much of the torque solution as keeps every rotor inside
        # its range. Torque direction is preserved, only its magnitude gives way.
        # Thrust on all six, then roll and pitch, then whatever yaw still fits.
        #
        # pinv alone returns the minimum-norm omega^2, which is free to go
        # negative; clipping that at zero quietly delivers a different wrench
        # than the one asked for. It shows as rotors alternating 1-3-5 / 2-4-6
        # straight to zero for a yaw error of a few hundredths of a radian,
        # because yaw comes from rotor drag and k_m / k_f is 0.02 here.
        #
        # The three torque axes cannot share one scale factor either: yaw needs
        # about fifty times the rotor spread of roll or pitch, so scaling the
        # whole torque vector by what yaw can afford zeroes roll and pitch with
        # it, and the vehicle loses attitude control entirely. Each group is
        # fitted against the headroom the previous one left, roll and pitch
        # first because they are what keeps it upright.
        _t2 = time.perf_counter()
        limit = ROTORS.max_omega ** 2
        base = np.full(6, target.thrust / (6.0 * ROTORS.thrust_coefficient))

        def fit(current, direction):
            """Largest s in [0, 1] keeping 0 <= current + s * direction <= limit."""
            s = 1.0
            for c, d in zip(current, direction):
                if d < -1e-12:
                    s = min(s, c / -d)
                elif d > 1e-12:
                    s = min(s, (limit - c) / d)
            return max(min(s, 1.0), 0.0)

        # ponytail: yaw torque forced to zero, not fixed. Upgrade path is the
        # tube feedback in RotationalTubeMPC, see below.
        #
        # tau_z came out of the MPC pinned at exactly +/-1.000 N m on every
        # sample, flipping sign each time. That is the axis' bound being hit,
        # not a control signal. The bound itself is right -- this airframe makes
        # 1.08 N m about z against 7.7 about x and y, because yaw comes from
        # rotor drag and k_m / k_f is 0.02. What is wrong is the demand: a
        # 0.05 rad yaw error asks for 0.0025 rad/s, which needs 0.03 N m with
        # I_zz = 0.3, and the optimiser asked for thirty times that. It is not
        # tracking the reference, so the suspect is the tube feedback term
        # K_eta @ (x - x_nominal) swamping u_nominal once the nominal state has
        # drifted.
        #
        # Yaw is passively stable here -- it never wandered past 0.07 rad even
        # while being driven bang-bang -- so commanding zero costs nothing and
        # takes the pathology out of the picture while roll and pitch are
        # judged. It has to come back before any yaw-holding manoeuvre.
        torque = np.array([torque[0], torque[1], 0.0])

        roll_pitch = self.allocation @ np.array([0.0, torque[0], torque[1], 0.0])
        scale_rp = fit(base, roll_pitch)
        squared = base + scale_rp * roll_pitch
        yaw = self.allocation @ np.array([0.0, 0.0, 0.0, torque[2]])
        squared = squared + fit(squared, yaw) * yaw
        omega = np.sqrt(np.clip(squared, 0.0, limit))
        self._prof['allocazione'] += time.perf_counter() - _t2

        # Logged because the position loop's sign was verified correct offline
        # while the vehicle flew the other way: the attitude actually reached has
        # to be readable next to the attitude asked for, and the torque next to
        # the fraction of it the rotors were able to take.
        self._log_target = np.array([target.roll, target.pitch])
        self._log_torque = torque.copy()
        self._log_scale = scale_rp
        self._log_rate_target = rate_command.copy()
        # What the rotors will actually produce, from the speeds about to be
        # published, against what was asked for. `fit` preserves the direction of
        # the torque but gives up its magnitude, and the final clip at zero can
        # give up more, so the realised wrench is not the commanded one and none
        # of the three loops is told. Reconstructed with the same allocation
        # matrix the command was built from, so any difference is the fitting and
        # the clipping alone -- not a modelling disagreement.
        realised = ROTORS.allocation_matrix() @ (omega ** 2)
        self._log_extra = (target.pitch, rate_command[1], torque[1], scale_rp,
                           omega.min(), target.thrust, force[0], force[2],
                           realised[2], realised[0], realised[1],
                           target.roll, rate_command[0])
        self._log_reference = reference[0][[0, 2, 4]]
        # The same point the trace records, sent to the plugin so the world draws
        # it. Until this existed nothing on screen said where the vehicle was
        # meant to be, and a run holding station to 12 cm looked exactly like one
        # drifting ten metres.
        #
        # Referred back to the link frame, because that is what Gazebo's world
        # coordinates are: `self.position` carries the 2 m offset onto the centre
        # of mass, and drawing the marker there would put it two metres above the
        # aircraft it is supposed to be marking.
        self.reference_marker.publish(Point(
            x=float(reference[0][0]), y=float(reference[0][2]),
            z=float(reference[0][4] - LINK_TO_COM[2])))

        self.rotors.publish(Float64MultiArray(data=omega.tolist()))
        self.last_thrust = target.thrust

        # Wrap the joint reference into the half turn around where the arm
        # actually is. These are revolute joints whose range spans a full turn,
        # so -pi and +pi are the same pose -- but the rest pose was captured as
        # -3.142 on joint 2 while the joint later reads +3.142, and the raw
        # difference is 2 pi of error for zero physical displacement. That alone
        # pins every joint at full torque. Same wrap the yaw channel needs, for
        # the same reason.
        # Fig. 7's reference: a periodic step of alternating sign, same amplitude
        # on all three joints, smoothed only enough to keep the commanded rate
        # finite. Also from `trajectory.py` rather than invented here -- the sine
        # that was flown before traced an arbitrary tip path and is not what the
        # paper asks the arm to do.
        # The whole horizon, rates included -- not one pose repeated N times with
        # a zero rate target.
        #
        # This was the manipulator's failure to track, and it was not the arm's
        # authority, its torque budget, the tube, or the vehicle's motion. On the
        # rigid-base bench, which builds this reference with `horizon_reference`,
        # the same controller follows the same waveform to 3 % of its amplitude.
        # In flight it was handed `tile(concat([pose, zeros]))`: be at 0.33 rad,
        # and be at rest, at every one of the four horizon steps, while the
        # reference was really moving at up to 0.83 rad/s. The optimiser split the
        # difference, and the measured joint angle correlated with its command at
        # -0.36, -0.15, -0.03, +0.10, +0.02 -- zero, at every torque budget tried.
        # The joints were moving because the vehicle shook them.
        #
        # A tiled constant is also the whole preview thrown away, which is the
        # only thing an MPC has over a PD on a reference that reverses every 5 s.
        n = ARM.n_links
        if t < 0.0:
            joint_reference = np.tile(
                np.concatenate([self.joint_target, np.zeros(n)]),
                (CONTROL.N_gamma, 1))
            desired_joints = self.joint_target
        else:
            joint_reference = joint_horizon(
                paper_joint_reference, self.scenario, tt,
                CONTROL.N_gamma, self._current_dt)
            joint_reference[:, :n] += self.joint_target
            desired_joints = joint_reference[0, :n]
        # The half-turn wrap still has to happen, for the reason it was added:
        # these joints span a full turn, the rest pose was captured as -3.142 on
        # joint 2 while the joint later reads +3.142, and the raw difference is
        # 2 pi of error for no physical displacement. Applied as one offset to the
        # whole horizon so the preview stays a trajectory rather than becoming a
        # sequence of independently wrapped poses.
        offset = (desired_joints - self.joints + np.pi) % (2 * np.pi) - np.pi
        joint_reference[:, :n] += (self.joints + offset) - desired_joints
        _t = time.perf_counter()
        if self.use_ertf:
            joint_tau = self.baseline.manipulator(
                joint_state_full, self.joints + offset, self.body_rate,
                max(target.thrust, 1.0), self._current_dt)
            # The baseline has no predicted acceleration of its own, so it carries
            # forward what the rotors actually delivered. Reconstructed from the
            # realised wrench rather than from the command, since `fit` gives up
            # torque magnitude and the two are not the same number.
            self.baseline.previous_body_rate_dot = np.linalg.inv(
                UAV.inertia_tensor()) @ (realised[1:] - self.coupling.tau_bar)
        else:
            joint_tau = self.manipulator(
                joint_state_full, joint_reference,
                self.body_rate, predicted_rate_dot,
                max(target.thrust, 1.0))
        self._prof['manipolatore'] += time.perf_counter() - _t
        if not self.arm_enabled:
            # Clamped, not free. Commanding zero torque leaves half a kilo of
            # pendulum swinging under five and a half of aircraft: it reached the
            # shoulder's stop at 1.571 rad, and the impact put a 229 N spike into
            # f_bar and threw the attitude. That is not a floor measurement, it is
            # a different experiment. Holding the joints at the hanging pose makes
            # the arm behave as the rigid mass it is meant to be for this test, and
            # costs almost nothing to do: q = 0 is the equilibrium, so gravity is
            # already balanced and the PD only has to reject what the vehicle's own
            # motion injects.
            joint_tau = np.clip(
                -CLAMP_KP * self.joints - CLAMP_KD * self.joint_rates,
                -CLAMP_LIMIT, CLAMP_LIMIT)
        self.joint_torques.publish(Float64MultiArray(data=joint_tau.tolist()))
        self._last_joint_tau = joint_tau.copy()

        # Every tick, not one in forty. The summary line below samples at 1 Hz,
        # which cannot show an oscillation whose period is a few samples, and it
        # is far too coarse to integrate an error against. Written straight to a
        # file so the ROS logger is not the bottleneck. Both references are here
        # alongside both measurements, so the IAE comes out of this file alone.
        (pitch_cmd, q_cmd, tau_y, sfit, om_min, thrust, fx, fz,
         tau_y_real, thrust_real, tau_x_real, roll_cmd, p_cmd) = self._log_extra
        row = (
            # Simulated time, for the same reason the trajectory uses it: a trace
            # whose time column is a tick count times a nominal period reports the
            # wrong instant for every sample after the first late tick, and every
            # IAE integrated against it is wrong with it.
            self.odom_stamp - self._engage_stamp,
            *self.position, *self._log_reference,
            self.attitude[1], pitch_cmd, q_cmd, self.body_rate[1], tau_y, sfit,
            om_min, thrust, self.velocity[0], fx, fz, self.coupling.f_bar[0],
            *self.joints, *(self.joints + offset),
            tau_y_real, thrust_real, tau_x_real, torque[0],
            time.perf_counter() - self._wall0, self.odom_stamp,
            1000.0 * (time.perf_counter() - tick_t0),
            self.attitude[0], roll_cmd, self.body_rate[0], p_cmd,
            self.attitude[2], self.yaw_reference,
            getattr(self, '_log_disturbance', 0.0),
        )
        _t = time.perf_counter()
        self._trace.write(",".join("%.5f" % v for v in row) + "\n")
        # Flushed once a second, not once a tick. Forcing 31 fields to disk forty
        # times a second for a diagnostic was costing part of a budget the loop
        # had already overrun: 53.5 ms of wall clock per tick against 25.
        if self.ticks % 40 == 0:
            self._trace.flush()
        self._prof['traccia'] += time.perf_counter() - _t
        self._prof_n += 1
        if self._prof_n == 200:
            parts = ' '.join('%s %.2f' % (k, 1000.0 * v / self._prof_n)
                             for k, v in self._prof.items())
            self.get_logger().info('[PROFILO ms] ' + parts + ' | totale %.2f'
                                   % (1000.0 * sum(self._prof.values())
                                      / self._prof_n))
            self._prof = {k: 0.0 for k in self._prof}
            self._prof_n = 0

        self.ticks += 1
        if self.ticks % 40 == 0:
            self.get_logger().info(
                f'pos={np.round(self.position,2)} '
                f'ref={np.round(reference[0][[0,2,4]],2)} '
                f'rpy={np.round(self.attitude,3)} '
                f'rp_cmd={np.round(self._log_target,3)} '
                f'tau={np.round(self._log_torque,2)} '
                f'sfit={self._log_scale:.3f} '
                f'rate_cmd={np.round(self._log_rate_target,3)} '
                f'r={np.round(self.body_rate,3)} '
                f'T={target.thrust:.1f} q={np.round(self.joints,3)} '
                f'qref={np.round(self.joints+offset,3)} '
                f'jtau={np.round(joint_tau,2)} '
                + ('[ERTF]' if self.use_ertf else
                   f'[MPC] nom={np.round(self.rotational.nominal_state,3)} '
                   f'qnom={np.round(self.manipulator.nominal_state[:3],3)} '
                   f'err={np.round(self.manipulator.last_error[:3],2)} '
                   f'Ke={np.round(self.manipulator.last_feedback,2)} '
                   f'unom={np.round(self.manipulator.last_nominal_input,2)} '
                   f'rpi_ok={self.rotational.terminal_feasible}')
                )


def main():
    rclpy.init()
    node = HoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
