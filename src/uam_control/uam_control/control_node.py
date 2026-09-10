"""ROS 2 node running Algorithm 1 against the Gazebo model.

Topic contract
--------------
subscribes
    /tf                 tf2_msgs/TFMessage      pose of hexacopter_robot::robot_base
    /joint_states       sensor_msgs/JointState  Joint_1..Joint_3
publishes
    /wrench_command     geometry_msgs/Wrench    force.z = T, torque = [tau_x, tau_y, tau_z]
    /joint_command      std_msgs/Float64MultiArray  three joint torques
    /uam_debug          std_msgs/Float64MultiArray  reference and error, for logging

`/wrench_command` is consumed by `hexacopter_control::allocation_node`, which
converts it into the six rotor speeds the Gazebo plugin applies.

State estimation
----------------
Gazebo publishes pose but not velocity, so linear velocity and body rates are
differentiated from successive poses and low-pass filtered. That is adequate in
simulation, where the pose is noise-free. On hardware this must be replaced by
an estimator fusing an IMU; the filter cutoff here is nowhere near tight enough
to survive real measurement noise.

Before this node can close the loop, three things must hold on the Gazebo side:
the rotor plugin must be loaded and spinning the six propellers, `/tf` must
carry the transform for `hexacopter_robot::robot_base`, and `/joint_states` must
be published for the three arm joints. The node reports which of those are
missing rather than publishing commands into the void.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Wrench
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from .controllers import (
    AttitudeReference,
    ManipulatorTubeMPC,
    RotationalTubeMPC,
    PositionPD,
    attitude_reference,
    euler_rate_to_body_rate,
)
from .coupling import decompose
from .params import ARM, CONTROL, GRAVITY, UAV
from .trajectory import SCENARIOS, horizon_reference, joint_reference, translational_reference

BASE_FRAME = "hexacopter_robot::robot_base"
JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3")
ATTITUDE_GAIN = 3.0
VELOCITY_FILTER = 0.25  # first-order blend on the differentiated velocities


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Roll, pitch and yaw in the Z-Y-X convention used by Eq. (1)."""
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return np.array([roll, pitch, yaw])


class UamControlNode(Node):
    def __init__(self) -> None:
        super().__init__("uam_control")

        self.declare_parameter("scenario", "nominal")
        self.declare_parameter("controller_rate", 1.0 / CONTROL.dt_eta)
        scenario_name = self.get_parameter("scenario").value
        self.scenario = SCENARIOS[scenario_name]

        self.translational = PositionPD()
        self.rotational = RotationalTubeMPC()
        self.manipulator = ManipulatorTubeMPC()

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)
        self.body_rate = np.zeros(3)
        self.joints = np.zeros(ARM.n_links)
        self.joint_rates = np.zeros(ARM.n_links)
        self.joint_acceleration = np.zeros(ARM.n_links)

        self.attitude_target = AttitudeReference(
            thrust=UAV.mass * GRAVITY, roll=0.0, pitch=0.0
        )
        self.body_rate_target = np.zeros(3)
        self.previous_attitude_ref = np.zeros(3)
        self.body_rate_dot = np.zeros(3)

        self._last_pose_time: float | None = None
        self._previous_position: np.ndarray | None = None
        self._previous_attitude: np.ndarray | None = None
        self._previous_body_rate = np.zeros(3)
        self._have_pose = False
        self._have_joints = False
        self._start_time: float | None = None
        self._slow_counter = 0

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(TFMessage, "/tf", self._on_transform, sensor_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joints, sensor_qos)

        self.wrench_publisher = self.create_publisher(Wrench, "/wrench_command", 10)
        self.joint_publisher = self.create_publisher(
            Float64MultiArray, "/joint_command", 10
        )
        self.debug_publisher = self.create_publisher(
            Float64MultiArray, "/uam_debug", 10
        )

        rate = float(self.get_parameter("controller_rate").value)
        self.create_timer(1.0 / rate, self._on_control_tick)
        self.create_timer(2.0, self._report_readiness)

        self.get_logger().info(
            f"uam_control started, scenario '{scenario_name}', "
            f"control rate {rate:.1f} Hz"
        )

    # -- sensing -----------------------------------------------------------

    def _on_transform(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if BASE_FRAME not in (transform.child_frame_id, transform.header.frame_id):
                continue

            stamp = transform.header.stamp
            now = stamp.sec + stamp.nanosec * 1e-9
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            position = np.array([translation.x, translation.y, translation.z])
            attitude = quaternion_to_euler(rotation.x, rotation.y, rotation.z, rotation.w)

            if self._last_pose_time is not None:
                dt = now - self._last_pose_time
                if dt > 1e-4:
                    velocity = (position - self._previous_position) / dt
                    self.velocity += VELOCITY_FILTER * (velocity - self.velocity)

                    euler_rate = self._angle_difference(
                        attitude, self._previous_attitude
                    ) / dt
                    body_rate = (
                        euler_rate_to_body_rate(attitude[0], attitude[1]) @ euler_rate
                    )
                    self.body_rate_dot = (body_rate - self._previous_body_rate) / dt
                    self._previous_body_rate = body_rate
                    self.body_rate += VELOCITY_FILTER * (body_rate - self.body_rate)

            self.position = position
            self.attitude = attitude
            self._previous_position = position.copy()
            self._previous_attitude = attitude.copy()
            self._last_pose_time = now
            self._have_pose = True
            if self._start_time is None:
                self._start_time = now
            return

    @staticmethod
    def _angle_difference(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Difference two angle triples without wrapping through +/- pi."""
        return (current - previous + np.pi) % (2.0 * np.pi) - np.pi

    def _on_joints(self, message: JointState) -> None:
        for index, name in enumerate(JOINT_NAMES):
            if name not in message.name:
                return
            source = message.name.index(name)
            self.joints[index] = message.position[source]
            if message.velocity and len(message.velocity) > source:
                self.joint_rates[index] = message.velocity[source]
        self._have_joints = True

    def _report_readiness(self) -> None:
        missing = []
        if not self._have_pose:
            missing.append(f"pose of {BASE_FRAME} on /tf")
        if not self._have_joints:
            missing.append("/joint_states for " + ", ".join(JOINT_NAMES))
        if missing:
            self.get_logger().warn(
                "waiting on: " + "; ".join(missing) + " - not commanding"
            )

    # -- control -----------------------------------------------------------

    def _on_control_tick(self) -> None:
        if not (self._have_pose and self._have_joints):
            return

        clock = self.get_clock().now().nanoseconds * 1e-9
        elapsed = clock - (self._start_time or clock)

        coupling = decompose(
            self.joints,
            self.joint_rates,
            self.manipulator.predicted_joint_acceleration,
            self.body_rate,
            self.rotational.predicted_body_rate_dot,
            self.attitude_target.thrust,
        )

        # Translational stage runs at half the rate of the inner loops.
        self._slow_counter += 1
        if self._slow_counter * CONTROL.dt_eta >= CONTROL.dt_zeta:
            self._slow_counter = 0
            plant_state = np.array(
                [
                    self.position[0],
                    self.velocity[0],
                    self.position[1],
                    self.velocity[1],
                    self.position[2],
                    self.velocity[2],
                ]
            )
            reference = horizon_reference(
                translational_reference,
                self.scenario,
                elapsed,
                CONTROL.N_zeta,
                CONTROL.dt_zeta,
            )
            optimal_force = self.translational(plant_state, reference)
            self.attitude_target = attitude_reference(
                optimal_force, coupling, 0.0, self.attitude
            )

            desired = np.array(
                [self.attitude_target.roll, self.attitude_target.pitch, 0.0]
            )
            self.previous_attitude_ref = desired
            # Proportional only: differencing the optimiser's attitude
            # reference produces a rate spike of (delta angle) / dt_zeta that
            # saturates the inner loop.
            euler_rates = ATTITUDE_GAIN * self._angle_difference(
                desired, self.attitude
            )
            self.body_rate_target = (
                euler_rate_to_body_rate(self.attitude[0], self.attitude[1])
                @ euler_rates
            )

        torque = self.rotational(
            self.body_rate,
            np.tile(self.body_rate_target, (CONTROL.N_eta, 1)),
            coupling,
        )
        joint_target = horizon_reference(
            joint_reference, self.scenario, elapsed, CONTROL.N_gamma, CONTROL.dt_gamma
        )
        joint_torque = self.manipulator(
            np.concatenate([self.joints, self.joint_rates]),
            joint_target,
            self.body_rate,
            self.rotational.predicted_body_rate_dot,
            self.attitude_target.thrust,
        )

        wrench = Wrench()
        wrench.force.z = float(self.attitude_target.thrust)
        wrench.torque.x = float(torque[0])
        wrench.torque.y = float(torque[1])
        wrench.torque.z = float(torque[2])
        self.wrench_publisher.publish(wrench)

        self.joint_publisher.publish(
            Float64MultiArray(data=[float(v) for v in joint_torque])
        )

        position_reference = translational_reference(self.scenario, elapsed)
        self.debug_publisher.publish(
            Float64MultiArray(
                data=[
                    elapsed,
                    *self.position,
                    *position_reference[[0, 2, 4]],
                    *self.attitude,
                    self.attitude_target.roll,
                    self.attitude_target.pitch,
                    *self.joints,
                    *joint_target[0][: ARM.n_links],
                ]
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UamControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
