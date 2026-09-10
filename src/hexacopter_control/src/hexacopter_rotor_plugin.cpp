#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <iostream>
#include <string>
#include <vector>

#include <ignition/gazebo/Joint.hh>
#include <ignition/gazebo/Link.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Types.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Inertial.hh>
#include <ignition/gazebo/components/ChildLinkName.hh>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/common/Console.hh>
#include <ignition/plugin/Register.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/msgs/marker.pb.h>
#include <ignition/transport/Node.hh>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace hexacopter_control
{
class HexacopterRotorPlugin final : public ignition::gazebo::System,
  public ignition::gazebo::ISystemConfigure,
  public ignition::gazebo::ISystemPreUpdate
{
public:
  void Configure(const ignition::gazebo::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 ignition::gazebo::EntityComponentManager &_ecm,
                 ignition::gazebo::EventManager &) override
  {
    ignerr << "[HexacopterRotorPlugin] Configure entered" << std::endl;
    this->model_ = ignition::gazebo::Model(_entity);
    const auto baseName = _sdf->Get<std::string>("base_link", "robot_base").first;
    this->baseLink_ = ignition::gazebo::Link(this->model_.LinkByName(_ecm, baseName));
    this->kf_ = _sdf->Get<double>("thrust_coefficient", 2.772e-5).first;
    this->km_ = _sdf->Get<double>("moment_coefficient", 5.544e-7).first;
    this->maxOmega_ = _sdf->Get<double>("max_omega", 900.0).first;

    for (std::size_t i = 0; i < this->rotors_.size(); ++i)
    {
      this->joints_[i] = this->model_.JointByName(_ecm, this->rotors_[i].joint);
      if (this->joints_[i] == ignition::gazebo::kNullEntity)
      {
        std::cerr << "[HexacopterRotorPlugin] Missing joint: " << this->rotors_[i].joint
                  << " -- keeping the fallback position" << std::endl;
        continue;
      }

    }

    if (!rclcpp::ok())
      rclcpp::init(0, nullptr);
    this->node_ = std::make_shared<rclcpp::Node>("hexacopter_rotor_plugin");
    this->subscription_ = this->node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/rotor_speeds", rclcpp::QoS(1),
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg)
      {
        if (msg->data.size() != this->omega_.size())
        {
          RCLCPP_WARN(this->node_->get_logger(), "Ignoring /rotor_speeds: expected 6 values, got %zu", msg->data.size());
          return;
        }
        std::lock_guard<std::mutex> lock(this->commandMutex_);
        for (std::size_t i = 0; i < this->omega_.size(); ++i)
          this->omega_[i] = std::clamp(msg->data[i], 0.0, this->maxOmega_);
        RCLCPP_INFO_ONCE(this->node_->get_logger(),
          "first /rotor_speeds received, omega[0] = %.1f rad/s", this->omega_[0]);
      });
    ignerr << "[HexacopterRotorPlugin] subscribed to /rotor_speeds" << std::endl;
    for (std::size_t i = 0; i < this->armNames_.size(); ++i)
    {
      this->armJoints_[i] = this->model_.JointByName(_ecm, this->armNames_[i]);
      if (this->armJoints_[i] == ignition::gazebo::kNullEntity)
      {
        std::cerr << "[HexacopterRotorPlugin] Missing arm joint: "
                  << this->armNames_[i] << std::endl;
        continue;
      }
      // Position and velocity are not populated unless asked for.
      ignition::gazebo::Joint(this->armJoints_[i]).EnablePositionCheck(_ecm, true);
      ignition::gazebo::Joint(this->armJoints_[i]).EnableVelocityCheck(_ecm, true);
    }

    this->jointStates_ = this->node_->create_publisher<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS());
    this->jointCommand_ = this->node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/joint_torques", rclcpp::QoS(1),
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg)
      {
        if (msg->data.size() != this->jointTorque_.size())
        {
          RCLCPP_WARN(this->node_->get_logger(),
            "Ignoring /joint_torques: expected 3 values, got %zu", msg->data.size());
          return;
        }
        std::lock_guard<std::mutex> lock(this->commandMutex_);
        for (std::size_t i = 0; i < this->jointTorque_.size(); ++i)
          this->jointTorque_[i] = std::clamp(msg->data[i], -this->maxJointTorque_,
                                             this->maxJointTorque_);
      });

    this->maxJointTorque_ = _sdf->Get<double>("max_joint_torque", 6.0).first;

    // The commanded pose, drawn in the world so a watcher can tell tracking from
    // wandering. Until now nothing in the window said where the vehicle was
    // *supposed* to be, so a run that held station to 12 cm and one that drifted
    // ten metres looked the same on screen and could only be told apart in the
    // CSV.
    //
    // Drawn from inside the plugin rather than from the controller because the
    // marker service belongs to Ignition, not to ROS: a Python node would have to
    // shell out to `ign service` once per tick, which costs more than the whole
    // control loop. Here it is one in-process request on a transport node that is
    // already open.
    this->reference_ = this->node_->create_subscription<geometry_msgs::msg::Point>(
      "/uav/reference", rclcpp::QoS(1),
      [this](const geometry_msgs::msg::Point::SharedPtr msg)
      {
        std::lock_guard<std::mutex> lock(this->commandMutex_);
        this->referencePoint_ = ignition::math::Vector3d(msg->x, msg->y, msg->z);
        this->haveReference_ = true;
      });

    this->odometry_ = this->node_->create_publisher<nav_msgs::msg::Odometry>(
      "/uav/odom", rclcpp::SensorDataQoS());
    this->baseLink_.EnableVelocityChecks(_ecm, true);

    this->executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    this->executor_->add_node(this->node_);
  }

  void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
                 ignition::gazebo::EntityComponentManager &_ecm) override
  {
    if (this->executor_)
      this->executor_->spin_some();
    if (_info.paused || this->baseLink_.Entity() == ignition::gazebo::kNullEntity)
      return;

    this->DrawReference();

    // Resolve the rotor geometry from the model, once, on the first update.
    //
    // Not in Configure: the pose components are not populated yet there, and a
    // joint's own Pose is the transform across the joint -- zero at q = 0 --
    // rather than its offset in the parent. What is wanted is where the
    // propeller link actually sits relative to the base, so take both world
    // poses and express one in the other's frame.
    if (!this->geometryResolved_)
    {
      const auto base = ignition::gazebo::worldPose(this->baseLink_.Entity(), _ecm);
      ignition::math::Vector3d sum{0, 0, 0};
      bool complete = true;
      for (std::size_t i = 0; i < this->rotors_.size(); ++i)
      {
        if (this->joints_[i] == ignition::gazebo::kNullEntity) { complete = false; continue; }
        const auto *child = _ecm.Component<ignition::gazebo::components::ChildLinkName>(
          this->joints_[i]);
        if (!child) { complete = false; continue; }
        const auto link = this->model_.LinkByName(_ecm, child->Data());
        if (link == ignition::gazebo::kNullEntity) { complete = false; continue; }
        this->rotors_[i].position =
          (base.Inverse() * ignition::gazebo::worldPose(link, _ecm)).Pos();
        sum += ignition::math::Vector3d(
          this->rotors_[i].position.X(), this->rotors_[i].position.Y(), 0);
        std::cerr << "[ROTORPOS] " << this->rotors_[i].joint << " -> "
                  << this->rotors_[i].position << std::endl;
      }
      std::cerr << "[ROTORPOS] in-plane sum = " << sum
                << (sum.Length() < 1e-6
                      ? "  (balanced)"
                      : "  (NOT balanced: equal thrusts make a parasitic torque)")
                << std::endl;
      this->geometryResolved_ = complete;
    }

    std::array<double, 6> omega;
    {
      std::lock_guard<std::mutex> lock(this->commandMutex_);
      omega = this->omega_;
    }

    // Not Link::WorldPose: that reads a components::WorldPose which nothing
    // creates for this link, so it returns nullopt on every update and the
    // rotors never push. The free function walks the pose chain instead and
    // always has an answer.
    const auto worldPose = ignition::gazebo::worldPose(this->baseLink_.Entity(), _ecm);

    // AddWorldWrench applies the force at the link's centre of mass, so every
    // moment arm must be measured from the centre of mass rather than from the
    // robot_base origin the rotor positions are tabulated in. The arm hangs
    // well below the vehicle, so that offset is not negligible: using the
    // origin would inject a constant spurious roll/pitch moment proportional to
    // total thrust.
    ignition::math::Vector3d centreOfMass{0, 0, 0};
    if (const auto inertialPose = this->baseLink_.WorldInertialPose(_ecm))
      centreOfMass = (worldPose.Inverse() * *inertialPose).Pos();

    // One-shot diagnostic separating the two candidate causes of the lateral
    // component seen in the open-loop climb:
    //   1. the wrench is applied at the centre of mass while r x F is computed
    //      from the link origin, so an in-plane CoM offset leaks into torque;
    //   2. the link frame is not aligned with the world, so rotating [0,0,T]
    //      does not put the thrust on the vertical.
    static bool reported = false;
    if (!reported)
    {
      reported = true;
      // WorldInertialPose has the same problem as WorldPose: the component it
      // reads is never created, so it answers nullopt. The Inertial component
      // is written by the SDF loader and always present; its pose carries the
      // centre of mass in the link frame, which is what the lever arm needs.
      const auto inertial = this->baseLink_.WorldInertialPose(_ecm);
      const auto *inertialComp =
        _ecm.Component<ignition::gazebo::components::Inertial>(this->baseLink_.Entity());
      if (inertialComp)
      {
        const auto &massMatrix = inertialComp->Data().MassMatrix();
        std::cerr << "[ROTORDIAG] link mass    = " << massMatrix.Mass() << " kg" << std::endl;
        std::cerr << "[ROTORDIAG] CoM in link  = " << inertialComp->Data().Pose().Pos() << std::endl;
      }
      else
        std::cerr << "[ROTORDIAG] Inertial component absent" << std::endl;
      const auto up = worldPose.Rot().RotateVector(ignition::math::Vector3d(0, 0, 1));
      std::cerr << "[ROTORDIAG] link origin  = " << worldPose.Pos() << std::endl;
      if (inertial)
        std::cerr << "[ROTORDIAG] link CoM     = " << inertial->Pos() << std::endl
                  << "[ROTORDIAG] CoM offset   = " << (inertial->Pos() - worldPose.Pos()) << std::endl;
      else
        std::cerr << "[ROTORDIAG] link CoM     = <unavailable>" << std::endl;
      std::cerr << "[ROTORDIAG] rot(0,0,1)   = " << up << std::endl;
      std::cerr << "[ROTORDIAG] quaternion   = " << worldPose.Rot() << std::endl;
    }

    ignition::math::Vector3d forceBody{0, 0, 0};
    ignition::math::Vector3d torqueBody{0, 0, 0};
    for (std::size_t i = 0; i < this->rotors_.size(); ++i)
    {
      if (this->joints_[i] != ignition::gazebo::kNullEntity)
        ignition::gazebo::Joint(this->joints_[i]).SetVelocity(_ecm, {this->rotors_[i].spin * omega[i]});

      const double thrust = this->kf_ * omega[i] * omega[i];
      const ignition::math::Vector3d rotorForce{0, 0, thrust};
      forceBody += rotorForce;
      torqueBody += (this->rotors_[i].position - centreOfMass).Cross(rotorForce);
      torqueBody.Z() += this->rotors_[i].spin * this->km_ * omega[i] * omega[i];
    }
    // Periodic pose report, taken in the link frame -- the same frame the wrench
    // below is applied in. The model frame published on
    // /world/empty/dynamic_pose/info is rotated 180 degrees with respect to this
    // one, so x and y read from there cannot be compared against a wrench
    // expressed here without first being transformed.
    {
      static std::chrono::steady_clock::duration nextReport{0};
      if (_info.simTime >= nextReport)
      {
        nextReport = _info.simTime + std::chrono::milliseconds(500);
        const double t =
          std::chrono::duration<double>(_info.simTime).count();
        const auto &pos = worldPose.Pos();
        const auto rpy = worldPose.Rot().Euler();
        std::cerr << "[POSE] t=" << t
                  << " x=" << pos.X() << " y=" << pos.Y() << " z=" << pos.Z()
                  << " roll=" << rpy.X() << " pitch=" << rpy.Y()
                  << " |f|=" << forceBody.Length()
                  << " tau=(" << torqueBody.X() << "," << torqueBody.Y()
                  << "," << torqueBody.Z() << ")" << std::endl;
      }
    }

    RCLCPP_INFO_ONCE(this->node_->get_logger(),
      "applying thrust: |f_body| = %.2f N", forceBody.Length());
    {
      static std::chrono::steady_clock::duration nextReport{0};
      if (_info.simTime >= nextReport)
      {
        nextReport = _info.simTime + std::chrono::milliseconds(500);
        const double t = std::chrono::duration<double>(_info.simTime).count();
        const auto &pos = worldPose.Pos();
        const auto rpy = worldPose.Rot().Euler();
        std::cerr << "[POSE] t=" << t
                  << " x=" << pos.X() << " y=" << pos.Y() << " z=" << pos.Z()
                  << " roll=" << rpy.X() << " pitch=" << rpy.Y()
                  << " |f|=" << forceBody.Length()
                  << " tau=(" << torqueBody.X() << "," << torqueBody.Y()
                  << "," << torqueBody.Z() << ")" << std::endl;
      }
    }

    {
      static std::chrono::steady_clock::duration nextOdom{0};
      if (_info.simTime >= nextOdom)
      {
        nextOdom = _info.simTime + std::chrono::milliseconds(10);
        const auto linear = this->baseLink_.WorldLinearVelocity(_ecm);
        const auto angular = this->baseLink_.WorldAngularVelocity(_ecm);
        nav_msgs::msg::Odometry odom;
        odom.header.stamp.sec =
          std::chrono::duration_cast<std::chrono::seconds>(_info.simTime).count();
        odom.header.stamp.nanosec = static_cast<uint32_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(_info.simTime).count() % 1000000000);
        odom.header.frame_id = "world";
        odom.child_frame_id = "robot_base";
        odom.pose.pose.position.x = worldPose.Pos().X();
        odom.pose.pose.position.y = worldPose.Pos().Y();
        odom.pose.pose.position.z = worldPose.Pos().Z();
        // Published as Gazebo reports it, not conjugated.
        //
        // A conjugation was tried here on the reading that the pose was
        // inverted. That reading came from comparing a 57 degree roll against
        // small-angle intuition while the vehicle was already tumbling, so it
        // was the comparison that was invalid, not the pose. With the
        // conjugation in place even hover broke: body rates of 20 rad/s and
        // 200 N of thrust within two seconds, where the unconjugated version
        // holds position dead still.
        odom.pose.pose.orientation.x = worldPose.Rot().X();
        odom.pose.pose.orientation.y = worldPose.Rot().Y();
        odom.pose.pose.orientation.z = worldPose.Rot().Z();
        odom.pose.pose.orientation.w = worldPose.Rot().W();
        if (linear)
        {
          odom.twist.twist.linear.x = linear->X();
          odom.twist.twist.linear.y = linear->Y();
          odom.twist.twist.linear.z = linear->Z();
        }
        if (angular)
        {
          // Body rates, in the same convention as the pose above.
          //
          // Both were briefly conjugated on the reading that the published pose
          // was inverted. It was not: that reading compared a 57 degree roll
          // against small-angle intuition while the vehicle was already
          // tumbling, so the comparison was invalid rather than the pose. With
          // the conjugation in, hover itself broke -- 20 rad/s of body rate and
          // 200 N of thrust within two seconds of a run that used to hold
          // position dead still.
          const auto body = worldPose.Rot().RotateVectorReverse(*angular);
          odom.twist.twist.angular.x = body.X();
          odom.twist.twist.angular.y = body.Y();
          odom.twist.twist.angular.z = body.Z();
        }
        this->odometry_->publish(odom);
      }
    }

    // Arm: apply the commanded torques and publish the joint state back.
    {
      std::array<double, 3> jointTorque;
      {
        std::lock_guard<std::mutex> lock(this->commandMutex_);
        jointTorque = this->jointTorque_;
      }

      sensor_msgs::msg::JointState state;
      state.header.stamp.sec =
        std::chrono::duration_cast<std::chrono::seconds>(_info.simTime).count();
      state.header.stamp.nanosec = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(_info.simTime).count()
        % 1000000000);

      // Joint acceleration, differentiated at the simulation step and printed
      // here rather than derived from /joint_states.
      //
      // Over ROS the state arrives at 58 Hz, one sample every 17 ms, and three
      // separate attempts to measure the mass matrix through it all failed for
      // different reasons: a 0.3 s window let the joints reach their 10 rad/s
      // limit so the reading was limit/window; a 20 ms window held barely one
      // sample; a 0.15 s window at low torque let the pendulum's own restoring
      // moment dominate the applied one. The valid band between those three is
      // narrower than the sample period allows.
      //
      // Differentiating inside PreUpdate uses the physics step directly, so the
      // acceleration is read at the instant the torque is applied, before the
      // velocity limit or gravity have had time to matter.
      {
        static std::array<double, 3> lastVel{};
        static std::chrono::steady_clock::duration lastT{};
        const auto dt = std::chrono::duration<double>(_info.simTime - lastT).count();
        std::array<double, 3> vel{}, pos{};
        bool haveAll = true;
        for (std::size_t i = 0; i < this->armJoints_.size(); ++i)
        {
          if (this->armJoints_[i] == ignition::gazebo::kNullEntity) { haveAll = false; break; }
          ignition::gazebo::Joint joint(this->armJoints_[i]);
          const auto v = joint.Velocity(_ecm);
          const auto q = joint.Position(_ecm);
          if (!v || v->empty() || !q || q->empty()) { haveAll = false; break; }
          vel[i] = (*v)[0];
          pos[i] = (*q)[0];
        }
        if (haveAll && dt > 1e-9 && dt < 0.01)
        {
          static std::chrono::steady_clock::duration nextArm{};
          const bool driven =
            std::abs(jointTorque[0]) + std::abs(jointTorque[1]) + std::abs(jointTorque[2]) > 1e-9;
          if (driven && _info.simTime >= nextArm)
          {
            // Every physics step. The valid measurement window is short: at 0.1 N m the
            // arm reaches 0.5 rad/s in about 9 ms, so a 2 ms throttle left one or two
            // usable samples and the estimate was noise.
            nextArm = _info.simTime + std::chrono::milliseconds(1);
            std::cerr << "[ARMDYN] tau=(" << jointTorque[0] << "," << jointTorque[1]
                      << "," << jointTorque[2] << ")"
                      << " q=(" << pos[0] << "," << pos[1] << "," << pos[2] << ")"
                      << " qd=(" << vel[0] << "," << vel[1] << "," << vel[2] << ")"
                      << " qdd=(" << (vel[0] - lastVel[0]) / dt << ","
                      << (vel[1] - lastVel[1]) / dt << ","
                      << (vel[2] - lastVel[2]) / dt << ")"
                      << " dt=" << dt << std::endl;
          }
        }
        if (haveAll) { lastVel = vel; }
        lastT = _info.simTime;
      }

      for (std::size_t i = 0; i < this->armJoints_.size(); ++i)
      {
        if (this->armJoints_[i] == ignition::gazebo::kNullEntity)
          continue;
        ignition::gazebo::Joint joint(this->armJoints_[i]);
        joint.SetForce(_ecm, {jointTorque[i]});

        const auto position = joint.Position(_ecm);
        const auto velocity = joint.Velocity(_ecm);
        state.name.push_back(this->armNames_[i]);
        state.position.push_back(position && !position->empty() ? position->at(0) : 0.0);
        state.velocity.push_back(velocity && !velocity->empty() ? velocity->at(0) : 0.0);
        state.effort.push_back(jointTorque[i]);
      }

      static std::chrono::steady_clock::duration nextJointReport{0};
      if (!state.name.empty() && _info.simTime >= nextJointReport)
      {
        nextJointReport = _info.simTime + std::chrono::milliseconds(10);
        this->jointStates_->publish(state);
      }
    }

    // RotateVector, not RotateVectorReverse: measured, not reasoned.
    //
    // The reverse was tried on the reading that worldPose.Rot() is
    // world-to-body, and it destroys the vehicle -- it tumbles within 0.2 s at
    // 19 rad/s. So this rotation is correct as written and worldPose.Rot() is
    // body-to-world, which also means the torque the plugin builds reaches the
    // body unchanged in sign.
    this->baseLink_.AddWorldWrench(
      _ecm,
      worldPose.Rot().RotateVector(forceBody),
      worldPose.Rot().RotateVector(torqueBody));
  }

private:
  struct Rotor
  {
    const char *joint;
    ignition::math::Vector3d position;  // filled in Configure, from the model
    double spin;
  };

  // Positions are expressed in robot_base. spin signs are provisional: verify CW/CCW once in Gazebo.
  // Only the joint names and the rotation senses are declared here. The
  // positions are read from the model in Configure, because a hand-written copy
  // of them does not survive edits to the URDF.
  //
  // They had already drifted: the propeller joint origins were rotated 30
  // degrees to line the blades up with their arms, and this table kept the old
  // values. Four of the six were then wrong, so r x F was computed on the wrong
  // lever arms and a commanded pitch torque pushed the vehicle the opposite way
  // from what the model predicted -- which is what made horizontal tracking
  // impossible in Gazebo while it worked offline. The values below are left as
  // a fallback for when a joint cannot be found.
  std::array<Rotor, 6> rotors_{{
    {"joint_front_left_prop",  { 0.000000,  0.200000, 2.021000},  1.0},
    {"joint_front_right_prop", { 0.173203,  0.100000, 2.021000}, -1.0},
    {"joint_right_prop",       { 0.173205, -0.100000, 2.021000},  1.0},
    {"joint_back_right_prop",  { 0.000000, -0.200000, 2.021000}, -1.0},
    {"joint_back_left_prop",   {-0.173203, -0.100000, 2.021000},  1.0},
    {"joint_left_prop",        {-0.173205,  0.100000, 2.021000}, -1.0},
  }};
  ignition::gazebo::Model model_{ignition::gazebo::kNullEntity};
  ignition::gazebo::Link baseLink_{ignition::gazebo::kNullEntity};
  std::array<ignition::gazebo::Entity, 6> joints_{};
  bool geometryResolved_{false};
  std::array<double, 6> omega_{};
  double kf_{2.772e-5};
  double km_{5.544e-7};
  double maxOmega_{900.0};
  /// Draw the commanded pose as a sphere, plus a trail of where it has been.
  ///
  /// Decimated: the physics runs at 1 kHz and the eye does not, so redrawing
  /// every step would spend more on the marker than on the vehicle. One in fifty
  /// is 20 Hz, which is smooth on screen and matches the control rate.
  ///
  /// The trail uses a distinct namespace and an id that advances, so the points
  /// accumulate instead of replacing each other -- that is what turns a moving
  /// dot into a visible commanded path next to the flown one.
  void DrawReference()
  {
    ignition::math::Vector3d target;
    {
      std::lock_guard<std::mutex> lock(this->commandMutex_);
      if (!this->haveReference_)
        return;
      target = this->referencePoint_;
    }
    if (this->markerCounter_++ % 50 != 0)
      return;

    ignition::msgs::Marker marker;
    marker.set_ns("riferimento");
    marker.set_id(1);
    marker.set_action(ignition::msgs::Marker::ADD_MODIFY);
    marker.set_type(ignition::msgs::Marker::SPHERE);
    marker.mutable_lifetime()->set_sec(0);
    marker.mutable_material()->mutable_ambient()->set_r(0.1f);
    marker.mutable_material()->mutable_ambient()->set_g(0.9f);
    marker.mutable_material()->mutable_ambient()->set_b(0.2f);
    marker.mutable_material()->mutable_ambient()->set_a(0.6f);
    marker.mutable_material()->mutable_diffuse()->set_r(0.1f);
    marker.mutable_material()->mutable_diffuse()->set_g(0.9f);
    marker.mutable_material()->mutable_diffuse()->set_b(0.2f);
    marker.mutable_material()->mutable_diffuse()->set_a(0.6f);
    ignition::msgs::Set(marker.mutable_scale(),
                        ignition::math::Vector3d(0.35, 0.35, 0.35));
    ignition::msgs::Set(marker.mutable_pose(),
                        ignition::math::Pose3d(target, ignition::math::Quaterniond::Identity));
    this->markerNode_.Request("/marker", marker);

    // The path it has traced, one small marker per sample, kept for good.
    ignition::msgs::Marker trail;
    trail.set_ns("scia");
    trail.set_id(1 + this->markerCounter_ / 50);
    trail.set_action(ignition::msgs::Marker::ADD_MODIFY);
    trail.set_type(ignition::msgs::Marker::SPHERE);
    trail.mutable_material()->mutable_ambient()->set_r(0.1f);
    trail.mutable_material()->mutable_ambient()->set_g(0.6f);
    trail.mutable_material()->mutable_ambient()->set_b(1.0f);
    trail.mutable_material()->mutable_diffuse()->set_r(0.1f);
    trail.mutable_material()->mutable_diffuse()->set_g(0.6f);
    trail.mutable_material()->mutable_diffuse()->set_b(1.0f);
    ignition::msgs::Set(trail.mutable_scale(),
                        ignition::math::Vector3d(0.08, 0.08, 0.08));
    ignition::msgs::Set(trail.mutable_pose(),
                        ignition::math::Pose3d(target, ignition::math::Quaterniond::Identity));
    this->markerNode_.Request("/marker", trail);
  }

  std::mutex commandMutex_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointStates_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr jointCommand_;
  const std::array<const char *, 3> armNames_{{"Joint_1", "Joint_2", "Joint_3"}};
  std::array<ignition::gazebo::Entity, 3> armJoints_{};
  std::array<double, 3> jointTorque_{};
  double maxJointTorque_{6.0};

  // Reference marker. `markerNode_` talks to Ignition's own transport, which is
  // a different bus from the ROS one the rest of this plugin uses.
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr reference_;
  ignition::transport::Node markerNode_;
  ignition::math::Vector3d referencePoint_{0, 0, 0};
  bool haveReference_{false};
  std::uint64_t markerCounter_{0};
  // Held by pointer, not by value. The executor builds a guard condition
  // against the global rclcpp context, so constructing it requires that
  // context to exist. Gazebo constructs the plugin object before it ever calls
  // Configure, which is where rclcpp::init runs, so a by-value member is built
  // against a null context and throws RCLInvalidArgument -- taking the whole
  // simulator down with 'terminate called after throwing'.
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
};
}  // namespace hexacopter_control

IGNITION_ADD_PLUGIN(hexacopter_control::HexacopterRotorPlugin,
  ignition::gazebo::System,
  hexacopter_control::HexacopterRotorPlugin::ISystemConfigure,
  hexacopter_control::HexacopterRotorPlugin::ISystemPreUpdate)
IGNITION_ADD_PLUGIN_ALIAS(hexacopter_control::HexacopterRotorPlugin,
  "hexacopter_control::HexacopterRotorPlugin")
