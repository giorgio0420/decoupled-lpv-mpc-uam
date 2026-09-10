#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

#include <Eigen/Core>
#include <Eigen/QR>
#include <geometry_msgs/msg/wrench.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

class AllocationNode final : public rclcpp::Node
{
public:
  AllocationNode() : Node("allocation_node")
  {
    this->declare_parameter<double>("thrust_coefficient", 8.0e-6);
    this->declare_parameter<double>("moment_coefficient", 1.6e-7);
    this->declare_parameter<double>("max_omega", 900.0);
    this->get_parameter("thrust_coefficient", this->kf_);
    this->get_parameter("moment_coefficient", this->km_);
    this->get_parameter("max_omega", this->maxOmega_);
    this->publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/rotor_speeds", 10);
    this->subscription_ = this->create_subscription<geometry_msgs::msg::Wrench>(
      "/wrench_command", 10, std::bind(&AllocationNode::allocate, this, std::placeholders::_1));
  }

private:
  void allocate(const geometry_msgs::msg::Wrench::SharedPtr command)
  {
    // Body convention: x forward, y left, z up. Input = [T, tau_x, tau_y, tau_z].
    const std::array<double, 6> x{{0.0, 0.173203, 0.173205, 0.0, -0.173203, -0.173205}};
    const std::array<double, 6> y{{0.2, 0.1, -0.1, -0.2, -0.1, 0.1}};
    const std::array<double, 6> spin{{1.0, -1.0, 1.0, -1.0, 1.0, -1.0}};
    Eigen::Matrix<double, 4, 6> allocation;
    for (int i = 0; i < 6; ++i)
      allocation.col(i) << this->kf_, y[i] * this->kf_, -x[i] * this->kf_, spin[i] * this->km_;

    Eigen::Vector4d wrench;
    wrench << command->force.z, command->torque.x, command->torque.y, command->torque.z;
    const Eigen::Matrix<double, 6, 1> squared = allocation.completeOrthogonalDecomposition().solve(wrench);

    // The minimum-norm solution of an under-determined system is free to return
    // negative squared speeds, and a rotor cannot produce negative thrust. The
    // clamp below keeps the command physical but silently changes the delivered
    // wrench, so report how far off it lands rather than hiding it: a
    // persistently large residual means the controller is asking for a wrench
    // this airframe cannot produce, and the fix is upstream.
    std_msgs::msg::Float64MultiArray output;
    output.data.resize(6);
    Eigen::Matrix<double, 6, 1> feasible;
    for (int i = 0; i < 6; ++i)
    {
      feasible(i) = std::clamp(squared(i), 0.0, this->maxOmega_ * this->maxOmega_);
      output.data[i] = std::sqrt(feasible(i));
    }

    const double residual = (allocation * feasible - wrench).norm();
    if (residual > 1e-3 * std::max(1.0, wrench.norm()))
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Allocation saturated: wrench residual %.3f (commanded norm %.3f)",
        residual, wrench.norm());

    this->publisher_->publish(output);
  }

  double kf_{8.0e-6};
  double km_{1.6e-7};
  double maxOmega_{900.0};
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
  rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AllocationNode>());
  rclcpp::shutdown();
  return 0;
}
