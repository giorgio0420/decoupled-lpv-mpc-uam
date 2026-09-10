cmake_minimum_required(VERSION 3.8)
project(hexacopter_control)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-plugin1 REQUIRED COMPONENTS register)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(HexacopterRotorPlugin SHARED src/hexacopter_rotor_plugin.cpp)
target_compile_features(HexacopterRotorPlugin PUBLIC cxx_std_17)
# ign-cmake exports version-suffixed variables (IGNITION-GAZEBO6_LIBRARIES), not
# the version-less ones, so link the imported targets directly instead.
# Plain signature, not PRIVATE: ament_target_dependencies below calls
# target_link_libraries without keywords, and CMake forbids mixing the two forms
# on one target.
target_link_libraries(HexacopterRotorPlugin
  ignition-gazebo6::ignition-gazebo6 ignition-plugin1::register)
ament_target_dependencies(HexacopterRotorPlugin
  geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs)

add_executable(allocation_node src/allocation_node.cpp)
target_compile_features(allocation_node PUBLIC cxx_std_17)
target_link_libraries(allocation_node Eigen3::Eigen)
ament_target_dependencies(allocation_node geometry_msgs rclcpp std_msgs)

install(TARGETS HexacopterRotorPlugin allocation_node
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES model_plugin.sdf.inc README.md DESTINATION share/${PROJECT_NAME})

ament_package()
