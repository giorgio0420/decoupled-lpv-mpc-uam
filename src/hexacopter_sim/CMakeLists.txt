cmake_minimum_required(VERSION 3.8)
project(hexacopter_sim)

#Find the ament build system

find_package(ament_cmake REQUIRED)

#Copy our folders to the install directory so ROS can find them

install(
DIRECTORY launch model parameters
DESTINATION share/${PROJECT_NAME}
)

#Export the package

ament_package()