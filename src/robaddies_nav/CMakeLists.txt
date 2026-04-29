cmake_minimum_required(VERSION 2.8.3)
project(robaddies_nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  gazebo_msgs
  std_msgs
)

catkin_package()
