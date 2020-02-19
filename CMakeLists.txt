cmake_minimum_required(VERSION 3.12)
project(bin_picking VERSION 0.1.0 LANGUAGES CXX)


option(GRASPRO_BUILD_ENSENSO_NODE "Build ensenso node." ON)
option(GRASPRO_BUILD_REALSENSE_NODE "Build realsense node." ON)
option(GRASPRO_BUILD_FRANKA_CONTROL "Build franka control." ON)


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(Eigen3_DIRS "/usr/local/include/eigen3/")
set(PYTHON_EXECUTABLE python3.6)
set(PYTHON_LIBRARY "/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6m.so")


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.3 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python 3.6 EXACT REQUIRED)
find_package(Franka 0.7.1 EXACT REQUIRED)


if(GRASPRO_BUILD_ENSENSO_NODE)
  find_package(Ensenso 2.2 REQUIRED)
endif()


if(GRASPRO_BUILD_REALSENSE_NODE)
  find_package(OpenGL REQUIRED)
  find_package(realsense2 REQUIRED)
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  actionlib
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
  franka_msgs
  interactive_markers
  image_transport
  cv_bridge
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
)


include_directories(include)


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Grasp.msg
  OrthographicImage.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  GetOrthographicImages.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS
    include
#  LIBRARIES
  CATKIN_DEPENDS
    roscpp
    message_runtime
    std_msgs
    sensor_msgs
    geometry_msgs
    interactive_markers
    franka_msgs
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
#  DEPENDS
)

###########
## Build ##
###########

add_library(ros INTERFACE)
target_include_directories(ros INTERFACE ${catkin_INCLUDE_DIRS})
target_link_libraries(ros INTERFACE ${catkin_LIBRARIES})
add_dependencies(ros ${catkin_EXPORTED_TARGETS})


pybind11_add_module(orthographical src/orthographic_image/python.cpp)
target_include_directories(orthographical PRIVATE ${Eigen3_DIRS})
target_link_libraries(orthographical PRIVATE ${OpenCV_LIBS})


add_library(frankr src/frankr/robot.cpp src/frankr/gripper.cpp)
target_include_directories(frankr PUBLIC ${catkin_INCLUDE_DIRS} ${Eigen3_DIRS})
target_link_libraries(frankr PUBLIC ${catkin_LIBRARIES} Franka::Franka)

pybind11_add_module(cfrankr src/frankr/python.cpp)
target_link_libraries(cfrankr PUBLIC frankr)


if(GRASPRO_BUILD_ENSENSO_NODE)
  add_library(ensenso src/ensenso/ensenso.cpp)
  target_include_directories(ensenso PUBLIC ${Eigen3_DIRS} ${ENSENSO_INCLUDE_DIRS})
  target_link_libraries(ensenso PUBLIC ${ENSENSO_LIBRARIES} ${OpenCV_LIBS} ros)

  add_executable(ensenso_node src/ensenso/ensenso_node.cpp)
  target_link_libraries(ensenso_node PRIVATE ${OpenCV_LIBS} ensenso ros)
endif()


if(GRASPRO_BUILD_REALSENSE_NODE)
  add_library(realsense src/realsense/realsense.cpp)
  target_include_directories(realsense PUBLIC ${Eigen3_DIRS} ${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} ${realsense_INCLUDE_DIR})
  target_link_libraries(realsense PUBLIC glfw ${GLFW_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${realsense2_LIBRARY} ros)

  add_executable(realsense_node src/realsense/realsense_node.cpp)
  target_link_libraries(realsense_node PRIVATE ${OpenCV_LIBS} realsense ros)
endif()
