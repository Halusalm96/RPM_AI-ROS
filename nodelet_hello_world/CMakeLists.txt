cmake_minimum_required(VERSION 3.0.2)
project(nodelet_hello_world)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  nodelet
  pluginlib
  sensor_msgs
  image_transport
)

find_package(OpenCV REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge nodelet pluginlib sensor_msgs image_transport
  LIBRARIES yolo_processor_wrapper
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# Python 스크립트를 설치하도록 설정
catkin_install_python(PROGRAMS
  src/yolo_processor_nodelet.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# YOLOProcessor Nodelet C++ 래퍼 추가
add_library(yolo_processor_wrapper src/yolo_processor_wrapper.cpp)
target_link_libraries(yolo_processor_wrapper ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
add_dependencies(yolo_processor_wrapper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

install(TARGETS yolo_processor_wrapper
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
