cmake_minimum_required(VERSION 2.8)
project(cv_bridge)

find_package(catkin REQUIRED COMPONENTS rosconsole sensor_msgs)

set(PYTHON_VERSION 3.8)
set(PYTHON_EXECUTABLE /home/user1/py3venv/bin/python3)
set(PYTHON_INCLUDE_DIR /usr/include/python3.8)
set(PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libpython3.8.so)

if(NOT ANDROID)
  find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)
  find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
  if(PYTHONLIBS_VERSION_STRING VERSION_LESS 3)
    find_package(Boost REQUIRED python)
  else()
    find_package(Boost REQUIRED python3)
  endif()
else()
  find_package(Boost REQUIRED)
endif()
find_package(OpenCV 3 REQUIRED
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
  CONFIG
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS rosconsole sensor_msgs
  DEPENDS OpenCV
  CFG_EXTRAS cv_bridge-extras.cmake
)

catkin_python_setup()

include_directories(include ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

if(NOT ANDROID)
  add_subdirectory(python)
endif()
add_subdirectory(src)
if(CATKIN_ENABLE_TESTING)
  add_subdirectory(test)
endif()

# install the include folder
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

