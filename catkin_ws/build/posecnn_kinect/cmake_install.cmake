# Install script for directory: /media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/src/posecnn_kinect

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  include("/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/posecnn_kinect/catkin_generated/safe_execute_install.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/posecnn_kinect/msg" TYPE FILE FILES "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/src/posecnn_kinect/msg/PoseCNNMsg.msg")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/posecnn_kinect/catkin_generated/installspace/posecnn_kinect.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/posecnn_kinect/cmake" TYPE FILE FILES
    "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/posecnn_kinect/catkin_generated/installspace/posecnn_kinectConfig.cmake"
    "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/posecnn_kinect/catkin_generated/installspace/posecnn_kinectConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/posecnn_kinect" TYPE FILE FILES "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/src/posecnn_kinect/package.xml")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/posecnn_kinect" TYPE PROGRAM FILES "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/posecnn_kinect/catkin_generated/installspace/hello")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/posecnn_kinect" TYPE PROGRAM FILES "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/PoseCNN/catkin_ws/build/posecnn_kinect/catkin_generated/installspace/test_images")
endif()

