# Install script for directory: D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/PBS")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
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

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libigl/cmake" TYPE FILE FILES "D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/libigl-config.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/libigl/cmake/libigl-export.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/libigl/cmake/libigl-export.cmake"
         "D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/CMakeFiles/Export/b00fa334c3cab0b257cdaf2865e998b7/libigl-export.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/libigl/cmake/libigl-export-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/libigl/cmake/libigl-export.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libigl/cmake" TYPE FILE FILES "D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/CMakeFiles/Export/b00fa334c3cab0b257cdaf2865e998b7/libigl-export.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/glad/cmake_install.cmake")
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/glfw/cmake_install.cmake")
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/imgui/cmake_install.cmake")
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/1_mass_spring/cmake_install.cmake")
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/2_fem/cmake_install.cmake")
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/3_fluid/cmake_install.cmake")
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/4-1_spinning/cmake_install.cmake")
  include("D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/4-2_gyro/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/ETH/Master/PhysicallyBasedSimulationInCG/pbs23-solution/build/debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
