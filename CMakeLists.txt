cmake_minimum_required(VERSION 3.16)

project(cpp_playground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/*.cpp")

if(NOT CPP_SOURCES)
  message(WARNING "No top-level .cpp files were found.")
endif()

foreach(CPP_SOURCE IN LISTS CPP_SOURCES)
  get_filename_component(TARGET_NAME "${CPP_SOURCE}" NAME_WE)
  add_executable("${TARGET_NAME}" "${CPP_SOURCE}")
  target_compile_options("${TARGET_NAME}" PRIVATE -Wall -Wextra -Wpedantic)
endforeach()
