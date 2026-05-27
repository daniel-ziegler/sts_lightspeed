cmake_minimum_required(VERSION 3.19)
# Set the SDK path before the project command
set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk")

project(
        sts_lightspeed
        VERSION 0.1
        DESCRIPTION "High Performance, rng accurate implementation of Slay the Spire with tools for tree search and ml."
        LANGUAGES CXX)

file (GLOB_RECURSE sts_lightspeed_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS  "-Wno-shift-count-overflow -O3")

add_subdirectory(json)
add_subdirectory(pybind11)

# Create a shared static library with all core source files
add_library(sts_core STATIC ${sts_lightspeed_SOURCES})
target_include_directories(sts_core PUBLIC include)
target_include_directories(sts_core PUBLIC json/include)
set_target_properties(sts_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(slaythespire bindings/slaythespire.cpp bindings/bindings-util.cpp)
target_link_libraries(slaythespire PRIVATE sts_core)
target_include_directories(slaythespire PUBLIC json/single_include)
target_include_directories(slaythespire PUBLIC bindings)

add_executable(main apps/main.cpp)
target_link_libraries(main PRIVATE sts_core)
target_link_directories(main PRIVATE json::nlohmann_json)

add_executable(test apps/test.cpp)
target_link_libraries(test PRIVATE sts_core)
target_link_directories(test PUBLIC json::nlohmann_json)

add_executable(small-test apps/small-test.cpp)
target_link_libraries(small-test PRIVATE sts_core)
target_link_directories(small-test PRIVATE json::nlohmann_json)
target_include_directories(small-test PUBLIC bindings)
