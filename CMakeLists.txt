# set the minimum version
cmake_minimum_required(VERSION "3.13")

# set project
project(image-data-cv)

# replace with your cmake/opencv and python path
set(OpenCV_DIR /Users/anshgodha/Developer/opencv/install/lib/cmake/opencv4)
set(PYTHON_INCLUDE_DIR /Users/anshgodha/opt/anaconda3/include/)
set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

# for matplotlibcpp
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

# populate matplotlib
include(FetchContent)
FetchContent_Declare(
    matplotlib
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        f23347fca25219d1c42cbb91608b5556814bf572
)
FetchContent_GetProperties(matplotlib)
if(NOT matplotlib_POPULATED)
    FetchContent_Populate(matplotlib)
endif()
include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

# link opencv to all targets. Note this shouldn't normally be done, but we expect to use
# this in every target so it is simpler to use this directory-wide link.
link_libraries(${OpenCV_LIBS})

add_subdirectory(src)