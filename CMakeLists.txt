# set the minimum version
cmake_minimum_required(VERSION "3.13")

# set project
project(image-data-cv)

# replace with your cmake/opencv path
set(OpenCV_DIR /Users/anshgodha/Developer/opencv/install/lib/cmake/opencv4)
set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

# for matplotlibcpp
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

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
MESSAGE(${matplotlib_SOURCE_DIR})

set(PROJECTS basic_data_measures;data_measure_img_seq)
foreach(PROJECT ${PROJECTS})
  add_executable(${PROJECT} ${PROJECT}/main.cpp)
  target_link_libraries(${PROJECT} PUBLIC ${OpenCV_LIBS})
  set_target_properties(${PROJECT} PROPERTIES OUTPUT_NAME ${PROJECT})
endforeach(PROJECT ${PROJECTS})

# link python and numpy
target_link_libraries(data_measure_img_seq
    PRIVATE
        ${PYTHON_LIBRARIES}
        Python3::NumPy
)
