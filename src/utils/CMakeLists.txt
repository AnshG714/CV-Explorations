add_library(utils
  load_resource.cpp load_resource.hpp
)
target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR})