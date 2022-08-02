add_library(
  metrics
  metrics.cpp metrics.hpp
)

target_include_directories(metrics PUBLIC ${CMAKE_SOURCE_DIR})