add_library(
  image_processing 
  histogram_equalization.cpp histogram_equalization.hpp
  linear_scaling.cpp linear_scaling.hpp
)
target_include_directories(image_processing PUBLIC ${CMAKE_SOURCE_DIR})