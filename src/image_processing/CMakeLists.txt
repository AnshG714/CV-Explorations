add_library(
  image_processing
  histogram_equalization.cpp histogram_equalization.hpp
  linear_scaling.cpp linear_scaling.hpp
  smoothing.cpp smoothing.hpp
  sharpening.cpp sharpening.hpp
  edge_detector.hpp edge_detector.cpp
  corner_detector.hpp corner_detector.cpp
)

target_link_libraries(image_processing metrics)