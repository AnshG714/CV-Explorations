set(IMAGE_PROCESSING_TARGETS histogram_equalization smoothing sharpening edge_detector)

foreach(image_processing_target ${IMAGE_PROCESSING_TARGETS})
  add_executable("${image_processing_target}" "image_processing/${image_processing_target}.cpp")
  target_link_libraries("${image_processing_target}" image_processing utils)
endforeach()

add_executable(playground image_processing/playground.cpp)
target_link_libraries(playground utils)