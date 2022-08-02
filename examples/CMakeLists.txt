set(IMAGE_PROCESSING_TARGETS histogram_equalization linear_scaling smoothing)

foreach(image_processing_target ${IMAGE_PROCESSING_TARGETS})
  add_executable("${image_processing_target}" "image_processing/${image_processing_target}.cpp")
  target_link_libraries("${image_processing_target}" image_processing utils)
endforeach()