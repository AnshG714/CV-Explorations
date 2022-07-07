add_subdirectory(basic_data_measures)

set(PROJECTS data_measure_img_seq)
foreach(PROJECT ${PROJECTS})
  add_executable(${PROJECT} ${PROJECT}/main.cpp)
  target_link_libraries(${PROJECT} PUBLIC ${OpenCV_LIBS})
  set_target_properties(${PROJECT} PROPERTIES OUTPUT_NAME ${PROJECT})
endforeach(PROJECT ${PROJECTS})