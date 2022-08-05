add_executable(data_measure_img_seq main.cpp)
set_target_properties(data_measure_img_seq PROPERTIES OUTPUT_NAME data_measure_img_seq)

# link python and numpy
target_link_libraries(data_measure_img_seq
    PRIVATE
        ${PYTHON3_LIBRARIES}
        Python3::NumPy
)