find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(wator_dynasoar_no_cell wator.cu ../rendering.cu)
target_link_libraries(wator_dynasoar_no_cell SDL2 cuda)

