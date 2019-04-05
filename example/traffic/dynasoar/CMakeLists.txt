find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(traffic_dynasoar traffic.cu ../rendering.cu)
target_link_libraries(traffic_dynasoar SDL2 cuda)
