find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(traffic_soa traffic.cu rendering.cu)
target_link_libraries(traffic_soa SDL2 cuda)

cuda_add_executable(traffic_baseline_2class_soa traffic_baseline_2class.cu)

cuda_add_executable(traffic_baseline_2class_aos traffic_baseline_2class_aos.cu)

cuda_add_executable(traffic_baseline_aos traffic_baseline_aos.cu)

cuda_add_executable(traffic_baseline traffic_baseline.cu)
