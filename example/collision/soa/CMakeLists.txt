find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(collision_soa collision.cu rendering.cu)
target_link_libraries(collision_soa SDL2 cuda)

cuda_add_executable(collision_baseline_soa collision_baseline.cu rendering.cu)
target_link_libraries(collision_baseline_soa SDL2)

cuda_add_executable(collision_baseline_aos collision_baseline_aos.cu rendering.cu)
target_link_libraries(collision_baseline_aos SDL2)
