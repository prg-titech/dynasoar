find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(collision_baseline_aos collision.cu ../rendering.cu)
target_link_libraries(collision_baseline_aos SDL2)
