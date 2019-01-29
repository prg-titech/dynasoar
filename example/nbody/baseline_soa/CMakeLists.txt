find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(nbody_baseline_soa nbody.cu ../rendering.cu)
target_link_libraries(nbody_baseline_soa SDL2)
