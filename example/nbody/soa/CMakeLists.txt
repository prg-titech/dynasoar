find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(nbody_soa nbody.cu rendering.cu)
target_link_libraries(nbody_soa SDL2)
cuda_add_executable(nbody_soa_baseline nbody_baseline.cu rendering.cu)
target_link_libraries(nbody_soa_baseline SDL2)
