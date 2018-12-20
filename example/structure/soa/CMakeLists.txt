find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(structure_soa structure.cu rendering.cu dataset.cc)
target_link_libraries(structure_soa SDL2)

cuda_add_executable(structure_baseline structure_baseline.cu dataset.cc)
