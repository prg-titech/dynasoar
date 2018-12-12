find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/example/configuration)
cuda_add_executable(collision_soa collision.cu rendering.cu)
target_link_libraries(collision_soa SDL2)
cuda_add_executable(collision_baseline collision_baseline.cu rendering.cu)
target_link_libraries(collision_baseline SDL2)

