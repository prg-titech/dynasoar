find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/example/configuration)
cuda_add_executable(sugarscape_soa sugarscape.cu)
target_link_libraries(sugarscape_soa SDL2)
