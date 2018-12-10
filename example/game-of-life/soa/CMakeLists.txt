find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/example/configuration)
cuda_add_executable(gol_soa gol.cu rendering.cu dataset_loader.cu)
target_link_libraries(gol_soa SDL2)
cuda_add_executable(gol_soa_baseline gol_baseline.cu rendering.cu dataset_loader.cu)
target_link_libraries(gol_soa_baseline SDL2)