find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(wator_soa wator.cu)
target_link_libraries(wator_soa SDL2)

cuda_add_executable(wator_soa_no_cell wator_no_cell.cu)
target_link_libraries(wator_soa_no_cell SDL2)
