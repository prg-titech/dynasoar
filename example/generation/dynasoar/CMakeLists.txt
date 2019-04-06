find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(generation_soa generation.cu ../rendering.cu ../dataset_loader.cu)
target_link_libraries(generation_soa SDL2 cuda)
