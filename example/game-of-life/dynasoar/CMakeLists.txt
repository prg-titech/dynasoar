find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(gol_soa gol.cu ../rendering.cu ../dataset_loader.cu)
target_link_libraries(gol_soa SDL2 cuda)

