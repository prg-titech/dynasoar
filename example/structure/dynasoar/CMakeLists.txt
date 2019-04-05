find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(structure_dynasoar structure.cu ../rendering.cu ../dataset.cc)
target_link_libraries(structure_dynasoar SDL2 cuda)

