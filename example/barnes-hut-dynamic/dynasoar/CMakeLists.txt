find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(barnes_dynasoar barnes_hut.cu)
target_link_libraries(barnes_dynasoar SDL2 cuda)


