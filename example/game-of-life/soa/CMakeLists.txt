find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

cuda_add_executable(gol_soa gol.cu rendering.cu dataset_loader.cu)
target_link_libraries(gol_soa SDL2 cuda)

cuda_add_executable(gol_soa_no_cell gol_no_cell.cu rendering.cu dataset_loader.cu)
target_link_libraries(gol_soa_no_cell SDL2 cuda)

cuda_add_executable(gol_baseline_simple gol_baseline_simple.cu rendering.cu dataset_loader.cu)
target_link_libraries(gol_baseline_simple SDL2)

cuda_add_executable(gol_baseline_soa gol_baseline.cu dataset_loader.cu)

cuda_add_executable(gol_baseline_aos gol_baseline_aos.cu dataset_loader.cu)
