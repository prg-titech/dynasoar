# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matthias/soa-alloc/mallocmc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matthias/soa-alloc/mallocmc

# Include any dependencies generated for this target.
include CMakeFiles/VerifyHeap.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/VerifyHeap.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VerifyHeap.dir/flags.make

CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o: CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o.depend
CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o: CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o.cmake
CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o: tests/verify_heap.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/matthias/soa-alloc/mallocmc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o"
	cd /home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/tests && /usr/bin/cmake -E make_directory /home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/tests/.
	cd /home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/tests && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/tests/./VerifyHeap_generated_verify_heap.cu.o -D generated_cubin_file:STRING=/home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/tests/./VerifyHeap_generated_verify_heap.cu.o.cubin.txt -P /home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o.cmake

# Object files for target VerifyHeap
VerifyHeap_OBJECTS =

# External object files for target VerifyHeap
VerifyHeap_EXTERNAL_OBJECTS = \
"/home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o"

VerifyHeap: CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o
VerifyHeap: CMakeFiles/VerifyHeap.dir/build.make
VerifyHeap: /usr/local/cuda/lib64/libcudart_static.a
VerifyHeap: /usr/lib/x86_64-linux-gnu/librt.so
VerifyHeap: CMakeFiles/VerifyHeap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/matthias/soa-alloc/mallocmc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable VerifyHeap"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VerifyHeap.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VerifyHeap.dir/build: VerifyHeap

.PHONY : CMakeFiles/VerifyHeap.dir/build

CMakeFiles/VerifyHeap.dir/requires:

.PHONY : CMakeFiles/VerifyHeap.dir/requires

CMakeFiles/VerifyHeap.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VerifyHeap.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VerifyHeap.dir/clean

CMakeFiles/VerifyHeap.dir/depend: CMakeFiles/VerifyHeap.dir/tests/VerifyHeap_generated_verify_heap.cu.o
	cd /home/matthias/soa-alloc/mallocmc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc/CMakeFiles/VerifyHeap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/VerifyHeap.dir/depend

