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
include CMakeFiles/mallocMC_Example02.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mallocMC_Example02.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mallocMC_Example02.dir/flags.make

CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o: CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o.depend
CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o: CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o.cmake
CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o: examples/mallocMC_example02.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/matthias/soa-alloc/mallocmc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o"
	cd /home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/examples && /usr/bin/cmake -E make_directory /home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/examples/.
	cd /home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/examples && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/examples/./mallocMC_Example02_generated_mallocMC_example02.cu.o -D generated_cubin_file:STRING=/home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/examples/./mallocMC_Example02_generated_mallocMC_example02.cu.o.cubin.txt -P /home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o.cmake

# Object files for target mallocMC_Example02
mallocMC_Example02_OBJECTS =

# External object files for target mallocMC_Example02
mallocMC_Example02_EXTERNAL_OBJECTS = \
"/home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o"

mallocMC_Example02: CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o
mallocMC_Example02: CMakeFiles/mallocMC_Example02.dir/build.make
mallocMC_Example02: /usr/local/cuda/lib64/libcudart_static.a
mallocMC_Example02: /usr/lib/x86_64-linux-gnu/librt.so
mallocMC_Example02: CMakeFiles/mallocMC_Example02.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/matthias/soa-alloc/mallocmc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mallocMC_Example02"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mallocMC_Example02.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mallocMC_Example02.dir/build: mallocMC_Example02

.PHONY : CMakeFiles/mallocMC_Example02.dir/build

CMakeFiles/mallocMC_Example02.dir/requires:

.PHONY : CMakeFiles/mallocMC_Example02.dir/requires

CMakeFiles/mallocMC_Example02.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mallocMC_Example02.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mallocMC_Example02.dir/clean

CMakeFiles/mallocMC_Example02.dir/depend: CMakeFiles/mallocMC_Example02.dir/examples/mallocMC_Example02_generated_mallocMC_example02.cu.o
	cd /home/matthias/soa-alloc/mallocmc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc /home/matthias/soa-alloc/mallocmc/CMakeFiles/mallocMC_Example02.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mallocMC_Example02.dir/depend

