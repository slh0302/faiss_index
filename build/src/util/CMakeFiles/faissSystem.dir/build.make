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
CMAKE_SOURCE_DIR = /home/slh/faiss_index

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/slh/faiss_index/build

# Include any dependencies generated for this target.
include src/util/CMakeFiles/faissSystem.dir/depend.make

# Include the progress variables for this target.
include src/util/CMakeFiles/faissSystem.dir/progress.make

# Include the compile flags for this target's objects.
include src/util/CMakeFiles/faissSystem.dir/flags.make

src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o: src/util/CMakeFiles/faissSystem.dir/flags.make
src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o: ../src/util/faissSystem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/slh/faiss_index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o"
	cd /home/slh/faiss_index/build/src/util && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/faissSystem.dir/faissSystem.cpp.o -c /home/slh/faiss_index/src/util/faissSystem.cpp

src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/faissSystem.dir/faissSystem.cpp.i"
	cd /home/slh/faiss_index/build/src/util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/slh/faiss_index/src/util/faissSystem.cpp > CMakeFiles/faissSystem.dir/faissSystem.cpp.i

src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/faissSystem.dir/faissSystem.cpp.s"
	cd /home/slh/faiss_index/build/src/util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/slh/faiss_index/src/util/faissSystem.cpp -o CMakeFiles/faissSystem.dir/faissSystem.cpp.s

src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.requires:

.PHONY : src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.requires

src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.provides: src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.requires
	$(MAKE) -f src/util/CMakeFiles/faissSystem.dir/build.make src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.provides.build
.PHONY : src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.provides

src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.provides.build: src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o


# Object files for target faissSystem
faissSystem_OBJECTS = \
"CMakeFiles/faissSystem.dir/faissSystem.cpp.o"

# External object files for target faissSystem
faissSystem_EXTERNAL_OBJECTS =

../bin/faissSystem: src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o
../bin/faissSystem: src/util/CMakeFiles/faissSystem.dir/build.make
../bin/faissSystem: ../../caffe-ssd/build/lib/libcaffe.so
../bin/faissSystem: src/util/CMakeFiles/faissSystem.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/slh/faiss_index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/faissSystem"
	cd /home/slh/faiss_index/build/src/util && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/faissSystem.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/util/CMakeFiles/faissSystem.dir/build: ../bin/faissSystem

.PHONY : src/util/CMakeFiles/faissSystem.dir/build

src/util/CMakeFiles/faissSystem.dir/requires: src/util/CMakeFiles/faissSystem.dir/faissSystem.cpp.o.requires

.PHONY : src/util/CMakeFiles/faissSystem.dir/requires

src/util/CMakeFiles/faissSystem.dir/clean:
	cd /home/slh/faiss_index/build/src/util && $(CMAKE_COMMAND) -P CMakeFiles/faissSystem.dir/cmake_clean.cmake
.PHONY : src/util/CMakeFiles/faissSystem.dir/clean

src/util/CMakeFiles/faissSystem.dir/depend:
	cd /home/slh/faiss_index/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/slh/faiss_index /home/slh/faiss_index/src/util /home/slh/faiss_index/build /home/slh/faiss_index/build/src/util /home/slh/faiss_index/build/src/util/CMakeFiles/faissSystem.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/util/CMakeFiles/faissSystem.dir/depend

