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
include src/util/CMakeFiles/extract_feature.dir/depend.make

# Include the progress variables for this target.
include src/util/CMakeFiles/extract_feature.dir/progress.make

# Include the compile flags for this target's objects.
include src/util/CMakeFiles/extract_feature.dir/flags.make

src/util/CMakeFiles/extract_feature.dir/test.cpp.o: src/util/CMakeFiles/extract_feature.dir/flags.make
src/util/CMakeFiles/extract_feature.dir/test.cpp.o: ../src/util/test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/slh/faiss_index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/util/CMakeFiles/extract_feature.dir/test.cpp.o"
	cd /home/slh/faiss_index/build/src/util && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/extract_feature.dir/test.cpp.o -c /home/slh/faiss_index/src/util/test.cpp

src/util/CMakeFiles/extract_feature.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/extract_feature.dir/test.cpp.i"
	cd /home/slh/faiss_index/build/src/util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/slh/faiss_index/src/util/test.cpp > CMakeFiles/extract_feature.dir/test.cpp.i

src/util/CMakeFiles/extract_feature.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/extract_feature.dir/test.cpp.s"
	cd /home/slh/faiss_index/build/src/util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/slh/faiss_index/src/util/test.cpp -o CMakeFiles/extract_feature.dir/test.cpp.s

src/util/CMakeFiles/extract_feature.dir/test.cpp.o.requires:

.PHONY : src/util/CMakeFiles/extract_feature.dir/test.cpp.o.requires

src/util/CMakeFiles/extract_feature.dir/test.cpp.o.provides: src/util/CMakeFiles/extract_feature.dir/test.cpp.o.requires
	$(MAKE) -f src/util/CMakeFiles/extract_feature.dir/build.make src/util/CMakeFiles/extract_feature.dir/test.cpp.o.provides.build
.PHONY : src/util/CMakeFiles/extract_feature.dir/test.cpp.o.provides

src/util/CMakeFiles/extract_feature.dir/test.cpp.o.provides.build: src/util/CMakeFiles/extract_feature.dir/test.cpp.o


# Object files for target extract_feature
extract_feature_OBJECTS = \
"CMakeFiles/extract_feature.dir/test.cpp.o"

# External object files for target extract_feature
extract_feature_EXTERNAL_OBJECTS =

../bin/extract_feature: src/util/CMakeFiles/extract_feature.dir/test.cpp.o
../bin/extract_feature: src/util/CMakeFiles/extract_feature.dir/build.make
../bin/extract_feature: ../../caffe-ssd/build/lib/libcaffe.so
../bin/extract_feature: src/util/CMakeFiles/extract_feature.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/slh/faiss_index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/extract_feature"
	cd /home/slh/faiss_index/build/src/util && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/extract_feature.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/util/CMakeFiles/extract_feature.dir/build: ../bin/extract_feature

.PHONY : src/util/CMakeFiles/extract_feature.dir/build

src/util/CMakeFiles/extract_feature.dir/requires: src/util/CMakeFiles/extract_feature.dir/test.cpp.o.requires

.PHONY : src/util/CMakeFiles/extract_feature.dir/requires

src/util/CMakeFiles/extract_feature.dir/clean:
	cd /home/slh/faiss_index/build/src/util && $(CMAKE_COMMAND) -P CMakeFiles/extract_feature.dir/cmake_clean.cmake
.PHONY : src/util/CMakeFiles/extract_feature.dir/clean

src/util/CMakeFiles/extract_feature.dir/depend:
	cd /home/slh/faiss_index/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/slh/faiss_index /home/slh/faiss_index/src/util /home/slh/faiss_index/build /home/slh/faiss_index/build/src/util /home/slh/faiss_index/build/src/util/CMakeFiles/extract_feature.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/util/CMakeFiles/extract_feature.dir/depend

