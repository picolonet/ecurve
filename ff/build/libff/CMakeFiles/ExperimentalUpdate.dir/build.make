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
CMAKE_SOURCE_DIR = /home/arunesh/github/ecurve/ff

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arunesh/github/ecurve/ff/build

# Utility rule file for ExperimentalUpdate.

# Include the progress variables for this target.
include libff/CMakeFiles/ExperimentalUpdate.dir/progress.make

libff/CMakeFiles/ExperimentalUpdate:
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/ctest -D ExperimentalUpdate

ExperimentalUpdate: libff/CMakeFiles/ExperimentalUpdate
ExperimentalUpdate: libff/CMakeFiles/ExperimentalUpdate.dir/build.make

.PHONY : ExperimentalUpdate

# Rule to build all files generated by this target.
libff/CMakeFiles/ExperimentalUpdate.dir/build: ExperimentalUpdate

.PHONY : libff/CMakeFiles/ExperimentalUpdate.dir/build

libff/CMakeFiles/ExperimentalUpdate.dir/clean:
	cd /home/arunesh/github/ecurve/ff/build/libff && $(CMAKE_COMMAND) -P CMakeFiles/ExperimentalUpdate.dir/cmake_clean.cmake
.PHONY : libff/CMakeFiles/ExperimentalUpdate.dir/clean

libff/CMakeFiles/ExperimentalUpdate.dir/depend:
	cd /home/arunesh/github/ecurve/ff/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arunesh/github/ecurve/ff /home/arunesh/github/ecurve/ff/libff /home/arunesh/github/ecurve/ff/build /home/arunesh/github/ecurve/ff/build/libff /home/arunesh/github/ecurve/ff/build/libff/CMakeFiles/ExperimentalUpdate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libff/CMakeFiles/ExperimentalUpdate.dir/depend

