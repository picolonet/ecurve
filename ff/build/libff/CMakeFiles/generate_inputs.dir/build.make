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

# Include any dependencies generated for this target.
include libff/CMakeFiles/generate_inputs.dir/depend.make

# Include the progress variables for this target.
include libff/CMakeFiles/generate_inputs.dir/progress.make

# Include the compile flags for this target's objects.
include libff/CMakeFiles/generate_inputs.dir/flags.make

libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o: libff/CMakeFiles/generate_inputs.dir/flags.make
libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o: ../libff/generate_inputs.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/arunesh/github/ecurve/ff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o -c /home/arunesh/github/ecurve/ff/libff/generate_inputs.cpp

libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/generate_inputs.dir/generate_inputs.cpp.i"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arunesh/github/ecurve/ff/libff/generate_inputs.cpp > CMakeFiles/generate_inputs.dir/generate_inputs.cpp.i

libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/generate_inputs.dir/generate_inputs.cpp.s"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arunesh/github/ecurve/ff/libff/generate_inputs.cpp -o CMakeFiles/generate_inputs.dir/generate_inputs.cpp.s

libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.requires:

.PHONY : libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.requires

libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.provides: libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.requires
	$(MAKE) -f libff/CMakeFiles/generate_inputs.dir/build.make libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.provides.build
.PHONY : libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.provides

libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.provides.build: libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o


# Object files for target generate_inputs
generate_inputs_OBJECTS = \
"CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o"

# External object files for target generate_inputs
generate_inputs_EXTERNAL_OBJECTS =

libff/generate_inputs: libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o
libff/generate_inputs: libff/CMakeFiles/generate_inputs.dir/build.make
libff/generate_inputs: libff/libff.a
libff/generate_inputs: /home/arunesh/anaconda3/lib/libgmp.so
libff/generate_inputs: /home/arunesh/anaconda3/lib/libgmpxx.so
libff/generate_inputs: depends/libzm.a
libff/generate_inputs: libff/CMakeFiles/generate_inputs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/arunesh/github/ecurve/ff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable generate_inputs"
	cd /home/arunesh/github/ecurve/ff/build/libff && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/generate_inputs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libff/CMakeFiles/generate_inputs.dir/build: libff/generate_inputs

.PHONY : libff/CMakeFiles/generate_inputs.dir/build

libff/CMakeFiles/generate_inputs.dir/requires: libff/CMakeFiles/generate_inputs.dir/generate_inputs.cpp.o.requires

.PHONY : libff/CMakeFiles/generate_inputs.dir/requires

libff/CMakeFiles/generate_inputs.dir/clean:
	cd /home/arunesh/github/ecurve/ff/build/libff && $(CMAKE_COMMAND) -P CMakeFiles/generate_inputs.dir/cmake_clean.cmake
.PHONY : libff/CMakeFiles/generate_inputs.dir/clean

libff/CMakeFiles/generate_inputs.dir/depend:
	cd /home/arunesh/github/ecurve/ff/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arunesh/github/ecurve/ff /home/arunesh/github/ecurve/ff/libff /home/arunesh/github/ecurve/ff/build /home/arunesh/github/ecurve/ff/build/libff /home/arunesh/github/ecurve/ff/build/libff/CMakeFiles/generate_inputs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libff/CMakeFiles/generate_inputs.dir/depend

