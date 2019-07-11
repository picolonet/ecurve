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
include libff/CMakeFiles/algebra_bilinearity_test.dir/depend.make

# Include the progress variables for this target.
include libff/CMakeFiles/algebra_bilinearity_test.dir/progress.make

# Include the compile flags for this target's objects.
include libff/CMakeFiles/algebra_bilinearity_test.dir/flags.make

libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o: libff/CMakeFiles/algebra_bilinearity_test.dir/flags.make
libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o: ../libff/algebra/curves/tests/test_bilinearity.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/arunesh/github/ecurve/ff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o -c /home/arunesh/github/ecurve/ff/libff/algebra/curves/tests/test_bilinearity.cpp

libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.i"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arunesh/github/ecurve/ff/libff/algebra/curves/tests/test_bilinearity.cpp > CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.i

libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.s"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arunesh/github/ecurve/ff/libff/algebra/curves/tests/test_bilinearity.cpp -o CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.s

libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.requires:

.PHONY : libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.requires

libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.provides: libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.requires
	$(MAKE) -f libff/CMakeFiles/algebra_bilinearity_test.dir/build.make libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.provides.build
.PHONY : libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.provides

libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.provides.build: libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o


# Object files for target algebra_bilinearity_test
algebra_bilinearity_test_OBJECTS = \
"CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o"

# External object files for target algebra_bilinearity_test
algebra_bilinearity_test_EXTERNAL_OBJECTS =

libff/algebra_bilinearity_test: libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o
libff/algebra_bilinearity_test: libff/CMakeFiles/algebra_bilinearity_test.dir/build.make
libff/algebra_bilinearity_test: libff/libff.a
libff/algebra_bilinearity_test: /home/arunesh/anaconda3/lib/libgmp.so
libff/algebra_bilinearity_test: /home/arunesh/anaconda3/lib/libgmpxx.so
libff/algebra_bilinearity_test: depends/libzm.a
libff/algebra_bilinearity_test: libff/CMakeFiles/algebra_bilinearity_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/arunesh/github/ecurve/ff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable algebra_bilinearity_test"
	cd /home/arunesh/github/ecurve/ff/build/libff && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/algebra_bilinearity_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libff/CMakeFiles/algebra_bilinearity_test.dir/build: libff/algebra_bilinearity_test

.PHONY : libff/CMakeFiles/algebra_bilinearity_test.dir/build

libff/CMakeFiles/algebra_bilinearity_test.dir/requires: libff/CMakeFiles/algebra_bilinearity_test.dir/algebra/curves/tests/test_bilinearity.cpp.o.requires

.PHONY : libff/CMakeFiles/algebra_bilinearity_test.dir/requires

libff/CMakeFiles/algebra_bilinearity_test.dir/clean:
	cd /home/arunesh/github/ecurve/ff/build/libff && $(CMAKE_COMMAND) -P CMakeFiles/algebra_bilinearity_test.dir/cmake_clean.cmake
.PHONY : libff/CMakeFiles/algebra_bilinearity_test.dir/clean

libff/CMakeFiles/algebra_bilinearity_test.dir/depend:
	cd /home/arunesh/github/ecurve/ff/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arunesh/github/ecurve/ff /home/arunesh/github/ecurve/ff/libff /home/arunesh/github/ecurve/ff/build /home/arunesh/github/ecurve/ff/build/libff /home/arunesh/github/ecurve/ff/build/libff/CMakeFiles/algebra_bilinearity_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libff/CMakeFiles/algebra_bilinearity_test.dir/depend

