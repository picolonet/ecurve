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
include libff/CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include libff/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include libff/CMakeFiles/main.dir/flags.make

libff/CMakeFiles/main.dir/main.cpp.o: libff/CMakeFiles/main.dir/flags.make
libff/CMakeFiles/main.dir/main.cpp.o: ../libff/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/arunesh/github/ecurve/ff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libff/CMakeFiles/main.dir/main.cpp.o"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.cpp.o -c /home/arunesh/github/ecurve/ff/libff/main.cpp

libff/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arunesh/github/ecurve/ff/libff/main.cpp > CMakeFiles/main.dir/main.cpp.i

libff/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /home/arunesh/github/ecurve/ff/build/libff && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arunesh/github/ecurve/ff/libff/main.cpp -o CMakeFiles/main.dir/main.cpp.s

libff/CMakeFiles/main.dir/main.cpp.o.requires:

.PHONY : libff/CMakeFiles/main.dir/main.cpp.o.requires

libff/CMakeFiles/main.dir/main.cpp.o.provides: libff/CMakeFiles/main.dir/main.cpp.o.requires
	$(MAKE) -f libff/CMakeFiles/main.dir/build.make libff/CMakeFiles/main.dir/main.cpp.o.provides.build
.PHONY : libff/CMakeFiles/main.dir/main.cpp.o.provides

libff/CMakeFiles/main.dir/main.cpp.o.provides.build: libff/CMakeFiles/main.dir/main.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

libff/main: libff/CMakeFiles/main.dir/main.cpp.o
libff/main: libff/CMakeFiles/main.dir/build.make
libff/main: libff/libff.a
libff/main: /home/arunesh/anaconda3/lib/libgmp.so
libff/main: /home/arunesh/anaconda3/lib/libgmpxx.so
libff/main: depends/libzm.a
libff/main: libff/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/arunesh/github/ecurve/ff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main"
	cd /home/arunesh/github/ecurve/ff/build/libff && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libff/CMakeFiles/main.dir/build: libff/main

.PHONY : libff/CMakeFiles/main.dir/build

libff/CMakeFiles/main.dir/requires: libff/CMakeFiles/main.dir/main.cpp.o.requires

.PHONY : libff/CMakeFiles/main.dir/requires

libff/CMakeFiles/main.dir/clean:
	cd /home/arunesh/github/ecurve/ff/build/libff && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : libff/CMakeFiles/main.dir/clean

libff/CMakeFiles/main.dir/depend:
	cd /home/arunesh/github/ecurve/ff/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arunesh/github/ecurve/ff /home/arunesh/github/ecurve/ff/libff /home/arunesh/github/ecurve/ff/build /home/arunesh/github/ecurve/ff/build/libff /home/arunesh/github/ecurve/ff/build/libff/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libff/CMakeFiles/main.dir/depend

