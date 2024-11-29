cmake_minimum_required(VERSION 3.15...3.26)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(density-calculator LANGUAGES CXX)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_C_COMPILER "gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT SKBUILD)
    message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve -Ceditable.rebuild=true .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

set(WRK_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(${WRK_DIR}/clipper/gemmi)
add_subdirectory(${WRK_DIR}/clipper/minimol)
add_subdirectory(${WRK_DIR}/clipper/core)
add_subdirectory(${WRK_DIR}/clipper/contrib)
add_subdirectory(${WRK_DIR}/clipper/cns)
add_subdirectory(${WRK_DIR}/clipper/mmdb)
add_subdirectory(${WRK_DIR}/clipper/cif)
add_subdirectory(${WRK_DIR}/clipper/phs)
add_subdirectory(${WRK_DIR}/clipper/ccp4)
add_subdirectory(${WRK_DIR}/fftw)
add_subdirectory(${WRK_DIR}/rfftw)
add_subdirectory(${WRK_DIR}/ccp4)
add_subdirectory(${WRK_DIR}/mmdb2)
add_subdirectory(${WRK_DIR}/gemmi)

message(CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${WRK_DIR}/checkout/mmdb2
        ${WRK_DIR}/checkout/clipper
        ${WRK_DIR}/checkout/libccp4/ccp4
        ${WRK_DIR}/checkout/gemmi/include
)

link_directories(
        ${WRK_DIR}/mmdb2
        ${WRK_DIR}/clipper
        ${WRK_DIR}/ccp4
        ${WRK_DIR}/src
        ${CMAKE_BINARY_DIR}
)

set(LINK_LIBS "gemmi_cpp nautilus_lib ccp4 fftw rfftw mmdb2 clipper-mmdb clipper-cif clipper-core clipper-contrib clipper-minimol clipper-ccp4 clipper-gemmi")

if (SKBUILD)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
        density_calculator

        STABLE_ABI
        NB_STATIC

        # Python bound functions here
        ${WRK_DIR}/src/cpp/difference-density.cpp
)


target_include_directories(density_calculator PUBLIC
        ${WRK_DIR}/checkout/gemmi
        ${WRK_DIR}/mmdb2
        ${WRK_DIR}/clipper/mmdb
        ${WRK_DIR}/clipper/cif
        ${WRK_DIR}/clipper/core
        ${WRK_DIR}/clipper/contrib
        ${WRK_DIR}/clipper/ccp4
        ${WRK_DIR}/clipper/minimol
        ${WRK_DIR}/clipper/gemmi
        ${WRK_DIR}/fftw
        ${WRK_DIR}/rfftw
        ${WRK_DIR}/ccp4

)


target_link_directories(density_calculator
        PUBLIC
        ${CMAKE_BINARY_DIR}/clipper/mmdb
        ${CMAKE_BINARY_DIR}/clipper/cif
        ${CMAKE_BINARY_DIR}/clipper/ccp4
        ${CMAKE_BINARY_DIR}/clipper/minimol
        ${CMAKE_BINARY_DIR}/clipper/contrib
        ${CMAKE_BINARY_DIR}/clipper/core
        ${CMAKE_BINARY_DIR}/clipper/gemmi
        ${CMAKE_BINARY_DIR}/fftw
        ${CMAKE_BINARY_DIR}/rfftw
        ${CMAKE_BINARY_DIR}/ccp4
        ${CMAKE_BINARY_DIR}/mmdb2
        ${CMAKE_BINARY_DIR}/gemmi
)


target_link_libraries(
        density_calculator
        PRIVATE
        # nautilus_lib
        clipper-gemmi
        clipper-mmdb
        clipper-cif
        clipper-ccp4
        clipper-minimol
        clipper-contrib
        clipper-core
        rfftw
        fftw
        ccp4
        mmdb2
        gemmi_cpp
)
# Install directive for scikit-build-core

install(TARGETS density_calculator LIBRARY DESTINATION density_calculator)
endif()

if (CMAKE_C_PLATFORM_ID STREQUAL "emscripten")
#    set_target_properties( density_calculator
#            PROPERTIES
#            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/wasm"
#            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/wasm"
#            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/wasm"
#    )

    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../webapp)
    add_executable(density_calculator ${WRK_DIR}/src/wasm/density-calculator-wasm.cpp)
    set_target_properties (density_calculator  PROPERTIES OUTPUT_NAME density_calculator )
    set_target_properties(density_calculator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../webapp)
    target_include_directories(density_calculator PUBLIC
            ${WRK_DIR}/checkout/gemmi
            ${WRK_DIR}/mmdb2
            ${WRK_DIR}/clipper/mmdb
            ${WRK_DIR}/clipper/cif
            ${WRK_DIR}/clipper/core
            ${WRK_DIR}/clipper/contrib
            ${WRK_DIR}/clipper/ccp4
            ${WRK_DIR}/clipper/minimol
            ${WRK_DIR}/clipper/gemmi
            ${WRK_DIR}/fftw
            ${WRK_DIR}/rfftw
            ${WRK_DIR}/ccp4
    )


    target_link_directories(density_calculator
            PUBLIC
            ${CMAKE_BINARY_DIR}/clipper/mmdb
            ${CMAKE_BINARY_DIR}/clipper/cif
            ${CMAKE_BINARY_DIR}/clipper/ccp4
            ${CMAKE_BINARY_DIR}/clipper/minimol
            ${CMAKE_BINARY_DIR}/clipper/contrib
            ${CMAKE_BINARY_DIR}/clipper/core
            ${CMAKE_BINARY_DIR}/clipper/gemmi
            ${CMAKE_BINARY_DIR}/fftw
            ${CMAKE_BINARY_DIR}/rfftw
            ${CMAKE_BINARY_DIR}/ccp4
            ${CMAKE_BINARY_DIR}/mmdb2
            ${CMAKE_BINARY_DIR}/gemmi
    )


    target_link_libraries(
            density_calculator
            PRIVATE
            clipper-gemmi
            clipper-mmdb
            clipper-cif
            clipper-ccp4
            clipper-minimol
            clipper-contrib
            clipper-core
            rfftw
            fftw
            ccp4
            mmdb2
            gemmi_cpp
    )

    target_link_options(density_calculator PRIVATE
            #                --preload-file data/linkage_torsions@/
            -sEXIT_RUNTIME=1 -sASSERTIONS=2 -fexceptions
            -sALLOW_MEMORY_GROWTH=1 --bind -sFORCE_FILESYSTEM=1 -sMODULARIZE=1
            -sEXPORT_NAME=sails_module -sEXPORTED_RUNTIME_METHODS=['FS']
            -sMODULARIZE=1 -sEXPORT_ES6=1 -sSTACK_SIZE=8000000 )

    set_target_properties(density_calculator PROPERTIES COMPILE_FLAGS "-s ENVIRONMENT='web'")


endif ()