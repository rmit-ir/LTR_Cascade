EXECUTE_PROCESS(COMMAND git submodule update --init
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
                OUTPUT_QUIET
        )

# Indri
externalproject_add(indri_proj
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/indri
    CONFIGURE_COMMAND ./configure --prefix=${CMAKE_CURRENT_SOURCE_DIR}/external/local
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    )
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/contrib/antlr/obj)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/contrib/lemur/obj)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/contrib/xpdf/obj)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/obj)
set(INDRI_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/contrib/antlr/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/contrib/lemur/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/contrib/xpdf/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/indri/contrib/zlib/include
    )
include_directories(${INDRI_INCLUDE_DIRS})
set(INDRI_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/indri)
add_library(indri STATIC IMPORTED)
set_target_properties(indri PROPERTIES IMPORTED_LOCATION ${INDRI_LIB_DIR}/obj/libindri.a)
add_library(lemur STATIC IMPORTED)
set_target_properties(lemur PROPERTIES IMPORTED_LOCATION ${INDRI_LIB_DIR}/contrib/lemur/obj/liblemur.a)
add_library(antlr STATIC IMPORTED)
set_target_properties(antlr PROPERTIES IMPORTED_LOCATION ${INDRI_LIB_DIR}/contrib/antlr/obj/libantlr.a)
add_library(xpdf STATIC IMPORTED)
set_target_properties(xpdf PROPERTIES IMPORTED_LOCATION ${INDRI_LIB_DIR}/contrib/xpdf/obj/libxpdf.a)
# binaries that link libindri.a need these flags set
set(INDRI_DEP_FLAGS "-DHAVE_EXT_ATOMICITY=1 -DP_NEEDS_GNU_CXX_NAMESPACE=1")

set(CEREAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/cereal/include)

set(CLI11_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/CLI11/include)


set(FASTPFOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/headers)
add_library(FastPFor STATIC ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/bitpacking.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/bitpacking.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/bitpackingaligned.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/bitpackingunaligned.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/horizontalbitpacking.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/simdunalignedbitpacking.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/simdbitpacking.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/varintdecode.c
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/src/streamvbyte.c
                                ${FASTPFOR_INCLUDE_DIR}
                                )
